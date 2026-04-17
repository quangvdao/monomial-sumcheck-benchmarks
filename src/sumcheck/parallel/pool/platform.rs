//! Platform-specific worker pinning and QoS hints.
//!
//! Goal: keep `PinnedPool` worker threads on a single high-performance
//! core each so that the pure-spin doorbell loop in
//! [`super::pinned_worker_loop`] is not preempted or demoted to an
//! efficiency core. Each backend is a best-effort no-op on failure;
//! losing pinning is a perf bug, not a correctness bug.
//!
//! - **macOS**: tag each worker with `QOS_CLASS_USER_INTERACTIVE` so
//!   the scheduler keeps it on a P-core. Affinity APIs are not exposed
//!   to user space on Apple Silicon; QoS is the canonical mechanism.
//! - **Linux**: `sched_setaffinity` to one logical thread per physical
//!   core (parsed from `/sys/devices/system/cpu/cpu*/topology/`).
//!   Optionally `sched_setscheduler(SCHED_FIFO, 1)` if the user sets
//!   `SUMCHECK_PINNED_SCHED_FIFO=1` AND the process holds
//!   `CAP_SYS_NICE` (we just attempt the call and silently fall back
//!   on `EPERM`). SCHED_FIFO is off by default because it can starve
//!   other threads on a busy system if our workers are spinning.
//! - **All other targets** (BSD, Windows, WASM): no-op.

/// Apply the platform's strongest "stay on a fast core, don't demote me"
/// hint. Called once per worker thread as the first thing in the loop.
///
/// `worker_idx` is the 1-based worker index (worker 0 is main, which
/// does not run this loop). Backends may use it to pick a per-worker
/// affinity mask.
#[inline]
pub(super) fn pin_current_worker(worker_idx: usize) {
    let _ = worker_idx;

    #[cfg(target_os = "macos")]
    macos::set_qos_user_interactive();

    #[cfg(target_os = "linux")]
    linux::pin_current(worker_idx);
}

#[cfg(target_os = "macos")]
mod macos {
    /// `QOS_CLASS_USER_INTERACTIVE` from `<pthread/qos.h>`.
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }

    pub(super) fn set_qos_user_interactive() {
        // SAFETY: `pthread_set_qos_class_self_np` is safe to call from any
        // thread; it has no side effects beyond the calling thread's QoS.
        unsafe {
            let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    //! Linux pinning backend.
    //!
    //! Strategy: at first call we discover one canonical logical CPU
    //! per physical core from `/sys/devices/system/cpu/cpu*/topology/
    //! thread_siblings_list`. Each sibling group (= physical core) is
    //! represented by the smallest logical CPU id in its list, which
    //! is stable across boots and matches the convention Linux
    //! userspace tools (e.g. `lscpu -p`) use. We then cache the
    //! deduplicated, sorted list in a `OnceLock<Vec<usize>>`.
    //!
    //! On a box with SMT enabled (Zen 5 aragorn = 16 cores × 2 SMT =
    //! 32 threads), this yields 16 physical CPUs; we pin worker `k`
    //! (1..) to `physical_cpus[k % physical_cpus.len()]`. That keeps
    //! our spinning workers from fighting SMT siblings of the same
    //! physical core, which would cause cache-port contention.
    //!
    //! All failures (sysfs missing, syscall EPERM, parse errors) are
    //! silent no-ops: correctness is unaffected (the kernel just
    //! schedules us normally). Set `SUMCHECK_PINNED_DEBUG=1` at
    //! startup to get one stderr line per worker describing what we
    //! pinned to.
    use std::fs;
    use std::io;
    use std::sync::OnceLock;

    pub(super) fn pin_current(worker_idx: usize) {
        let cpus = physical_cpus();
        if cpus.is_empty() {
            // Fallback: let the kernel schedule us. We still try
            // SCHED_FIFO in case the user asked for it.
            maybe_set_fifo(worker_idx, None);
            return;
        }
        let cpu = cpus[worker_idx % cpus.len()];
        let pinned = match pin_to(cpu) {
            Ok(()) => Some(cpu),
            Err(_) => None,
        };
        maybe_set_fifo(worker_idx, pinned);
    }

    /// Cache of `{ canonical logical cpu id : one per physical core }`,
    /// sorted ascending. Empty if sysfs could not be read (e.g.
    /// containers without `/sys/devices/system/cpu`).
    fn physical_cpus() -> &'static [usize] {
        static PHYSICAL: OnceLock<Vec<usize>> = OnceLock::new();
        PHYSICAL.get_or_init(discover_physical_cpus)
    }

    /// Read `/sys/devices/system/cpu/possible`, then for each listed
    /// logical CPU read its `topology/thread_siblings_list` and keep
    /// the smallest id from each group. Returns a sorted deduplicated
    /// vector of canonical logical CPU ids.
    fn discover_physical_cpus() -> Vec<usize> {
        let possible = match fs::read_to_string("/sys/devices/system/cpu/possible") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let logical = match parse_cpu_list(possible.trim()) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut canonical = Vec::with_capacity(logical.len());
        for cpu in &logical {
            let path = format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            );
            let siblings_raw = match fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => {
                    // Missing topology: treat this cpu as its own
                    // physical core. Better a too-fine-grained mask
                    // than to drop the cpu entirely.
                    canonical.push(*cpu);
                    continue;
                }
            };
            let siblings = match parse_cpu_list(siblings_raw.trim()) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    canonical.push(*cpu);
                    continue;
                }
            };
            // Canonical = smallest sibling, so `core N` gets the same
            // representative regardless of which of its threads is
            // reading the sysfs file.
            canonical.push(*siblings.iter().min().expect("non-empty by guard"));
        }
        canonical.sort_unstable();
        canonical.dedup();
        canonical
    }

    /// Parse a Linux cpuset string of the form `"0-3,5,7-9"` into a
    /// flat list. Returns `None` on any malformed token.
    fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
        let mut out = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                None => out.push(part.parse().ok()?),
                Some((lo, hi)) => {
                    let lo: usize = lo.parse().ok()?;
                    let hi: usize = hi.parse().ok()?;
                    if lo > hi {
                        return None;
                    }
                    out.extend(lo..=hi);
                }
            }
        }
        Some(out)
    }

    /// Invoke `sched_setaffinity(0, sizeof(cpu_set_t), &set)` for the
    /// calling thread with a single bit set for `cpu`.
    fn pin_to(cpu: usize) -> io::Result<()> {
        // SAFETY: cpu_set_t is a plain integer array; zero-init is valid.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_SET(cpu, &mut set) };
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const _,
            )
        };
        if rc == 0 {
            debug_log(&format!("worker pinned to cpu{cpu}"));
            Ok(())
        } else {
            let e = io::Error::last_os_error();
            debug_log(&format!("sched_setaffinity(cpu{cpu}) failed: {e}"));
            Err(e)
        }
    }

    /// Opt-in: when `SUMCHECK_PINNED_SCHED_FIFO=1`, raise the calling
    /// thread to `SCHED_FIFO` priority 1. Silent no-op if the process
    /// doesn't hold `CAP_SYS_NICE` (EPERM) or the env var is unset.
    /// We deliberately use priority 1 (the lowest FIFO priority) so a
    /// user's other realtime threads outrank us.
    fn maybe_set_fifo(worker_idx: usize, pinned_cpu: Option<usize>) {
        if std::env::var("SUMCHECK_PINNED_SCHED_FIFO").as_deref() != Ok("1") {
            return;
        }
        let param = libc::sched_param { sched_priority: 1 };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc == 0 {
            debug_log(&format!(
                "worker {worker_idx} (cpu={pinned_cpu:?}) elevated to SCHED_FIFO prio 1"
            ));
        } else {
            let e = io::Error::last_os_error();
            debug_log(&format!(
                "worker {worker_idx} SCHED_FIFO failed: {e} (need CAP_SYS_NICE?)"
            ));
        }
    }

    /// stderr log gated on `SUMCHECK_PINNED_DEBUG=1`. Evaluated lazily
    /// per call so we don't pay the env-var cost when debugging is
    /// off. One line per worker is fine for a small pool and gives
    /// us enough signal to verify pinning worked on unfamiliar
    /// hardware.
    fn debug_log(msg: &str) {
        if std::env::var("SUMCHECK_PINNED_DEBUG").as_deref() == Ok("1") {
            eprintln!("[sumcheck-pinned] {msg}");
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_simple() {
            assert_eq!(parse_cpu_list("0").unwrap(), vec![0]);
            assert_eq!(parse_cpu_list("0,2,4").unwrap(), vec![0, 2, 4]);
            assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
            assert_eq!(
                parse_cpu_list("0-1,4,7-8").unwrap(),
                vec![0, 1, 4, 7, 8]
            );
        }

        #[test]
        fn parse_empty_and_malformed() {
            assert_eq!(parse_cpu_list("").unwrap(), Vec::<usize>::new());
            assert!(parse_cpu_list("3-1").is_none());
            assert!(parse_cpu_list("abc").is_none());
            assert!(parse_cpu_list("1,abc,3").is_none());
        }
    }
}
