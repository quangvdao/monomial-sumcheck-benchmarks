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
//! - **Linux**: stub for Phase A. The full implementation will use
//!   `sched_setaffinity` to pin to one logical thread per physical
//!   core, optionally elevating to `SCHED_FIFO` if the process holds
//!   `CAP_SYS_NICE` (see Phase A task a9 in the productionization
//!   plan). Until then, the spin-then-yield fallback in
//!   [`super::spin_until_ge`] keeps things correct.
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
    linux::pin_to_physical_core(worker_idx);
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
    /// Phase-A stub. The real implementation (task a9 in the
    /// productionization plan) will:
    /// 1. Parse `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`
    ///    to enumerate physical cores.
    /// 2. Pick the `worker_idx`-th physical core and call
    ///    `sched_setaffinity(0, sizeof(set), &set)`.
    /// 3. Optionally elevate to `SCHED_FIFO` if `CAP_SYS_NICE` is held.
    /// 4. Warn (not fail) if the cpufreq governor is not `performance`.
    pub(super) fn pin_to_physical_core(_worker_idx: usize) {
        // No-op for Phase A. The spin-then-yield fallback in
        // `spin_until_ge` keeps correctness without pinning, at a small
        // throughput cost from possible OS migration.
    }
}
