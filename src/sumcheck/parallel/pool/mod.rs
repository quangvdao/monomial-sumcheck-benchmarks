//! Process-global persistent worker pool with doorbell synchronization.
//!
//! Each worker spins on a Release/Acquire epoch counter, runs the
//! published task, and Release-bumps a `done` counter. The main thread
//! (worker 0) publishes a task by storing a closure pointer into a
//! cell, Release-incrementing the epoch, executing the task itself,
//! then waiting on the `done` counter for the active extras.
//!
//! ### Why this exists
//!
//! Rayon and chili pay 1-3 µs per fork/join scope at small workloads,
//! which is comparable to one full reduce-bind round of a small
//! sumcheck. By pinning a fixed pool to P-cores (via QoS on macOS or
//! `sched_setaffinity` on Linux) and using a pure-spin doorbell, we
//! drop the per-broadcast cost to ~300 ns, making parallelism net-win
//! at `n ≥ 12` on Apple M4. See `PARALLELISM.md` and
//! `docs/notes/parallelism-design-discussion.md` for the full design.
//!
//! ### Lifetime safety of the broadcast closure
//!
//! [`PinnedPool::broadcast_scoped`] launders the caller-scoped closure
//! into a `'static` reference via `transmute`. This is sound because
//! the function blocks until every active worker has returned from
//! the closure, so the closure outlives all worker accesses. The
//! `n_active` field tells inactive workers to skip both the closure
//! and the `done` counter, so they cannot keep a stale pointer alive.

mod platform;

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

/// Cache-line padding to keep hot atomics from false-sharing with each
/// other. 128 bytes covers typical sector-prefetch pairs (two 64 B
/// lines on most cores).
#[repr(align(128))]
struct CachePadded<T>(T);

type TaskPtr = *const (dyn Fn(usize) + Sync + 'static);

struct PoolShared {
    /// Monotonically increasing epoch. Main Release-bumps it to publish
    /// a new task; workers Acquire-spin waiting for `epoch > last_seen`.
    epoch: CachePadded<AtomicU64>,
    /// Count of extra (non-main) active workers that have finished the
    /// current task. Only workers with `worker_idx < n_active` bump it.
    done: CachePadded<AtomicUsize>,
    /// Number of workers active for the current dispatch, including
    /// main (always `1..=n_total`). Set by main BEFORE bumping epoch;
    /// read by workers AFTER observing the epoch bump. Workers with
    /// `worker_idx >= n_active` skip both `task` and `done` for this
    /// dispatch (they just advance `last_epoch` and keep spinning).
    n_active: AtomicUsize,
    /// Shutdown flag set on drop.
    shutdown: AtomicBool,
    /// Scoped task pointer, published via the Release-bump on `epoch`.
    /// `None` during startup/shutdown.
    task: UnsafeCell<Option<TaskPtr>>,
}

// SAFETY: the only `!Send`/`!Sync` field is the
// `UnsafeCell<Option<TaskPtr>>` raw pointer. Access to it is protected
// by the epoch Release/Acquire pair and the done-counter broadcast
// barrier; no worker touches the pointer outside a dispatch window.
unsafe impl Send for PoolShared {}
unsafe impl Sync for PoolShared {}

/// A process-global pool of pinned worker threads.
///
/// Construct via [`PinnedPool::global`] for the standard configuration
/// (lazy, `min(available_parallelism, 8)` workers, overridable by the
/// `SUMCHECK_PINNED_WORKERS` env var). Each `broadcast_scoped` call
/// runs `f` on the requested number of workers in parallel and blocks
/// until every active worker has returned.
pub struct PinnedPool {
    n_total: usize,
    shared: Arc<PoolShared>,
    workers: Vec<JoinHandle<()>>,
}

impl PinnedPool {
    /// Total number of workers available, including main (main = worker
    /// 0). `broadcast_scoped` accepts `n_active` in `1..=n_workers()`.
    #[inline]
    pub fn n_workers(&self) -> usize {
        self.n_total
    }

    /// Returns the process-global pinned worker pool, spawning it on
    /// first use. Worker count defaults to
    /// `min(available_parallelism, 8)` and can be overridden by the
    /// `SUMCHECK_PINNED_WORKERS` environment variable.
    pub fn global() -> &'static PinnedPool {
        static PINNED_POOL: OnceLock<PinnedPool> = OnceLock::new();
        PINNED_POOL.get_or_init(|| {
            let n_total = std::env::var("SUMCHECK_PINNED_WORKERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|p| p.get().min(8))
                        .unwrap_or(4)
                })
                .max(1);
            let n_extra = n_total.saturating_sub(1);
            PinnedPool::new(n_extra)
        })
    }

    fn new(n_extra: usize) -> Self {
        let shared = Arc::new(PoolShared {
            epoch: CachePadded(AtomicU64::new(0)),
            done: CachePadded(AtomicUsize::new(0)),
            n_active: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            task: UnsafeCell::new(None),
        });

        let mut workers = Vec::with_capacity(n_extra);
        for extra_idx in 0..n_extra {
            let worker_idx = extra_idx + 1;
            let shared_clone = shared.clone();
            let handle = thread::Builder::new()
                .name(format!("sumcheck-pinned-{worker_idx}"))
                .spawn(move || pinned_worker_loop(shared_clone, worker_idx))
                .expect("failed to spawn pinned worker thread");
            workers.push(handle);
        }
        PinnedPool {
            n_total: n_extra + 1,
            shared,
            workers,
        }
    }

    /// Broadcast `f` to `n_active` workers: main (worker 0) plus
    /// `n_active - 1` extras. Inactive extras skip `f` entirely and do
    /// not touch the done counter, so the barrier contention cost
    /// scales with `n_active` rather than the full pool size. Blocks
    /// until every active worker has returned from `f`.
    ///
    /// `n_active` must be in `1..=self.n_workers()`.
    pub fn broadcast_scoped(&self, n_active: usize, f: &(dyn Fn(usize) + Sync)) {
        assert!(
            (1..=self.n_total).contains(&n_active),
            "n_active={n_active} out of range 1..={}",
            self.n_total
        );
        let n_extra_active = n_active - 1;

        // Reset the done counter. Safe because the previous broadcast
        // blocked until done reached its target; workers Release-bumped
        // strictly after finishing `task()`.
        self.shared.done.0.store(0, Ordering::Relaxed);

        if n_extra_active == 0 && self.n_total == 1 {
            // No extras at all: just run on main.
            f(0);
            return;
        }

        // Publish n_active before the task pointer; both are
        // synchronized via the Release-bump on `epoch` below.
        self.shared.n_active.store(n_active, Ordering::Relaxed);

        // Launder the scoped lifetime. SAFETY: we block until every
        // active worker finishes `task()`, so `f` outlives all
        // accesses.
        let f_static: &(dyn Fn(usize) + Sync + 'static) = unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), &(dyn Fn(usize) + Sync + 'static)>(f)
        };
        unsafe {
            *self.shared.task.get() = Some(f_static as TaskPtr);
        }

        // Release-bump the epoch. Publishes the writes above to any
        // worker that subsequently Acquire-loads the epoch.
        self.shared.epoch.0.fetch_add(1, Ordering::Release);

        // Main runs as worker 0.
        f(0);

        // Wait for active extra workers to Release-bump `done`.
        if n_extra_active > 0 {
            spin_until_ge(&self.shared.done.0, n_extra_active);
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Wake spinners. They'll observe `shutdown` after Acquire-loading
        // the bumped epoch and return.
        self.shared.epoch.0.fetch_add(1, Ordering::Release);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn pinned_worker_loop(shared: Arc<PoolShared>, worker_idx: usize) {
    platform::pin_current_worker(worker_idx);

    let mut last_epoch: u64 = 0;
    loop {
        // Spin-wait for a new epoch or shutdown. With `pin_current_worker`
        // applied, the OS keeps us on a fast core so pure `spin_loop`
        // is safe. Shutdown is checked on every iteration.
        let new_epoch = loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let e = shared.epoch.0.load(Ordering::Acquire);
            if e > last_epoch {
                break e;
            }
            std::hint::spin_loop();
        };
        last_epoch = new_epoch;

        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        // Synchronized with the Release on `epoch`: n_active and task
        // are both visible. If this worker is inactive for the current
        // dispatch, skip cleanly without touching `done`.
        let n_active = shared.n_active.load(Ordering::Relaxed);
        if worker_idx >= n_active {
            continue;
        }

        let task_opt: Option<TaskPtr> = unsafe { *shared.task.get() };
        if let Some(task_ptr) = task_opt {
            let task: &(dyn Fn(usize) + Sync + 'static) = unsafe { &*task_ptr };
            task(worker_idx);
        }

        // Release-signal completion.
        shared.done.0.fetch_add(1, Ordering::Release);
    }
}

/// Bounded spin then `yield_now` fallback. Used by `broadcast_scoped`
/// to wait on the `done` counter, where the calling thread is whatever
/// thread invoked the broadcast (not necessarily a pinned worker), so
/// pure spinning could starve other tasks if the OS preempts us
/// mid-spin. The 2048-iteration budget corresponds to roughly 1 µs of
/// spinning on a 3-4 GHz core before the first yield.
#[inline]
fn spin_until_ge(counter: &AtomicUsize, target: usize) {
    const SPIN_BUDGET: u32 = 2048;
    let mut spins = 0u32;
    loop {
        if counter.load(Ordering::Acquire) >= target {
            return;
        }
        if spins < SPIN_BUDGET {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
            spins = 0;
        }
    }
}
