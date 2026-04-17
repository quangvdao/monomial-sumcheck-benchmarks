//! Process-global persistent worker pool with per-worker doorbells.
//!
//! Each extra worker owns a private `WorkerSlot` (cache-line padded)
//! containing its own `assigned_gen`, `done_gen`, and `task` cell.
//! Main (worker 0) dispatches a task by writing the task pointer and
//! bumping `assigned_gen` only on the active extras' slots, runs its
//! own piece of the task, then waits for those extras' `done_gen` to
//! catch up. Inactive extras never wake up: they keep spinning on
//! their own `assigned_gen` and incur zero cost on the broadcasting
//! thread.
//!
//! ### Why per-worker slots
//!
//! A previous design used a single shared `epoch + done` counter and
//! a single shared `task` cell, with all extras polling the same
//! epoch line. Two problems:
//!
//! 1. **Correctness race** with shared task: if some extras were
//!    "inactive" (`worker_idx >= n_active`), they would still
//!    consume the shared epoch but skip the task body. Because
//!    `n_active` and `task` were `Relaxed` writes synchronized only
//!    by the Release on `epoch`, a lagging inactive extra could
//!    read `n_active` from one broadcast and `task` from a later
//!    broadcast (Acquire forbids past-stale reads but not future-
//!    torn reads), then run a closure whose captured environment
//!    had already been dropped, panicking with
//!    `index out of bounds: the len is 0 but the index is N`.
//!
//! 2. **Performance**: forcing inactive extras to ack every epoch
//!    (a fix for #1) makes every broadcast pay for cache-line
//!    ping-pong on the shared `done` counter across all extras,
//!    even when only a few are active.
//!
//! Per-worker slots fix both: each extra writes only its own cache
//! line, main reads only the active extras' lines, inactive extras
//! never participate. Each broadcast at `n_active = k` costs roughly
//! `k` cache-line round trips between main and the active extras.
//!
//! ### Lifetime safety of the broadcast closure
//!
//! [`PinnedPool::broadcast_scoped`] launders the caller-scoped closure
//! into a `'static` reference via `transmute`. This is sound because
//! the function blocks until every active extra has stored its
//! `done_gen >= new_gen`, which happens after that extra's call to
//! the closure has fully returned. Inactive extras never read the
//! closure pointer (they read only their own `assigned_gen`/`task`,
//! and `task` is only written for active extras), so the closure
//! pointer cannot dangle.

mod platform;

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

type TaskPtr = *const (dyn Fn(usize) + Sync + 'static);

/// Per-worker inbox: written by main, read by the worker. Padded so
/// it doesn't false-share with the worker's outbox (`Outbox`), which
/// avoids the cache-line ping-pong that would otherwise happen on
/// every dispatch as ownership bounces between main and worker.
#[repr(align(128))]
struct WorkerInbox {
    /// Generation of the most recently dispatched task for this
    /// worker. Main Release-stores a fresh value to wake the worker;
    /// the worker Acquire-spins on this and runs the task whenever it
    /// observes a value greater than its `last_done`.
    assigned_gen: AtomicU64,
    /// Task pointer for this worker. Main writes it BEFORE
    /// `assigned_gen`; worker reads it AFTER observing the new
    /// `assigned_gen`. Release/Acquire on `assigned_gen` synchronizes
    /// the publication.
    task: UnsafeCell<Option<TaskPtr>>,
}

/// Per-worker outbox: written by worker, read by main. Padded for the
/// same reason as `WorkerInbox`.
#[repr(align(128))]
struct WorkerOutbox {
    /// Generation of the most recently completed task. Worker
    /// Release-stores `assigned_gen` here after the task returns; main
    /// Acquire-spins on this to know the task is done.
    done_gen: AtomicU64,
}

// SAFETY: the `UnsafeCell<Option<TaskPtr>>` is published via the
// Release/Acquire pair on `assigned_gen`; main never overwrites it
// while a worker is still running the previous task (because main
// waits on `done_gen >= prev_gen` before publishing the next one).
// The raw pointer inside is `Send + Sync` because the closure trait
// object itself is `Sync` (and we never call it from a thread that
// doesn't hold the dispatch_lock + observed assigned_gen).
unsafe impl Sync for WorkerInbox {}
unsafe impl Send for WorkerInbox {}

struct PoolShared {
    /// Per-extra inboxes. `len() == n_total - 1`. Each inbox is on
    /// its own cache line.
    inboxes: Vec<WorkerInbox>,
    /// Per-extra outboxes. `len() == n_total - 1`. Each outbox is on
    /// its own cache line.
    outboxes: Vec<WorkerOutbox>,
    /// Shutdown flag set on drop.
    shutdown: AtomicBool,
}

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
    /// Monotonic dispatch-generation counter. Bumped on every
    /// `broadcast_scoped` call (under the dispatch lock).
    next_gen: AtomicU64,
    /// Serializes concurrent callers of [`Self::broadcast_scoped`].
    /// `broadcast_scoped` writes per-worker `task` and `assigned_gen`;
    /// two callers from different threads would race those writes
    /// against each other and against worker reads. The mutex makes
    /// the API safe to call from anywhere.
    dispatch_lock: Mutex<()>,
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
        let inboxes: Vec<_> = (0..n_extra)
            .map(|_| WorkerInbox {
                assigned_gen: AtomicU64::new(0),
                task: UnsafeCell::new(None),
            })
            .collect();
        let outboxes: Vec<_> = (0..n_extra)
            .map(|_| WorkerOutbox {
                done_gen: AtomicU64::new(0),
            })
            .collect();
        let shared = Arc::new(PoolShared {
            inboxes,
            outboxes,
            shutdown: AtomicBool::new(false),
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
            next_gen: AtomicU64::new(0),
            dispatch_lock: Mutex::new(()),
        }
    }

    /// Broadcast `f` to `n_active` workers: main (worker 0) plus
    /// `n_active - 1` extras. Inactive extras keep spinning on their
    /// own slot and pay zero broadcast overhead. Blocks until every
    /// active extra has finished `f`.
    ///
    /// `n_active` must be in `1..=self.n_workers()`.
    pub fn broadcast_scoped(&self, n_active: usize, f: &(dyn Fn(usize) + Sync)) {
        assert!(
            (1..=self.n_total).contains(&n_active),
            "n_active={n_active} out of range 1..={}",
            self.n_total
        );

        if n_active == 1 {
            f(0);
            return;
        }

        // Serialize concurrent callers; per-worker `task` and
        // `assigned_gen` writes from different threads would race.
        let _guard = self.dispatch_lock.lock().expect("dispatch_lock poisoned");

        let n_extra_active = n_active - 1;
        let new_gen = self.next_gen.fetch_add(1, Ordering::Relaxed) + 1;

        // Launder the scoped lifetime. SAFETY: we block until every
        // active extra finishes `f`, so the closure outlives all
        // worker accesses. Inactive extras never see the closure
        // pointer (they don't read `task`).
        let f_static: &(dyn Fn(usize) + Sync + 'static) = unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), &(dyn Fn(usize) + Sync + 'static)>(f)
        };
        let task_ptr = f_static as TaskPtr;

        // Publish task to each active extra, then bump its
        // assigned_gen. Release-store on assigned_gen synchronizes
        // the prior task write.
        for inbox in self.shared.inboxes[..n_extra_active].iter() {
            unsafe {
                *inbox.task.get() = Some(task_ptr);
            }
            inbox.assigned_gen.store(new_gen, Ordering::Release);
        }

        // Main runs as worker 0.
        f(0);

        // Wait for active extras to ack via `done_gen >= new_gen`.
        // Each outbox is on its own cache line, so this scan reads
        // cold lines but never contends with main's inbox writes.
        for outbox in self.shared.outboxes[..n_extra_active].iter() {
            spin_until_ge_u64(&outbox.done_gen, new_gen);
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Wake spinners. They poll `shutdown` between iterations so
        // they'll observe it and return on their own.
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn pinned_worker_loop(shared: Arc<PoolShared>, worker_idx: usize) {
    platform::pin_current_worker(worker_idx);

    let inbox = &shared.inboxes[worker_idx - 1];
    let outbox = &shared.outboxes[worker_idx - 1];
    let mut last_done: u64 = 0;
    loop {
        // Spin-wait for a new generation or shutdown. With
        // `pin_current_worker` applied the OS keeps us on a fast core
        // so a pure spin is acceptable.
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let g = inbox.assigned_gen.load(Ordering::Acquire);
            if g > last_done {
                last_done = g;
                break;
            }
            std::hint::spin_loop();
        }

        // Synchronized with main's Release on `assigned_gen`: the
        // task pointer write is visible. SAFETY: main waits for
        // `done_gen >= last_done` before overwriting `task`, so the
        // pointer is valid for the entire duration of `task(worker_idx)`.
        let task_opt: Option<TaskPtr> = unsafe { *inbox.task.get() };
        if let Some(task_ptr) = task_opt {
            let task: &(dyn Fn(usize) + Sync + 'static) = unsafe { &*task_ptr };
            task(worker_idx);
        }

        // Release-store `done_gen = last_done` so main's Acquire-load
        // sees the task as fully complete (incl. all task-side stores).
        outbox.done_gen.store(last_done, Ordering::Release);
    }
}

/// Bounded spin then `yield_now` fallback. Used by `broadcast_scoped`
/// to wait on per-extra `done_gen` counters, where the calling thread
/// is whatever thread invoked the broadcast (not necessarily a pinned
/// worker), so pure spinning could starve other tasks if the OS
/// preempts us mid-spin. The 2048-iteration budget corresponds to
/// roughly 1 µs of spinning on a 3-4 GHz core before the first yield.
#[inline]
fn spin_until_ge_u64(counter: &AtomicU64, target: u64) {
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
