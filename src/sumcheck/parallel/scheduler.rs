//! Field-agnostic scheduler driving a [`SumcheckRound`] through a
//! sequence of `broadcast_scoped` calls (D2: per-phase `n_active`
//! shrinkage).
//!
//! ### Per-round protocol inside one phase (matches Approach 4 in
//! [`super::legacy`])
//!
//! ```text
//!   workers do reduce_chunk → store partial → reduce_counter.fetch_add(Release)
//!   main:    spin until reduce_counter == (round + 1) * n_active
//!            linear-reduce partials via R::combine → R::observe_partial
//!            bind_go.store(round + 1, Release)
//!   workers: spin until bind_go ≥ round + 1, then bind_chunk
//! ```
//!
//! There is **no** bind barrier between rounds *within a phase*: at
//! round `r` worker `i` writes pair indices `[i·C, (i+1)·C)` where
//! `C = live/n_active`, and at round `r + 1` it reads exactly those
//! same indices in the freshly-written buffer (live halves, the chunk
//! halves with it). So the next-round read is from the worker's *own*
//! writes; no cross-worker visibility required.
//!
//! ### Multi-phase shrinkage (D2)
//!
//! [`par_sumcheck`] dispatches a sequence of `broadcast_scoped`
//! phases, halving `n_active` at each phase boundary. Within one
//! phase we keep the same `n_active` and run as many rounds as the
//! per-field [`SumcheckRound::MIN_PAIRS_PER_WORKER`] allows. When the
//! per-worker pair count drops below `MIN_PAIRS_PER_WORKER` we halve
//! the worker count (one extra `broadcast_scoped` dispatch per
//! halving) until either the worker count falls below 2 or all rounds
//! have been processed.
//!
//! The trade-off, per phase boundary:
//!
//! - **Save**: per-round barrier cost shrinks roughly linearly with
//!   `n_active` (305 ns at k=2, 443 ns at k=4, 681 ns at k=8 on M4).
//!   Halving doubles the per-worker chunk size which gives the
//!   sequential inner loop more amortisation.
//! - **Pay**: one extra `broadcast_scoped` dispatch (~300-680 ns)
//!   plus a fresh round-counter reset.
//!
//! Net win when the saved per-round barrier cost across the new
//! phase's rounds exceeds the dispatch cost. Bigger pools and
//! heavier per-pair work both make the win larger.
//!
//! ### Sequential tail
//!
//! The scheduler returns the number of rounds completed inside the
//! parallel scope; the caller is responsible for the sequential tail
//! (`challenges.len() - rounds_done`).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::field::SumcheckRound;
use super::pool::PinnedPool;

/// `Vec<UnsafeCell<P>>` wrapped so workers can share an `&` reference.
/// Race-free as long as each `worker_idx` is the unique writer of slot
/// `worker_idx`, which [`run_phase`] enforces.
///
/// The wrapper also exists so Rust 2021's disjoint-capture analysis
/// backs off to capturing the whole `&PartialsSlots` (which our
/// `Sync` impl applies to) rather than `&Vec<UnsafeCell<P>>` (which
/// is `!Sync` because `UnsafeCell` is `!Sync`).
#[repr(transparent)]
struct PartialsSlots<P>(Vec<UnsafeCell<P>>);

// SAFETY: see PartialsSlots doc.
unsafe impl<P: Send> Sync for PartialsSlots<P> {}

/// Pick the number of workers to dispatch for a sumcheck of size
/// `initial_pairs`, given a pool with `pool_total` workers and a
/// per-field `target_pairs_per_worker`.
///
/// Returns at least 1 (no parallelism) and at most `pool_total`.
#[inline]
pub(super) fn pick_n_workers(
    initial_pairs: usize,
    pool_total: usize,
    target_pairs_per_worker: usize,
) -> usize {
    if initial_pairs == 0 || pool_total <= 1 || target_pairs_per_worker == 0 {
        return 1;
    }
    (initial_pairs / target_pairs_per_worker)
        .max(2)
        .min(pool_total)
}

/// Drive `round` through a multi-phase parallel prefix and return the
/// number of rounds completed inside the parallel scope. The caller
/// is responsible for the sequential tail
/// (`challenges.len() - rounds_done`) and for extracting the live
/// `(f, g)` buffers using the returned `rounds_done` to decide
/// ping-pong parity.
///
/// `n_workers_initial` is the worker count for the first phase; D2
/// halves it at each phase boundary. `initial_pairs` is the live pair
/// count at round 0 (i.e. `f.len() / 2`).
pub fn par_sumcheck<R>(
    round: &R,
    challenges: &[R::Elem],
    pool: &PinnedPool,
    n_workers_initial: usize,
    initial_pairs: usize,
) -> usize
where
    R: SumcheckRound,
{
    let n_rounds = challenges.len();
    if n_rounds == 0 || n_workers_initial <= 1 || initial_pairs == 0 {
        return 0;
    }

    let mut current_pairs = initial_pairs;
    let mut rounds_done = 0usize;
    let mut current_workers = n_workers_initial.min(pool.n_workers());

    // Pre-allocate the per-phase scratch slots once, sized to the
    // largest `n_workers` we'll ever use this call (= the initial
    // worker count). Subsequent phases address only the leading
    // `current_workers` slots, so the tail of `partials` is unused.
    // Saves an allocation per phase boundary.
    let partials: PartialsSlots<R::Partial> = PartialsSlots(
        (0..current_workers)
            .map(|_| UnsafeCell::new(R::Partial::default()))
            .collect(),
    );

    while current_workers >= 2 && rounds_done < n_rounds {
        let per_worker = current_pairs / current_workers;
        if per_worker < R::MIN_PAIRS_PER_WORKER {
            // Too little work to justify this many workers. Halve
            // and re-evaluate: with a smaller `current_workers` the
            // per-worker chunk doubles, which may cross back above
            // MIN_PAIRS_PER_WORKER and let us run another phase.
            current_workers /= 2;
            continue;
        }

        let ratio = per_worker / R::MIN_PAIRS_PER_WORKER;
        let phase_rounds = ((ratio.ilog2() as usize) + 1).min(n_rounds - rounds_done);

        run_phase(
            round,
            &challenges[rounds_done..rounds_done + phase_rounds],
            pool,
            current_workers,
            current_pairs,
            rounds_done,
            &partials,
        );

        rounds_done += phase_rounds;
        current_pairs >>= phase_rounds;
        // D2: halve at every phase boundary. The next iteration
        // either (a) finds enough work to run more rounds at the
        // smaller worker count, or (b) finds per_worker still below
        // MIN and keeps halving until we exit.
        current_workers /= 2;
    }

    rounds_done
}

/// Run one D2 phase: `phase_rounds` rounds at `n_workers`, starting
/// at absolute round `round_offset`, with read-buffer parity at
/// `round_offset & 1`.
///
/// Caller (the [`par_sumcheck`] loop) must guarantee
/// `(live_pairs_start >> phase_rounds) >= 0`, i.e. the per-worker
/// chunk size at every round inside the phase stays consistent with
/// the worker count.
#[allow(clippy::too_many_arguments)]
fn run_phase<R>(
    round: &R,
    challenges: &[R::Elem],
    pool: &PinnedPool,
    n_workers: usize,
    live_pairs_start: usize,
    round_offset: usize,
    partials: &PartialsSlots<R::Partial>,
) where
    R: SumcheckRound,
{
    let phase_rounds = challenges.len();
    debug_assert!(phase_rounds > 0);
    debug_assert!(n_workers >= 2);
    debug_assert!(n_workers <= partials.0.len());

    // Phase-local barriers. Reset on every phase entry; the
    // `broadcast_scoped` itself joins all workers before returning,
    // so the previous phase's atomics cannot leak into this one.
    let reduce_counter = AtomicUsize::new(0);
    let bind_go = AtomicUsize::new(0);

    let partials_ref = partials;

    let worker_body = |worker_idx: usize| {
        for r in 0..phase_rounds {
            let abs_round = round_offset + r;
            let live_pairs = live_pairs_start >> r;
            let (lo, hi) = chunk_range(live_pairs, n_workers, worker_idx);
            let len = hi - lo;

            let partial = round.reduce_chunk(abs_round, lo, len);
            // SAFETY: this worker is the unique writer of slot
            // `worker_idx`; main only reads after reduce_counter has
            // advanced, which the Release/Acquire pair makes visible.
            unsafe {
                *partials_ref.0[worker_idx].get() = partial;
            }
            reduce_counter.fetch_add(1, Ordering::Release);

            if worker_idx == 0 {
                spin_until_ge(&reduce_counter, (r + 1) * n_workers);
                let mut sum = R::Partial::default();
                for slot in partials_ref.0[..n_workers].iter() {
                    // SAFETY: reduce_counter == (r + 1) * n_workers
                    // proves every worker has finished writing its
                    // slot for this round.
                    let part = unsafe { *slot.get() };
                    sum = R::combine(sum, part);
                }
                R::observe_partial(abs_round, sum, &challenges[r]);
                bind_go.store(r + 1, Ordering::Release);
            } else {
                spin_until_ge(&bind_go, r + 1);
            }

            if len > 0 {
                // SAFETY: chunk_range partitions [0, live_pairs) into
                // disjoint windows across workers, and the ping-pong
                // layout in the SumcheckRound impl ensures the read
                // buffer (parity `abs_round & 1`) is not the write
                // buffer (parity `(abs_round + 1) & 1`).
                unsafe {
                    round.bind_chunk(abs_round, lo, len, &challenges[r]);
                }
            }
        }
    };

    pool.broadcast_scoped(n_workers, &worker_body);
}

#[inline]
fn chunk_range(live_pairs: usize, n_workers: usize, worker_idx: usize) -> (usize, usize) {
    let chunk = live_pairs.div_ceil(n_workers);
    let lo = (worker_idx * chunk).min(live_pairs);
    let hi = ((worker_idx + 1) * chunk).min(live_pairs);
    (lo, hi)
}

/// Bounded spin then `yield_now` fallback for per-round atomics. The
/// pool's worker doorbell uses a pure spin (workers are pinned), but
/// the per-round counters are touched by both pinned workers AND the
/// main thread (which is whatever thread the caller is on), so we
/// keep the yield fallback to avoid starving an unpinned main thread
/// if it ends up spinning under load.
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
