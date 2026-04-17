//! [`SumcheckRound`] implementation and public wrapper for the GF128
//! delayed kernel.
//!
//! The struct [`GF128DelayedRound`] owns the ping-pong buffers
//! (`buf0`, `buf1` for both `f` and `g`) and uses round parity to
//! pick read vs. write each round. Per-pair math reuses
//! [`super::super::legacy::partial_triple_gf128`] and
//! [`super::super::legacy::bind_chunk_gf128`] so the inner loops are
//! identical to the existing `_pinned` baseline.
//!
//! [`sumcheck_deg2_delayed_gf128_pinned`] is the public entry
//! point used by tests and benches.

use binius_field::Field as _;
use pinned_pool::PinnedPool;
use sumcheck_parallel::{par_sumcheck, pick_n_workers, Schedule, SumcheckRound};

use super::super::super::gf128::{sumcheck_deg2_delayed_gf128, sumcheck_deg2_delayed_gf128_fused};
use super::super::super::GF128;
use super::super::legacy::{bind_chunk_gf128, bind_then_reduce_chunk_gf128, partial_triple_gf128};

/// Owns the ping-pong buffers for one sumcheck call.
///
/// The four `buf*` `Vec<GF128>` fields are the allocations kept alive
/// for the duration of the parallel scope; the four `buf*_ptr` fields
/// are raw pointers into those same allocations, cached at
/// construction time. Workers read the pointers (cheap
/// constant-address loads that LLVM hoists out of the per-round loop)
/// instead of re-deriving them from `Vec::as_ptr` through an
/// `UnsafeCell` on every round. Saves ~200-300 ns per round vs the
/// UnsafeCell-around-Vec version.
///
/// The Vec allocations are never resized between `new` and
/// `into_live_buffers`, so the cached pointers stay valid for the
/// full parallel scope.
pub(crate) struct GF128DelayedRound {
    buf0_f_ptr: *mut GF128,
    buf0_g_ptr: *mut GF128,
    buf1_f_ptr: *mut GF128,
    buf1_g_ptr: *mut GF128,
    buf0_f: Vec<GF128>,
    buf0_g: Vec<GF128>,
    buf1_f: Vec<GF128>,
    buf1_g: Vec<GF128>,
    initial_pairs: usize,
}

// SAFETY: workers hold `&self` and call `reduce_chunk` (read-only on
// the round-r read buffer) and `bind_chunk` (write-disjoint chunks
// into the round-r write buffer). The scheduler partitions
// `[0, current_pairs)` into disjoint windows per worker, and the
// ping-pong layout ensures the read buffer at round r is never the
// same as the write buffer, so no cross-worker aliasing is possible.
// Per-round reduce/bind barriers ensure writes from round r are
// visible to reads in round r+1.
unsafe impl Sync for GF128DelayedRound {}

impl GF128DelayedRound {
    pub(crate) fn new(mut buf0_f: Vec<GF128>, mut buf0_g: Vec<GF128>, initial_pairs: usize) -> Self {
        let mut buf1_f = vec![GF128::ZERO; initial_pairs];
        let mut buf1_g = vec![GF128::ZERO; initial_pairs];
        let buf0_f_ptr = buf0_f.as_mut_ptr();
        let buf0_g_ptr = buf0_g.as_mut_ptr();
        let buf1_f_ptr = buf1_f.as_mut_ptr();
        let buf1_g_ptr = buf1_g.as_mut_ptr();
        Self {
            buf0_f_ptr,
            buf0_g_ptr,
            buf1_f_ptr,
            buf1_g_ptr,
            buf0_f,
            buf0_g,
            buf1_f,
            buf1_g,
            initial_pairs,
        }
    }

    pub(crate) fn initial_pairs(&self) -> usize {
        self.initial_pairs
    }

    /// Pull the live `(f, g)` buffers back out after `scope_rounds`
    /// completed parallel rounds. Caller is responsible for
    /// `truncate`-ing them to the post-tail length.
    pub(crate) fn into_live_buffers(self, scope_rounds: usize) -> (Vec<GF128>, Vec<GF128>) {
        if scope_rounds & 1 == 0 {
            (self.buf0_f, self.buf0_g)
        } else {
            (self.buf1_f, self.buf1_g)
        }
    }

    #[inline(always)]
    fn read_ptrs(&self, round: usize) -> (*const GF128, *const GF128) {
        if round & 1 == 0 {
            (self.buf0_f_ptr as *const _, self.buf0_g_ptr as *const _)
        } else {
            (self.buf1_f_ptr as *const _, self.buf1_g_ptr as *const _)
        }
    }

    #[inline(always)]
    fn write_ptrs(&self, round: usize) -> (*mut GF128, *mut GF128) {
        if round & 1 == 0 {
            (self.buf1_f_ptr, self.buf1_g_ptr)
        } else {
            (self.buf0_f_ptr, self.buf0_g_ptr)
        }
    }
}

impl SumcheckRound for GF128DelayedRound {
    type Elem = GF128;
    type Partial = (GF128, GF128, GF128);

    const MIN_PAIRS_PER_WORKER: usize = 8;
    const TARGET_PAIRS_PER_WORKER: usize = 256;

    #[inline]
    fn reduce_chunk(&self, round: usize, lo: usize, len: usize) -> Self::Partial {
        if len == 0 {
            return (GF128::ZERO, GF128::ZERO, GF128::ZERO);
        }
        let (rf, rg) = self.read_ptrs(round);
        partial_triple_gf128(rf, rg, lo, len)
    }

    #[inline]
    fn combine(a: Self::Partial, b: Self::Partial) -> Self::Partial {
        (a.0 + b.0, a.1 + b.1, a.2 + b.2)
    }

    #[inline]
    unsafe fn bind_chunk(&self, round: usize, lo: usize, len: usize, r: &Self::Elem) {
        if len == 0 {
            return;
        }
        let (rf, rg) = self.read_ptrs(round);
        let (wf, wg) = self.write_ptrs(round);
        bind_chunk_gf128(rf, rg, wf, wg, lo, len, *r);
    }

    /// Fused bind-(round - 1) + reduce-round in a single pass over the
    /// round-(round - 1) buffer. `lo` and `len` are in round-`round`
    /// pair units.
    #[inline]
    unsafe fn bind_then_reduce_chunk(
        &self,
        round: usize,
        lo: usize,
        len: usize,
        r_prev: &Self::Elem,
    ) -> Self::Partial {
        if len == 0 {
            return (GF128::ZERO, GF128::ZERO, GF128::ZERO);
        }
        debug_assert!(round >= 1);
        let (rf_prev, rg_prev) = self.read_ptrs(round - 1);
        let (wf, wg) = self.write_ptrs(round - 1);
        // SAFETY: caller's disjoint-windows contract for round-`round`
        // pair units, and the ping-pong layout makes
        // `read_ptrs(round - 1)` and `write_ptrs(round - 1)`
        // distinct buffers for this worker's window.
        unsafe { bind_then_reduce_chunk_gf128(rf_prev, rg_prev, wf, wg, lo, len, *r_prev) }
    }
}

/// Pinned-pool implementation for the GF128 deg-2 delayed sumcheck.
///
/// Adaptive: picks `n_workers_initial` from
/// [`GF128DelayedRound::TARGET_PAIRS_PER_WORKER`] and the global
/// pool size, then hands off to [`par_sumcheck`] which dispatches a
/// sequence of D2 phases (halving `n_active` at each phase boundary
/// while [`GF128DelayedRound::MIN_PAIRS_PER_WORKER`] still has room).
/// The remainder runs through the sequential
/// [`sumcheck_deg2_delayed_gf128`]. If parallelism is not profitable
/// (small problem), runs the sequential kernel directly.
///
/// `use_fused_path`: if `true`, rounds `1..` of each phase go through
/// [`GF128DelayedRound::bind_then_reduce_chunk`] (single pass over
/// the previous-round buffer for bind + reduce). If `false`, the
/// classic two-pass `bind_chunk` + `reduce_chunk` protocol is used.
/// The sequential tail (small-round fallback) follows the same
/// choice via `sumcheck_deg2_delayed_gf128_fused` vs
/// `sumcheck_deg2_delayed_gf128`.
///
/// `schedule` selects the per-round work-distribution policy. See
/// [`Schedule`] for the full trade-off; `Schedule::default()` =
/// `Schedule::Static` is the best choice for dedicated-core
/// deployments, and `Schedule::guided()` is the best choice for
/// noisy / preemption-prone hosts.
pub fn sumcheck_deg2_delayed_gf128_pinned(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
    use_fused_path: bool,
    schedule: Schedule,
) {
    let n_rounds = challenges.len();
    if n_rounds == 0 {
        return;
    }
    let initial_len = f.len();
    if initial_len == 0 {
        return;
    }
    let initial_pairs = initial_len / 2;

    let pool = PinnedPool::global();
    let pool_total = pool.n_workers();
    let n_workers_initial = pick_n_workers(
        initial_pairs,
        pool_total,
        GF128DelayedRound::TARGET_PAIRS_PER_WORKER,
    );

    if n_workers_initial <= 1 {
        if use_fused_path {
            sumcheck_deg2_delayed_gf128_fused(f, g, challenges);
        } else {
            sumcheck_deg2_delayed_gf128(f, g, challenges);
        }
        return;
    }

    let round = GF128DelayedRound::new(
        std::mem::take(f),
        std::mem::take(g),
        initial_pairs,
    );

    let rounds_done = par_sumcheck(
        &round,
        challenges,
        pool,
        n_workers_initial,
        round.initial_pairs(),
        use_fused_path,
        schedule,
    );

    if rounds_done == 0 {
        // par_sumcheck declined to parallelise (tiny problem). Put
        // the buffers back and run sequential over all rounds.
        let (live_f, live_g) = round.into_live_buffers(0);
        *f = live_f;
        *g = live_g;
        if use_fused_path {
            sumcheck_deg2_delayed_gf128_fused(f, g, challenges);
        } else {
            sumcheck_deg2_delayed_gf128(f, g, challenges);
        }
        return;
    }

    let (mut live_f, mut live_g) = round.into_live_buffers(rounds_done);
    let live_len = initial_len >> rounds_done;
    live_f.truncate(live_len);
    live_g.truncate(live_len);
    *f = live_f;
    *g = live_g;

    if rounds_done < n_rounds {
        if use_fused_path {
            sumcheck_deg2_delayed_gf128_fused(f, g, &challenges[rounds_done..]);
        } else {
            sumcheck_deg2_delayed_gf128(f, g, &challenges[rounds_done..]);
        }
    }
}
