//! [`SumcheckRound`] implementation and public wrapper for the Fp128
//! delayed kernel.
//!
//! Same shape as the GF128 impl: ping-pong `buf0`/`buf1` for `f` and
//! `g`, round parity picks read vs. write, per-pair math reuses the
//! shared helpers in [`super::super::legacy`].

use hachi_pcs::AdditiveGroup as _;
use pinned_pool::PinnedPool;
use sumcheck_parallel::{par_sumcheck, pick_n_workers, Schedule, SumcheckRound};

use super::super::super::fp128::{sumcheck_deg2_delayed_fp128, sumcheck_deg2_delayed_fp128_fused};
use super::super::super::Fp128;
use super::super::legacy::{bind_chunk_fp128, bind_then_reduce_chunk_fp128, partial_triple_fp128};

/// Owns the ping-pong buffers for one sumcheck call. See
/// [`super::gf128::GF128DelayedRound`] for the rationale behind
/// caching the raw pointers alongside the backing Vecs.
pub(crate) struct Fp128DelayedRound {
    buf0_f_ptr: *mut Fp128,
    buf0_g_ptr: *mut Fp128,
    buf1_f_ptr: *mut Fp128,
    buf1_g_ptr: *mut Fp128,
    buf0_f: Vec<Fp128>,
    buf0_g: Vec<Fp128>,
    buf1_f: Vec<Fp128>,
    buf1_g: Vec<Fp128>,
    initial_pairs: usize,
}

// SAFETY: see GF128DelayedRound. Same partition/ping-pong invariants.
unsafe impl Sync for Fp128DelayedRound {}

impl Fp128DelayedRound {
    pub(crate) fn new(
        mut buf0_f: Vec<Fp128>,
        mut buf0_g: Vec<Fp128>,
        initial_pairs: usize,
    ) -> Self {
        let mut buf1_f = vec![Fp128::ZERO; initial_pairs];
        let mut buf1_g = vec![Fp128::ZERO; initial_pairs];
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

    pub(crate) fn into_live_buffers(self, scope_rounds: usize) -> (Vec<Fp128>, Vec<Fp128>) {
        if scope_rounds & 1 == 0 {
            (self.buf0_f, self.buf0_g)
        } else {
            (self.buf1_f, self.buf1_g)
        }
    }

    #[inline(always)]
    fn read_ptrs(&self, round: usize) -> (*const Fp128, *const Fp128) {
        if round & 1 == 0 {
            (self.buf0_f_ptr as *const _, self.buf0_g_ptr as *const _)
        } else {
            (self.buf1_f_ptr as *const _, self.buf1_g_ptr as *const _)
        }
    }

    #[inline(always)]
    fn write_ptrs(&self, round: usize) -> (*mut Fp128, *mut Fp128) {
        if round & 1 == 0 {
            (self.buf1_f_ptr, self.buf1_g_ptr)
        } else {
            (self.buf0_f_ptr, self.buf0_g_ptr)
        }
    }
}

impl SumcheckRound for Fp128DelayedRound {
    type Elem = Fp128;
    type Partial = (Fp128, Fp128, Fp128);

    const MIN_PAIRS_PER_WORKER: usize = 8;
    const TARGET_PAIRS_PER_WORKER: usize = 256;

    #[inline]
    fn reduce_chunk(&self, round: usize, lo: usize, len: usize) -> Self::Partial {
        if len == 0 {
            return (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO);
        }
        let (rf, rg) = self.read_ptrs(round);
        partial_triple_fp128(rf, rg, lo, len)
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
        bind_chunk_fp128(rf, rg, wf, wg, lo, len, *r);
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
            return (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO);
        }
        debug_assert!(round >= 1);
        let (rf_prev, rg_prev) = self.read_ptrs(round - 1);
        let (wf, wg) = self.write_ptrs(round - 1);
        // SAFETY: caller's disjoint-windows contract for round-`round`
        // pair units, and the ping-pong layout makes
        // `read_ptrs(round - 1)` (parity `(round - 1) & 1`) and
        // `write_ptrs(round - 1)` (parity `round & 1`) distinct
        // buffers for this worker's window.
        unsafe { bind_then_reduce_chunk_fp128(rf_prev, rg_prev, wf, wg, lo, len, *r_prev) }
    }
}

/// Pinned-pool implementation for the Fp128 deg-2 delayed sumcheck.
///
/// See [`super::gf128::sumcheck_deg2_delayed_gf128_pinned`] for
/// the orchestration shape.
///
/// `use_fused_path`: if `true`, rounds `1..` of each phase go through
/// [`Fp128DelayedRound::bind_then_reduce_chunk`] (single pass over
/// the previous-round buffer for bind + reduce). If `false`, the
/// classic two-pass `bind_chunk` + `reduce_chunk` protocol is used.
/// The sequential tail (small-round fallback) follows the same
/// choice via `sumcheck_deg2_delayed_fp128_fused` vs
/// `sumcheck_deg2_delayed_fp128`.
///
/// `schedule` selects the per-round work-distribution policy. See
/// [`Schedule`] for the full trade-off; `Schedule::default()` =
/// `Schedule::Static` is the best choice for dedicated-core
/// deployments, and `Schedule::guided()` is the best choice for
/// noisy / preemption-prone hosts.
pub fn sumcheck_deg2_delayed_fp128_pinned(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
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
        Fp128DelayedRound::TARGET_PAIRS_PER_WORKER,
    );

    if n_workers_initial <= 1 {
        if use_fused_path {
            sumcheck_deg2_delayed_fp128_fused(f, g, challenges);
        } else {
            sumcheck_deg2_delayed_fp128(f, g, challenges);
        }
        return;
    }

    let round = Fp128DelayedRound::new(
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
        let (live_f, live_g) = round.into_live_buffers(0);
        *f = live_f;
        *g = live_g;
        if use_fused_path {
            sumcheck_deg2_delayed_fp128_fused(f, g, challenges);
        } else {
            sumcheck_deg2_delayed_fp128(f, g, challenges);
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
            sumcheck_deg2_delayed_fp128_fused(f, g, &challenges[rounds_done..]);
        } else {
            sumcheck_deg2_delayed_fp128(f, g, &challenges[rounds_done..]);
        }
    }
}
