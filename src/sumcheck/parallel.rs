//! Parallel wrappers for the delayed sumcheck kernels, covering three approaches.
//!
//! **Approach 1 — manual chunked `rayon::scope`** (`*_par1_scope`): one
//! `rayon::scope` per round with exactly `n_workers` spawns. Reduce and bind
//! are fused into the same worker closure, so each round pays exactly one
//! scope-level dispatch. A `*_par1_pariter` control uses
//! `(0..n_workers).into_par_iter()` + `par_chunks_mut` to isolate the manual
//! vs. par_iter dispatch difference.
//!
//! **Approach 2 — chili recursive fork/join** (`*_par2_chili`): uses
//! `chili::Scope::global()` with recursive `scope.join(left, right)` down to a
//! `base` pair-count threshold. Chili's workers don't park, so dispatch cost
//! is bounded by a spinning handoff rather than the rayon wake-up floor.
//!
//! **Approach 3 — persistent pool + atomic barrier** (`*_par3_persistent`):
//! spends one `rayon::scope` for the entire call. Workers run all rounds
//! in-thread on a fixed contiguous chunk of the data; cross-round sync is via
//! two `AtomicUsize` counters (one published by workers after reduce, one
//! published by main after black_box). Buffers ping-pong between two
//! preallocated arenas. Once per-worker chunks shrink below a threshold, we
//! exit the scope and finish remaining rounds on the main thread via the
//! sequential kernel.
//!
//! **Approach 4 — globally-persistent pinned pool + doorbell** (`*_par4_pinned`):
//! a process-global `std::thread` pool spawned lazily on first use, with
//! each worker marked `QOS_CLASS_USER_INTERACTIVE` on macOS to keep them on
//! P-cores. Task dispatch is via a Release/Acquire epoch counter
//! (`broadcast_scoped`); workers never park. This removes the per-call
//! `rayon::scope` setup cost that dominates at small `n` in Approach 3.
//! The pool defaults to at most 8 workers (vs rayon's default of
//! `available_parallelism()`) to reduce barrier-counter contention on
//! heterogeneous-core silicon. Inside one call the protocol is identical
//! to Approach 3 (reduce_counter / bind_go atomics, ping-pong buffers).
//!
//! **Bind safety.** All approaches write the bound output out-of-place into a
//! scratch buffer and swap it in after sync, to avoid the in-place read-after-
//! write hazard across workers (worker A's read `f[2*j]` would race worker B's
//! write `f[j]` for `j < j'` in a different chunk).
//!
//! **Accumulator reuse.** The accumulator types (`GF128Accum`, `Fp128Accum`)
//! are reused verbatim from their sibling modules so the hot per-pair loop
//! body matches the sequential kernel. Summing partial `reduce()` values is
//! valid because:
//!   - `GF128Accum` uses F_2-linear XOR accumulation and an F_2-linear
//!     reduction, so `reduce(Σ a_i) == Σ reduce(a_i)` in GF(2^128).
//!   - `Fp128Accum` sums `[u128; 4]` limbs without carry; per-chunk totals stay
//!     well under `u128::MAX` for all benchmarked sizes, and `solinas_reduce`
//!     is a ring hom ℤ → ℤ/p so summing reduced values mod p matches.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::black_box;
use rayon::prelude::*;

use super::fp128::Fp128Accum;
use super::gf128::GF128Accum;
use super::{Fp128, GF128};

use binius_field::Field as _;
use hachi_pcs::AdditiveGroup as _;

/// Number of (reduce,bind) worker slots to use for a given `half`.
#[inline]
fn n_chunks(half: usize) -> usize {
    rayon::current_num_threads().max(1).min(half.max(1))
}

// -------------------- Shared helpers for Approach 3 --------------------

/// A `*mut T` wrapper that is `Send + Sync`. Used only inside `rayon::scope`
/// closures where the pointed-to region is kept alive by the outer stack frame
/// and accesses are coordinated via atomics.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// A `Vec<UnsafeCell<T>>` wrapped so we can share an `&` reference across
/// worker threads. Writes to distinct indices are race-free; reads are
/// synchronized via the reduce counter.
#[repr(transparent)]
struct SharedSlots<T>(Vec<UnsafeCell<T>>);
unsafe impl<T: Send> Sync for SharedSlots<T> {}

/// How many rounds Phase-3 runs inside the persistent scope before cutting
/// over to the sequential kernel. The per-worker pair count halves each round
/// we keep in the scope.
const PAR3_MIN_PAIRS_PER_WORKER: usize = 8;

/// Target pairs-per-worker in the first round for Approach 4. Used to pick
/// `n_active` adaptively: we want at least this much work per worker to
/// amortise the per-dispatch barrier cost (which scales with `n_active`
/// on the `done` cache line). Tuned empirically on M4 Max: at 170 the
/// `n = 10` case (512 initial pairs) picks 3 active workers, which is the
/// sweet spot between barrier contention and parallelism.
const PAR4_TARGET_PAIRS_PER_WORKER: usize = 256;

/// Returns the number of rounds to keep inside the persistent scope for
/// Approach 3, given an initial pair count and worker count.
#[inline]
fn par3_scope_rounds(initial_pairs: usize, n_workers: usize, n_rounds: usize) -> usize {
    if n_workers <= 1 || n_rounds == 0 {
        return 0;
    }
    let initial_per_worker = initial_pairs / n_workers;
    if initial_per_worker < PAR3_MIN_PAIRS_PER_WORKER {
        return 0;
    }
    // Keep the scope while `initial_per_worker >> r >= MIN`, i.e.
    // r <= floor(log2(initial_per_worker / MIN)). Number of runnable rounds is
    // that plus one (for r = 0).
    let ratio = initial_per_worker / PAR3_MIN_PAIRS_PER_WORKER;
    let max_scope_rounds = ratio.ilog2() as usize + 1;
    max_scope_rounds.min(n_rounds)
}

// ----------------------------- GF128 delayed -----------------------------

/// Manual chunked `rayon::scope` version of `sumcheck_deg2_delayed_gf128`.
///
/// One `rayon::scope` per round. Each spawn covers one contiguous chunk and
/// performs reduce + bind fused.
pub fn sumcheck_deg2_delayed_gf128_par1_scope(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
) {
    let mut scratch_f: Vec<GF128> = Vec::new();
    let mut scratch_g: Vec<GF128> = Vec::new();

    for round in 0..challenges.len() {
        let half = f.len() / 2;
        let r = challenges[round];

        let n = n_chunks(half);
        let chunk_pairs = half.div_ceil(n);

        scratch_f.resize(half, GF128::ZERO);
        scratch_g.resize(half, GF128::ZERO);

        let mut partials: Vec<(GF128, GF128, GF128)> =
            vec![(GF128::ZERO, GF128::ZERO, GF128::ZERO); n];

        {
            let f_src: &[GF128] = f;
            let g_src: &[GF128] = g;
            let mut rem_sf: &mut [GF128] = scratch_f.as_mut_slice();
            let mut rem_sg: &mut [GF128] = scratch_g.as_mut_slice();
            let mut rem_p: &mut [(GF128, GF128, GF128)] = partials.as_mut_slice();

            rayon::scope(|s| {
                for i in 0..n {
                    let lo = i * chunk_pairs;
                    let this = chunk_pairs.min(half.saturating_sub(lo));
                    if this == 0 {
                        break;
                    }

                    let (slot_slice, tail_p) = rem_p.split_at_mut(1);
                    let (dst_f, tail_sf) = rem_sf.split_at_mut(this);
                    let (dst_g, tail_sg) = rem_sg.split_at_mut(this);
                    rem_p = tail_p;
                    rem_sf = tail_sf;
                    rem_sg = tail_sg;

                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];

                    s.spawn(move |_| {
                        let mut h0 = GF128Accum::zero();
                        let mut h1 = GF128Accum::zero();
                        let mut h_inf = GF128Accum::zero();

                        for j in 0..this {
                            let f0 = chunk_f[2 * j];
                            let f1 = chunk_f[2 * j + 1];
                            let g0 = chunk_g[2 * j];
                            let g1 = chunk_g[2 * j + 1];
                            let df = f1 - f0;
                            let dg = g1 - g0;

                            h0.fmadd(f0, g0);
                            h1.fmadd(f1, g1);
                            h_inf.fmadd(df, dg);
                        }

                        slot_slice[0] = (h0.reduce(), h1.reduce(), h_inf.reduce());

                        for j in 0..this {
                            let f0 = chunk_f[2 * j];
                            let f1 = chunk_f[2 * j + 1];
                            let g0 = chunk_g[2 * j];
                            let g1 = chunk_g[2 * j + 1];
                            dst_f[j] = f0 + r * (f1 - f0);
                            dst_g[j] = g0 + r * (g1 - g0);
                        }
                    });
                }
            });
        }

        let triple = partials.iter().copied().fold(
            (GF128::ZERO, GF128::ZERO, GF128::ZERO),
            |(a0, a1, ai), (b0, b1, bi)| (a0 + b0, a1 + b1, ai + bi),
        );
        black_box(triple);

        std::mem::swap(f, &mut scratch_f);
        std::mem::swap(g, &mut scratch_g);
        f.truncate(half);
        g.truncate(half);
    }
}

/// `par_iter`-based control version of `sumcheck_deg2_delayed_gf128`.
pub fn sumcheck_deg2_delayed_gf128_par1_pariter(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
) {
    let mut scratch_f: Vec<GF128> = Vec::new();
    let mut scratch_g: Vec<GF128> = Vec::new();

    for round in 0..challenges.len() {
        let half = f.len() / 2;
        let r = challenges[round];

        let n = n_chunks(half);
        let chunk_pairs = half.div_ceil(n);

        let triple = {
            let f_src: &[GF128] = f;
            let g_src: &[GF128] = g;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let lo = i * chunk_pairs;
                    let this = chunk_pairs.min(half.saturating_sub(lo));
                    let mut h0 = GF128Accum::zero();
                    let mut h1 = GF128Accum::zero();
                    let mut h_inf = GF128Accum::zero();
                    if this == 0 {
                        return (h0.reduce(), h1.reduce(), h_inf.reduce());
                    }
                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];
                    for j in 0..this {
                        let f0 = chunk_f[2 * j];
                        let f1 = chunk_f[2 * j + 1];
                        let g0 = chunk_g[2 * j];
                        let g1 = chunk_g[2 * j + 1];
                        let df = f1 - f0;
                        let dg = g1 - g0;
                        h0.fmadd(f0, g0);
                        h1.fmadd(f1, g1);
                        h_inf.fmadd(df, dg);
                    }
                    (h0.reduce(), h1.reduce(), h_inf.reduce())
                })
                .reduce(
                    || (GF128::ZERO, GF128::ZERO, GF128::ZERO),
                    |(a0, a1, ai), (b0, b1, bi)| (a0 + b0, a1 + b1, ai + bi),
                )
        };
        black_box(triple);

        scratch_f.resize(half, GF128::ZERO);
        scratch_g.resize(half, GF128::ZERO);
        {
            let f_src: &[GF128] = f;
            let g_src: &[GF128] = g;
            scratch_f
                .par_chunks_mut(chunk_pairs)
                .zip(scratch_g.par_chunks_mut(chunk_pairs))
                .enumerate()
                .for_each(|(i, (dst_f, dst_g))| {
                    let lo = i * chunk_pairs;
                    let this = dst_f.len();
                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];
                    for j in 0..this {
                        let f0 = chunk_f[2 * j];
                        let f1 = chunk_f[2 * j + 1];
                        let g0 = chunk_g[2 * j];
                        let g1 = chunk_g[2 * j + 1];
                        dst_f[j] = f0 + r * (f1 - f0);
                        dst_g[j] = g0 + r * (g1 - g0);
                    }
                });
        }

        std::mem::swap(f, &mut scratch_f);
        std::mem::swap(g, &mut scratch_g);
        f.truncate(half);
        g.truncate(half);
    }
}

// ----------------------------- Fp128 delayed -----------------------------

/// Manual chunked `rayon::scope` version of `sumcheck_deg2_delayed_fp128`.
pub fn sumcheck_deg2_delayed_fp128_par1_scope(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let mut scratch_f: Vec<Fp128> = Vec::new();
    let mut scratch_g: Vec<Fp128> = Vec::new();

    for round in 0..challenges.len() {
        let half = f.len() / 2;
        let r = challenges[round];

        let n = n_chunks(half);
        let chunk_pairs = half.div_ceil(n);

        scratch_f.resize(half, Fp128::ZERO);
        scratch_g.resize(half, Fp128::ZERO);

        let mut partials: Vec<(Fp128, Fp128, Fp128)> =
            vec![(Fp128::ZERO, Fp128::ZERO, Fp128::ZERO); n];

        {
            let f_src: &[Fp128] = f;
            let g_src: &[Fp128] = g;
            let mut rem_sf: &mut [Fp128] = scratch_f.as_mut_slice();
            let mut rem_sg: &mut [Fp128] = scratch_g.as_mut_slice();
            let mut rem_p: &mut [(Fp128, Fp128, Fp128)] = partials.as_mut_slice();

            rayon::scope(|s| {
                for i in 0..n {
                    let lo = i * chunk_pairs;
                    let this = chunk_pairs.min(half.saturating_sub(lo));
                    if this == 0 {
                        break;
                    }

                    let (slot_slice, tail_p) = rem_p.split_at_mut(1);
                    let (dst_f, tail_sf) = rem_sf.split_at_mut(this);
                    let (dst_g, tail_sg) = rem_sg.split_at_mut(this);
                    rem_p = tail_p;
                    rem_sf = tail_sf;
                    rem_sg = tail_sg;

                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];

                    s.spawn(move |_| {
                        let mut h0 = Fp128Accum::zero();
                        let mut h1 = Fp128Accum::zero();
                        let mut h_inf = Fp128Accum::zero();

                        for j in 0..this {
                            let f0 = chunk_f[2 * j];
                            let f1 = chunk_f[2 * j + 1];
                            let g0 = chunk_g[2 * j];
                            let g1 = chunk_g[2 * j + 1];
                            let df = f1 - f0;
                            let dg = g1 - g0;

                            h0.fmadd(f0, g0);
                            h1.fmadd(f1, g1);
                            h_inf.fmadd(df, dg);
                        }

                        slot_slice[0] = (h0.reduce(), h1.reduce(), h_inf.reduce());

                        for j in 0..this {
                            let f0 = chunk_f[2 * j];
                            let f1 = chunk_f[2 * j + 1];
                            let g0 = chunk_g[2 * j];
                            let g1 = chunk_g[2 * j + 1];
                            dst_f[j] = (f1 - f0).mul_add(r, f0);
                            dst_g[j] = (g1 - g0).mul_add(r, g0);
                        }
                    });
                }
            });
        }

        let triple = partials.iter().copied().fold(
            (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO),
            |(a0, a1, ai), (b0, b1, bi)| (a0 + b0, a1 + b1, ai + bi),
        );
        black_box(triple);

        std::mem::swap(f, &mut scratch_f);
        std::mem::swap(g, &mut scratch_g);
        f.truncate(half);
        g.truncate(half);
    }
}

/// `par_iter`-based control version of `sumcheck_deg2_delayed_fp128`.
pub fn sumcheck_deg2_delayed_fp128_par1_pariter(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let mut scratch_f: Vec<Fp128> = Vec::new();
    let mut scratch_g: Vec<Fp128> = Vec::new();

    for round in 0..challenges.len() {
        let half = f.len() / 2;
        let r = challenges[round];

        let n = n_chunks(half);
        let chunk_pairs = half.div_ceil(n);

        let triple = {
            let f_src: &[Fp128] = f;
            let g_src: &[Fp128] = g;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let lo = i * chunk_pairs;
                    let this = chunk_pairs.min(half.saturating_sub(lo));
                    let mut h0 = Fp128Accum::zero();
                    let mut h1 = Fp128Accum::zero();
                    let mut h_inf = Fp128Accum::zero();
                    if this == 0 {
                        return (h0.reduce(), h1.reduce(), h_inf.reduce());
                    }
                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];
                    for j in 0..this {
                        let f0 = chunk_f[2 * j];
                        let f1 = chunk_f[2 * j + 1];
                        let g0 = chunk_g[2 * j];
                        let g1 = chunk_g[2 * j + 1];
                        let df = f1 - f0;
                        let dg = g1 - g0;
                        h0.fmadd(f0, g0);
                        h1.fmadd(f1, g1);
                        h_inf.fmadd(df, dg);
                    }
                    (h0.reduce(), h1.reduce(), h_inf.reduce())
                })
                .reduce(
                    || (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO),
                    |(a0, a1, ai), (b0, b1, bi)| (a0 + b0, a1 + b1, ai + bi),
                )
        };
        black_box(triple);

        scratch_f.resize(half, Fp128::ZERO);
        scratch_g.resize(half, Fp128::ZERO);
        {
            let f_src: &[Fp128] = f;
            let g_src: &[Fp128] = g;
            scratch_f
                .par_chunks_mut(chunk_pairs)
                .zip(scratch_g.par_chunks_mut(chunk_pairs))
                .enumerate()
                .for_each(|(i, (dst_f, dst_g))| {
                    let lo = i * chunk_pairs;
                    let this = dst_f.len();
                    let chunk_f = &f_src[lo * 2..(lo + this) * 2];
                    let chunk_g = &g_src[lo * 2..(lo + this) * 2];
                    for j in 0..this {
                        let f0 = chunk_f[2 * j];
                        let f1 = chunk_f[2 * j + 1];
                        let g0 = chunk_g[2 * j];
                        let g1 = chunk_g[2 * j + 1];
                        dst_f[j] = (f1 - f0).mul_add(r, f0);
                        dst_g[j] = (g1 - g0).mul_add(r, g0);
                    }
                });
        }

        std::mem::swap(f, &mut scratch_f);
        std::mem::swap(g, &mut scratch_g);
        f.truncate(half);
        g.truncate(half);
    }
}

// ===========================================================================
// Approach 2: chili recursive fork/join
// ===========================================================================

#[cfg(feature = "parallel_chili")]
mod chili_impl {
    use super::*;

    // --------- GF128 recursion helpers ---------

    fn reduce_recurse_gf128(
        slice_f: &[GF128],
        slice_g: &[GF128],
        base: usize,
        scope: &mut chili::Scope<'_>,
    ) -> (GF128, GF128, GF128) {
        let n_pairs = slice_f.len() / 2;
        if n_pairs <= base {
            let mut h0 = GF128Accum::zero();
            let mut h1 = GF128Accum::zero();
            let mut h_inf = GF128Accum::zero();
            for j in 0..n_pairs {
                let f0 = slice_f[2 * j];
                let f1 = slice_f[2 * j + 1];
                let g0 = slice_g[2 * j];
                let g1 = slice_g[2 * j + 1];
                let df = f1 - f0;
                let dg = g1 - g0;
                h0.fmadd(f0, g0);
                h1.fmadd(f1, g1);
                h_inf.fmadd(df, dg);
            }
            return (h0.reduce(), h1.reduce(), h_inf.reduce());
        }
        let mid_elem = (n_pairs / 2) * 2;
        let (f_lo, f_hi) = slice_f.split_at(mid_elem);
        let (g_lo, g_hi) = slice_g.split_at(mid_elem);
        let (l, r) = scope.join(
            |s| reduce_recurse_gf128(f_lo, g_lo, base, s),
            |s| reduce_recurse_gf128(f_hi, g_hi, base, s),
        );
        (l.0 + r.0, l.1 + r.1, l.2 + r.2)
    }

    fn bind_recurse_gf128(
        src_f: &[GF128],
        src_g: &[GF128],
        dst_f: &mut [GF128],
        dst_g: &mut [GF128],
        r: GF128,
        base: usize,
        scope: &mut chili::Scope<'_>,
    ) {
        let n_pairs = src_f.len() / 2;
        if n_pairs <= base {
            for j in 0..n_pairs {
                let f0 = src_f[2 * j];
                let f1 = src_f[2 * j + 1];
                let g0 = src_g[2 * j];
                let g1 = src_g[2 * j + 1];
                dst_f[j] = f0 + r * (f1 - f0);
                dst_g[j] = g0 + r * (g1 - g0);
            }
            return;
        }
        let mid_pair = n_pairs / 2;
        let mid_elem = mid_pair * 2;
        let (f_lo, f_hi) = src_f.split_at(mid_elem);
        let (g_lo, g_hi) = src_g.split_at(mid_elem);
        let (df_lo, df_hi) = dst_f.split_at_mut(mid_pair);
        let (dg_lo, dg_hi) = dst_g.split_at_mut(mid_pair);
        scope.join(
            |s| bind_recurse_gf128(f_lo, g_lo, df_lo, dg_lo, r, base, s),
            |s| bind_recurse_gf128(f_hi, g_hi, df_hi, dg_hi, r, base, s),
        );
    }

    pub fn sumcheck_deg2_delayed_gf128_par2_chili(
        f: &mut Vec<GF128>,
        g: &mut Vec<GF128>,
        challenges: &[GF128],
        base: usize,
    ) {
        let mut scratch_f: Vec<GF128> = Vec::new();
        let mut scratch_g: Vec<GF128> = Vec::new();

        for round in 0..challenges.len() {
            let half = f.len() / 2;
            let r = challenges[round];

            let mut scope = chili::Scope::global();
            let triple = reduce_recurse_gf128(f, g, base, &mut scope);
            black_box(triple);

            scratch_f.resize(half, GF128::ZERO);
            scratch_g.resize(half, GF128::ZERO);
            {
                let f_src: &[GF128] = f;
                let g_src: &[GF128] = g;
                bind_recurse_gf128(
                    f_src,
                    g_src,
                    &mut scratch_f,
                    &mut scratch_g,
                    r,
                    base,
                    &mut scope,
                );
            }

            std::mem::swap(f, &mut scratch_f);
            std::mem::swap(g, &mut scratch_g);
            f.truncate(half);
            g.truncate(half);
        }
    }

    // --------- Fp128 recursion helpers ---------

    fn reduce_recurse_fp128(
        slice_f: &[Fp128],
        slice_g: &[Fp128],
        base: usize,
        scope: &mut chili::Scope<'_>,
    ) -> (Fp128, Fp128, Fp128) {
        let n_pairs = slice_f.len() / 2;
        if n_pairs <= base {
            let mut h0 = Fp128Accum::zero();
            let mut h1 = Fp128Accum::zero();
            let mut h_inf = Fp128Accum::zero();
            for j in 0..n_pairs {
                let f0 = slice_f[2 * j];
                let f1 = slice_f[2 * j + 1];
                let g0 = slice_g[2 * j];
                let g1 = slice_g[2 * j + 1];
                let df = f1 - f0;
                let dg = g1 - g0;
                h0.fmadd(f0, g0);
                h1.fmadd(f1, g1);
                h_inf.fmadd(df, dg);
            }
            return (h0.reduce(), h1.reduce(), h_inf.reduce());
        }
        let mid_elem = (n_pairs / 2) * 2;
        let (f_lo, f_hi) = slice_f.split_at(mid_elem);
        let (g_lo, g_hi) = slice_g.split_at(mid_elem);
        let (l, r) = scope.join(
            |s| reduce_recurse_fp128(f_lo, g_lo, base, s),
            |s| reduce_recurse_fp128(f_hi, g_hi, base, s),
        );
        (l.0 + r.0, l.1 + r.1, l.2 + r.2)
    }

    fn bind_recurse_fp128(
        src_f: &[Fp128],
        src_g: &[Fp128],
        dst_f: &mut [Fp128],
        dst_g: &mut [Fp128],
        r: Fp128,
        base: usize,
        scope: &mut chili::Scope<'_>,
    ) {
        let n_pairs = src_f.len() / 2;
        if n_pairs <= base {
            for j in 0..n_pairs {
                let f0 = src_f[2 * j];
                let f1 = src_f[2 * j + 1];
                let g0 = src_g[2 * j];
                let g1 = src_g[2 * j + 1];
                dst_f[j] = (f1 - f0).mul_add(r, f0);
                dst_g[j] = (g1 - g0).mul_add(r, g0);
            }
            return;
        }
        let mid_pair = n_pairs / 2;
        let mid_elem = mid_pair * 2;
        let (f_lo, f_hi) = src_f.split_at(mid_elem);
        let (g_lo, g_hi) = src_g.split_at(mid_elem);
        let (df_lo, df_hi) = dst_f.split_at_mut(mid_pair);
        let (dg_lo, dg_hi) = dst_g.split_at_mut(mid_pair);
        scope.join(
            |s| bind_recurse_fp128(f_lo, g_lo, df_lo, dg_lo, r, base, s),
            |s| bind_recurse_fp128(f_hi, g_hi, df_hi, dg_hi, r, base, s),
        );
    }

    pub fn sumcheck_deg2_delayed_fp128_par2_chili(
        f: &mut Vec<Fp128>,
        g: &mut Vec<Fp128>,
        challenges: &[Fp128],
        base: usize,
    ) {
        let mut scratch_f: Vec<Fp128> = Vec::new();
        let mut scratch_g: Vec<Fp128> = Vec::new();

        for round in 0..challenges.len() {
            let half = f.len() / 2;
            let r = challenges[round];

            let mut scope = chili::Scope::global();
            let triple = reduce_recurse_fp128(f, g, base, &mut scope);
            black_box(triple);

            scratch_f.resize(half, Fp128::ZERO);
            scratch_g.resize(half, Fp128::ZERO);
            {
                let f_src: &[Fp128] = f;
                let g_src: &[Fp128] = g;
                bind_recurse_fp128(
                    f_src,
                    g_src,
                    &mut scratch_f,
                    &mut scratch_g,
                    r,
                    base,
                    &mut scope,
                );
            }

            std::mem::swap(f, &mut scratch_f);
            std::mem::swap(g, &mut scratch_g);
            f.truncate(half);
            g.truncate(half);
        }
    }
}

#[cfg(feature = "parallel_chili")]
pub use chili_impl::{
    sumcheck_deg2_delayed_fp128_par2_chili, sumcheck_deg2_delayed_gf128_par2_chili,
};

// ===========================================================================
// Approach 3: persistent pool + atomic barrier
// ===========================================================================

// Per-round sync layout:
//
//   reduce_counter.fetch_add(1, Release)  // every worker after reduce
//   main: spin until counter == (r+1) * n_workers, combine, black_box,
//         store bind_go = r + 1 (Release)
//   workers: spin until bind_go >= r + 1 (Acquire), then bind
//   (no bind barrier)
//
// No bind barrier is needed in the common (power-of-2) case: at round r,
// worker i writes pair indices `[i*C, (i+1)*C)` where C = live_pairs/n_workers.
// At round r+1, live_pairs halves to live_pairs/2 so the new chunk is
// `C/2 = live_pairs/(2*n_workers)`, and worker i reads pair indices
// `[i*(C/2), (i+1)*(C/2))` in pair units, which is elements
// `[i*C, (i+1)*C)` in the (now-read) buffer. Exactly the range worker i
// itself wrote last round. Self-read → no cross-worker hazard.
//
// The live_pairs-not-divisible-by-n_workers case is handled by bailing out
// of the scope early (via `par3_scope_rounds`) before alignment breaks.

fn partial_triple_gf128(
    rf: *const GF128,
    rg: *const GF128,
    lo: usize,
    n_pairs: usize,
) -> (GF128, GF128, GF128) {
    let mut h0 = GF128Accum::zero();
    let mut h1 = GF128Accum::zero();
    let mut h_inf = GF128Accum::zero();
    unsafe {
        for j in 0..n_pairs {
            let f0 = *rf.add((lo + j) * 2);
            let f1 = *rf.add((lo + j) * 2 + 1);
            let g0 = *rg.add((lo + j) * 2);
            let g1 = *rg.add((lo + j) * 2 + 1);
            let df = f1 - f0;
            let dg = g1 - g0;
            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(df, dg);
        }
    }
    (h0.reduce(), h1.reduce(), h_inf.reduce())
}

fn bind_chunk_gf128(
    rf: *const GF128,
    rg: *const GF128,
    wf: *mut GF128,
    wg: *mut GF128,
    lo: usize,
    n_pairs: usize,
    r: GF128,
) {
    unsafe {
        for j in 0..n_pairs {
            let f0 = *rf.add((lo + j) * 2);
            let f1 = *rf.add((lo + j) * 2 + 1);
            let g0 = *rg.add((lo + j) * 2);
            let g1 = *rg.add((lo + j) * 2 + 1);
            *wf.add(lo + j) = f0 + r * (f1 - f0);
            *wg.add(lo + j) = g0 + r * (g1 - g0);
        }
    }
}

fn partial_triple_fp128(
    rf: *const Fp128,
    rg: *const Fp128,
    lo: usize,
    n_pairs: usize,
) -> (Fp128, Fp128, Fp128) {
    let mut h0 = Fp128Accum::zero();
    let mut h1 = Fp128Accum::zero();
    let mut h_inf = Fp128Accum::zero();
    unsafe {
        for j in 0..n_pairs {
            let f0 = *rf.add((lo + j) * 2);
            let f1 = *rf.add((lo + j) * 2 + 1);
            let g0 = *rg.add((lo + j) * 2);
            let g1 = *rg.add((lo + j) * 2 + 1);
            let df = f1 - f0;
            let dg = g1 - g0;
            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(df, dg);
        }
    }
    (h0.reduce(), h1.reduce(), h_inf.reduce())
}

fn bind_chunk_fp128(
    rf: *const Fp128,
    rg: *const Fp128,
    wf: *mut Fp128,
    wg: *mut Fp128,
    lo: usize,
    n_pairs: usize,
    r: Fp128,
) {
    unsafe {
        for j in 0..n_pairs {
            let f0 = *rf.add((lo + j) * 2);
            let f1 = *rf.add((lo + j) * 2 + 1);
            let g0 = *rg.add((lo + j) * 2);
            let g1 = *rg.add((lo + j) * 2 + 1);
            *wf.add(lo + j) = (f1 - f0).mul_add(r, f0);
            *wg.add(lo + j) = (g1 - g0).mul_add(r, g0);
        }
    }
}

#[inline]
fn worker_chunk_range(live_pairs: usize, n_workers: usize, worker_idx: usize) -> (usize, usize) {
    let chunk = live_pairs.div_ceil(n_workers);
    let lo = (worker_idx * chunk).min(live_pairs);
    let hi = ((worker_idx + 1) * chunk).min(live_pairs);
    (lo, hi)
}

/// Busy-wait until `counter >= target`, with a bounded spin budget before
/// cooperatively yielding. macOS' QoS scheduler demotes threads that spin
/// heavily, which wrecks tail latency on heterogeneous-core silicon (M-series
/// P + E clusters). Yielding after a few thousand spins keeps the fast path
/// cheap while letting the OS keep threads on P cores.
#[inline]
fn spin_until_ge(counter: &AtomicUsize, target: usize) {
    // Roughly corresponds to ~1 µs of spinning on a 3-4 GHz core before the
    // first yield. Tuned empirically on M4 Max.
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

/// Persistent-pool + atomic-barrier implementation for GF128.
pub fn sumcheck_deg2_delayed_gf128_par3_persistent(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
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
    let n_workers = rayon::current_num_threads().max(1).min(initial_pairs.max(1));
    let scope_rounds = par3_scope_rounds(initial_pairs, n_workers, n_rounds);

    if scope_rounds == 0 {
        super::gf128::sumcheck_deg2_delayed_gf128(f, g, challenges);
        return;
    }

    let mut buf0_f: Vec<GF128> = std::mem::take(f);
    let mut buf0_g: Vec<GF128> = std::mem::take(g);
    let mut buf1_f: Vec<GF128> = vec![GF128::ZERO; initial_pairs];
    let mut buf1_g: Vec<GF128> = vec![GF128::ZERO; initial_pairs];

    let partials: SharedSlots<(GF128, GF128, GF128)> = SharedSlots(
        (0..n_workers)
            .map(|_| UnsafeCell::new((GF128::ZERO, GF128::ZERO, GF128::ZERO)))
            .collect(),
    );
    let reduce_counter = AtomicUsize::new(0);
    let bind_go = AtomicUsize::new(0);

    let buf0_f_ptr = SendPtr(buf0_f.as_mut_ptr());
    let buf0_g_ptr = SendPtr(buf0_g.as_mut_ptr());
    let buf1_f_ptr = SendPtr(buf1_f.as_mut_ptr());
    let buf1_g_ptr = SendPtr(buf1_g.as_mut_ptr());

    let partials_ref = &partials;
    let rc_ref = &reduce_counter;
    let bg_ref = &bind_go;

    let worker_body = |worker_idx: usize| {
        // Force the closure to capture the whole `SendPtr` wrappers (which are
        // Sync), not the inner `*mut T` (which is not).
        let buf0_f_ptr = buf0_f_ptr;
        let buf0_g_ptr = buf0_g_ptr;
        let buf1_f_ptr = buf1_f_ptr;
        let buf1_g_ptr = buf1_g_ptr;
        let rw_f: [SendPtr<GF128>; 2] = [buf0_f_ptr, buf1_f_ptr];
        let rw_g: [SendPtr<GF128>; 2] = [buf0_g_ptr, buf1_g_ptr];

        for round in 0..scope_rounds {
            let r = challenges[round];
            let live_pairs = initial_pairs >> round;
            let (lo, hi) = worker_chunk_range(live_pairs, n_workers, worker_idx);
            let this = hi - lo;

            let rf = rw_f[round & 1].0;
            let rg = rw_g[round & 1].0;
            let wf = rw_f[(round + 1) & 1].0;
            let wg = rw_g[(round + 1) & 1].0;

            let triple = if this > 0 {
                partial_triple_gf128(rf, rg, lo, this)
            } else {
                (GF128::ZERO, GF128::ZERO, GF128::ZERO)
            };
            unsafe {
                *partials_ref.0[worker_idx].get() = triple;
            }
            rc_ref.fetch_add(1, Ordering::Release);

            if worker_idx == 0 {
                spin_until_ge(rc_ref, (round + 1) * n_workers);
                let mut sum = (GF128::ZERO, GF128::ZERO, GF128::ZERO);
                for slot in partials_ref.0.iter() {
                    let (a, b, c) = unsafe { *slot.get() };
                    sum = (sum.0 + a, sum.1 + b, sum.2 + c);
                }
                black_box(sum);
                bg_ref.store(round + 1, Ordering::Release);
            } else {
                spin_until_ge(bg_ref, round + 1);
            }

            if this > 0 {
                bind_chunk_gf128(rf, rg, wf, wg, lo, this, r);
            }
        }
    };

    // Spawn all n_workers tasks and return immediately from the scope
    // closure. Rayon then "donates" the main thread to the pool for the
    // duration of the scope, so main helps drain the 16 tasks across 16
    // available workers. (If main instead spun inside the scope body waiting
    // on a barrier, the spawned tasks would only be stolen by the 15 other
    // pool workers, adding wake-up latency to every call. This was measured
    // empirically to add ~30 ms per call; see PARALLELISM.md.)
    rayon::scope(|s| {
        for worker_idx in 0..n_workers {
            s.spawn(move |_| worker_body(worker_idx));
        }
    });

    let (mut live_f, mut live_g) = if scope_rounds & 1 == 0 {
        (buf0_f, buf0_g)
    } else {
        (buf1_f, buf1_g)
    };
    let live_len = initial_len >> scope_rounds;
    live_f.truncate(live_len);
    live_g.truncate(live_len);
    *f = live_f;
    *g = live_g;

    if scope_rounds < n_rounds {
        super::gf128::sumcheck_deg2_delayed_gf128(f, g, &challenges[scope_rounds..]);
    }
}

/// Persistent-pool + atomic-barrier implementation for Fp128.
pub fn sumcheck_deg2_delayed_fp128_par3_persistent(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
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
    let n_workers = rayon::current_num_threads().max(1).min(initial_pairs.max(1));
    let scope_rounds = par3_scope_rounds(initial_pairs, n_workers, n_rounds);

    if scope_rounds == 0 {
        super::fp128::sumcheck_deg2_delayed_fp128(f, g, challenges);
        return;
    }

    let mut buf0_f: Vec<Fp128> = std::mem::take(f);
    let mut buf0_g: Vec<Fp128> = std::mem::take(g);
    let mut buf1_f: Vec<Fp128> = vec![Fp128::ZERO; initial_pairs];
    let mut buf1_g: Vec<Fp128> = vec![Fp128::ZERO; initial_pairs];

    let partials: SharedSlots<(Fp128, Fp128, Fp128)> = SharedSlots(
        (0..n_workers)
            .map(|_| UnsafeCell::new((Fp128::ZERO, Fp128::ZERO, Fp128::ZERO)))
            .collect(),
    );
    let reduce_counter = AtomicUsize::new(0);
    let bind_go = AtomicUsize::new(0);

    let buf0_f_ptr = SendPtr(buf0_f.as_mut_ptr());
    let buf0_g_ptr = SendPtr(buf0_g.as_mut_ptr());
    let buf1_f_ptr = SendPtr(buf1_f.as_mut_ptr());
    let buf1_g_ptr = SendPtr(buf1_g.as_mut_ptr());

    let partials_ref = &partials;
    let rc_ref = &reduce_counter;
    let bg_ref = &bind_go;

    let worker_body = |worker_idx: usize| {
        let buf0_f_ptr = buf0_f_ptr;
        let buf0_g_ptr = buf0_g_ptr;
        let buf1_f_ptr = buf1_f_ptr;
        let buf1_g_ptr = buf1_g_ptr;
        let rw_f: [SendPtr<Fp128>; 2] = [buf0_f_ptr, buf1_f_ptr];
        let rw_g: [SendPtr<Fp128>; 2] = [buf0_g_ptr, buf1_g_ptr];

        for round in 0..scope_rounds {
            let r = challenges[round];
            let live_pairs = initial_pairs >> round;
            let (lo, hi) = worker_chunk_range(live_pairs, n_workers, worker_idx);
            let this = hi - lo;

            let rf = rw_f[round & 1].0;
            let rg = rw_g[round & 1].0;
            let wf = rw_f[(round + 1) & 1].0;
            let wg = rw_g[(round + 1) & 1].0;

            let triple = if this > 0 {
                partial_triple_fp128(rf, rg, lo, this)
            } else {
                (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO)
            };
            unsafe {
                *partials_ref.0[worker_idx].get() = triple;
            }
            rc_ref.fetch_add(1, Ordering::Release);

            if worker_idx == 0 {
                spin_until_ge(rc_ref, (round + 1) * n_workers);
                let mut sum = (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO);
                for slot in partials_ref.0.iter() {
                    let (a, b, c) = unsafe { *slot.get() };
                    sum = (sum.0 + a, sum.1 + b, sum.2 + c);
                }
                black_box(sum);
                bg_ref.store(round + 1, Ordering::Release);
            } else {
                spin_until_ge(bg_ref, round + 1);
            }

            if this > 0 {
                bind_chunk_fp128(rf, rg, wf, wg, lo, this, r);
            }
        }
    };

    rayon::scope(|s| {
        for worker_idx in 0..n_workers {
            s.spawn(move |_| worker_body(worker_idx));
        }
    });

    let (mut live_f, mut live_g) = if scope_rounds & 1 == 0 {
        (buf0_f, buf0_g)
    } else {
        (buf1_f, buf1_g)
    };
    let live_len = initial_len >> scope_rounds;
    live_f.truncate(live_len);
    live_g.truncate(live_len);
    *f = live_f;
    *g = live_g;

    if scope_rounds < n_rounds {
        super::fp128::sumcheck_deg2_delayed_fp128(f, g, &challenges[scope_rounds..]);
    }
}

// ============================================================================
// Approach 4: Globally-persistent pinned pool with doorbell synchronization
// ============================================================================
//
// A single process-global pool of `std::thread` workers spawned lazily on
// first use. Each worker:
//   - Sets `QOS_CLASS_USER_INTERACTIVE` on macOS to prevent E-core demotion.
//   - Pure-spins on a Release/Acquire epoch counter waiting for a task.
//   - Runs the task, Release-increments a shared `done` counter, goes back
//     to spinning.
//
// Dispatch (`broadcast_scoped`): main stores a task pointer, Release-bumps
// the epoch, runs `f(0)` itself (main is worker 0), and spins on `done`.
// The transmute that laundered the task closure's lifetime is sound because
// the call blocks until every extra worker has returned from `task()`.
//
// Pool size defaults to `min(available_parallelism, 8)`, overridable via
// `SUMCHECK_PINNED_WORKERS`. At larger sizes on Apple Silicon, atomic
// contention on `reduce_counter` becomes the dominant cost at small `n`.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

/// Cache-line padding to keep hot atomics from false-sharing with each other.
/// 128 bytes covers typical sector-prefetch pairs (two 64B lines).
#[repr(align(128))]
struct CachePadded<T>(T);

type TaskPtr = *const (dyn Fn(usize) + Sync + 'static);

struct PoolShared {
    /// Monotonically increasing epoch. Main Release-bumps it to publish a
    /// new task; workers Acquire-spin waiting for `epoch > last_seen`.
    epoch: CachePadded<AtomicU64>,
    /// Count of extra (non-main) active workers that have finished the
    /// current task. Only workers with `worker_idx < n_active` bump this.
    done: CachePadded<AtomicUsize>,
    /// Number of workers active for the current dispatch, including main
    /// (always 1..=n_total). Set by main BEFORE bumping epoch; read by
    /// workers AFTER observing the epoch bump. Workers with
    /// `worker_idx >= n_active` skip both `task` and `done` for this
    /// dispatch (they just advance `last_epoch` and keep spinning).
    n_active: AtomicUsize,
    /// Shutdown flag set on drop.
    shutdown: AtomicBool,
    /// Scoped task pointer, published via the Release-bump on `epoch`.
    /// `None` during startup/shutdown.
    task: UnsafeCell<Option<TaskPtr>>,
}

// SAFETY: the only `!Send`/`!Sync` field is the `UnsafeCell<Option<TaskPtr>>`
// raw pointer. Access to it is protected by the epoch Release/Acquire pair
// and the done-counter broadcast barrier; no worker touches the pointer
// outside a dispatch window.
unsafe impl Send for PoolShared {}
unsafe impl Sync for PoolShared {}

/// A process-global pool of pinned worker threads used by the `_par4_pinned`
/// sumcheck variants.
pub struct PinnedPool {
    n_total: usize,
    shared: Arc<PoolShared>,
    workers: Vec<JoinHandle<()>>,
}

impl PinnedPool {
    /// Total number of workers available, including main (main = worker 0).
    #[inline]
    pub fn n_workers(&self) -> usize {
        self.n_total
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
    /// not touch the done counter, so the barrier contention cost scales
    /// with `n_active` rather than the full pool size. Blocks until every
    /// active worker has returned from `f`.
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

        // Publish n_active before the task pointer; both are synchronized
        // via the Release-bump on `epoch` below.
        self.shared
            .n_active
            .store(n_active, Ordering::Relaxed);

        // Launder the scoped lifetime. SAFETY: we block until every
        // active worker finishes `task()`, so `f` outlives all accesses.
        let f_static: &(dyn Fn(usize) + Sync + 'static) = unsafe {
            std::mem::transmute::<
                &(dyn Fn(usize) + Sync),
                &(dyn Fn(usize) + Sync + 'static),
            >(f)
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
    #[cfg(target_os = "macos")]
    unsafe {
        set_qos_user_interactive();
    }

    let mut last_epoch: u64 = 0;
    loop {
        // Spin-wait for a new epoch or shutdown. Under
        // `QOS_CLASS_USER_INTERACTIVE` on macOS the OS will not demote us
        // to E-cores, so pure `spin_loop` is safe. We still check
        // shutdown on each iteration.
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

        // Synchronized with the Release on `epoch`: n_active and task are
        // both visible. If this worker is inactive for the current
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

#[cfg(target_os = "macos")]
unsafe fn set_qos_user_interactive() {
    // QOS_CLASS_USER_INTERACTIVE from <pthread/qos.h>.
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

static PINNED_POOL: OnceLock<PinnedPool> = OnceLock::new();

/// Returns the process-global pinned worker pool, spawning it on first use.
///
/// Worker count is chosen as `min(available_parallelism, 8)` by default,
/// overridable via the `SUMCHECK_PINNED_WORKERS` environment variable.
pub fn pinned_pool() -> &'static PinnedPool {
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

/// Pinned-pool implementation for GF128 sumcheck.
pub fn sumcheck_deg2_delayed_gf128_par4_pinned(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
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

    let pool = pinned_pool();
    let pool_total = pool.n_workers();
    // Adaptive: keep ~`PAR4_TARGET_PAIRS_PER_WORKER` pairs per worker in
    // the first round. Below that, barrier contention on `done` dominates
    // real work.
    let n_workers = (initial_pairs / PAR4_TARGET_PAIRS_PER_WORKER)
        .max(2)
        .min(pool_total);
    let scope_rounds = par3_scope_rounds(initial_pairs, n_workers, n_rounds);

    if scope_rounds == 0 || n_workers <= 1 {
        super::gf128::sumcheck_deg2_delayed_gf128(f, g, challenges);
        return;
    }

    let mut buf0_f: Vec<GF128> = std::mem::take(f);
    let mut buf0_g: Vec<GF128> = std::mem::take(g);
    let mut buf1_f: Vec<GF128> = vec![GF128::ZERO; initial_pairs];
    let mut buf1_g: Vec<GF128> = vec![GF128::ZERO; initial_pairs];

    let partials: SharedSlots<(GF128, GF128, GF128)> = SharedSlots(
        (0..n_workers)
            .map(|_| UnsafeCell::new((GF128::ZERO, GF128::ZERO, GF128::ZERO)))
            .collect(),
    );
    let reduce_counter = AtomicUsize::new(0);
    let bind_go = AtomicUsize::new(0);

    let buf0_f_ptr = SendPtr(buf0_f.as_mut_ptr());
    let buf0_g_ptr = SendPtr(buf0_g.as_mut_ptr());
    let buf1_f_ptr = SendPtr(buf1_f.as_mut_ptr());
    let buf1_g_ptr = SendPtr(buf1_g.as_mut_ptr());

    let partials_ref = &partials;
    let rc_ref = &reduce_counter;
    let bg_ref = &bind_go;

    let worker_body = |worker_idx: usize| {
        // `broadcast_scoped(n_workers, ...)` guarantees we're only called
        // for `worker_idx < n_workers`; inactive pool workers never enter.
        let rw_f: [SendPtr<GF128>; 2] = [buf0_f_ptr, buf1_f_ptr];
        let rw_g: [SendPtr<GF128>; 2] = [buf0_g_ptr, buf1_g_ptr];

        for round in 0..scope_rounds {
            let r = challenges[round];
            let live_pairs = initial_pairs >> round;
            let (lo, hi) = worker_chunk_range(live_pairs, n_workers, worker_idx);
            let this = hi - lo;

            let rf = rw_f[round & 1].0;
            let rg = rw_g[round & 1].0;
            let wf = rw_f[(round + 1) & 1].0;
            let wg = rw_g[(round + 1) & 1].0;

            let triple = if this > 0 {
                partial_triple_gf128(rf, rg, lo, this)
            } else {
                (GF128::ZERO, GF128::ZERO, GF128::ZERO)
            };
            unsafe {
                *partials_ref.0[worker_idx].get() = triple;
            }
            rc_ref.fetch_add(1, Ordering::Release);

            if worker_idx == 0 {
                spin_until_ge(rc_ref, (round + 1) * n_workers);
                let mut sum = (GF128::ZERO, GF128::ZERO, GF128::ZERO);
                for slot in partials_ref.0.iter() {
                    let (a, b, c) = unsafe { *slot.get() };
                    sum = (sum.0 + a, sum.1 + b, sum.2 + c);
                }
                black_box(sum);
                bg_ref.store(round + 1, Ordering::Release);
            } else {
                spin_until_ge(bg_ref, round + 1);
            }

            if this > 0 {
                bind_chunk_gf128(rf, rg, wf, wg, lo, this, r);
            }
        }
    };

    pool.broadcast_scoped(n_workers, &worker_body);

    let (mut live_f, mut live_g) = if scope_rounds & 1 == 0 {
        (buf0_f, buf0_g)
    } else {
        (buf1_f, buf1_g)
    };
    let live_len = initial_len >> scope_rounds;
    live_f.truncate(live_len);
    live_g.truncate(live_len);
    *f = live_f;
    *g = live_g;

    if scope_rounds < n_rounds {
        super::gf128::sumcheck_deg2_delayed_gf128(f, g, &challenges[scope_rounds..]);
    }
}

/// Pinned-pool implementation for Fp128 sumcheck.
pub fn sumcheck_deg2_delayed_fp128_par4_pinned(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
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

    let pool = pinned_pool();
    let pool_total = pool.n_workers();
    // Adaptive: keep ~`PAR4_TARGET_PAIRS_PER_WORKER` pairs per worker in
    // the first round. Fp128's per-pair work is heavier than GF128's, so
    // the same threshold parallelises earlier in absolute wall time.
    let n_workers = (initial_pairs / PAR4_TARGET_PAIRS_PER_WORKER)
        .max(2)
        .min(pool_total);
    let scope_rounds = par3_scope_rounds(initial_pairs, n_workers, n_rounds);

    if scope_rounds == 0 || n_workers <= 1 {
        super::fp128::sumcheck_deg2_delayed_fp128(f, g, challenges);
        return;
    }

    let mut buf0_f: Vec<Fp128> = std::mem::take(f);
    let mut buf0_g: Vec<Fp128> = std::mem::take(g);
    let mut buf1_f: Vec<Fp128> = vec![Fp128::ZERO; initial_pairs];
    let mut buf1_g: Vec<Fp128> = vec![Fp128::ZERO; initial_pairs];

    let partials: SharedSlots<(Fp128, Fp128, Fp128)> = SharedSlots(
        (0..n_workers)
            .map(|_| UnsafeCell::new((Fp128::ZERO, Fp128::ZERO, Fp128::ZERO)))
            .collect(),
    );
    let reduce_counter = AtomicUsize::new(0);
    let bind_go = AtomicUsize::new(0);

    let buf0_f_ptr = SendPtr(buf0_f.as_mut_ptr());
    let buf0_g_ptr = SendPtr(buf0_g.as_mut_ptr());
    let buf1_f_ptr = SendPtr(buf1_f.as_mut_ptr());
    let buf1_g_ptr = SendPtr(buf1_g.as_mut_ptr());

    let partials_ref = &partials;
    let rc_ref = &reduce_counter;
    let bg_ref = &bind_go;

    let worker_body = |worker_idx: usize| {
        let rw_f: [SendPtr<Fp128>; 2] = [buf0_f_ptr, buf1_f_ptr];
        let rw_g: [SendPtr<Fp128>; 2] = [buf0_g_ptr, buf1_g_ptr];

        for round in 0..scope_rounds {
            let r = challenges[round];
            let live_pairs = initial_pairs >> round;
            let (lo, hi) = worker_chunk_range(live_pairs, n_workers, worker_idx);
            let this = hi - lo;

            let rf = rw_f[round & 1].0;
            let rg = rw_g[round & 1].0;
            let wf = rw_f[(round + 1) & 1].0;
            let wg = rw_g[(round + 1) & 1].0;

            let triple = if this > 0 {
                partial_triple_fp128(rf, rg, lo, this)
            } else {
                (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO)
            };
            unsafe {
                *partials_ref.0[worker_idx].get() = triple;
            }
            rc_ref.fetch_add(1, Ordering::Release);

            if worker_idx == 0 {
                spin_until_ge(rc_ref, (round + 1) * n_workers);
                let mut sum = (Fp128::ZERO, Fp128::ZERO, Fp128::ZERO);
                for slot in partials_ref.0.iter() {
                    let (a, b, c) = unsafe { *slot.get() };
                    sum = (sum.0 + a, sum.1 + b, sum.2 + c);
                }
                black_box(sum);
                bg_ref.store(round + 1, Ordering::Release);
            } else {
                spin_until_ge(bg_ref, round + 1);
            }

            if this > 0 {
                bind_chunk_fp128(rf, rg, wf, wg, lo, this, r);
            }
        }
    };

    pool.broadcast_scoped(n_workers, &worker_body);

    let (mut live_f, mut live_g) = if scope_rounds & 1 == 0 {
        (buf0_f, buf0_g)
    } else {
        (buf1_f, buf1_g)
    };
    let live_len = initial_len >> scope_rounds;
    live_f.truncate(live_len);
    live_g.truncate(live_len);
    *f = live_f;
    *g = live_g;

    if scope_rounds < n_rounds {
        super::fp128::sumcheck_deg2_delayed_fp128(f, g, &challenges[scope_rounds..]);
    }
}
