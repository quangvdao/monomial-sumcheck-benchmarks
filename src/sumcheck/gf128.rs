use super::*;

// On aarch64 (macOS M-class) we drive a hand-rolled deferred-reduction
// FMA accumulator using NEON `vmull_p64` (see `GF128Accum` below), so
// we never touch `GF128::ZERO` and don't need `Field` in scope.
//
// On non-aarch64 (x86_64) we fall back to a simpler accumulator that
// just calls `a * b` and accumulates into a `GF128`. With the repo's
// `.cargo/config.toml` (`target-cpu=native`) that mul dispatches to
// binius-field's `packed_ghash_128` / `packed_ghash_256` SIMD path
// (PCLMULQDQ / VPCLMULQDQ-256), so this is *not* a u128 software
// multiply, just one without the deferred-reduction trick the NEON
// path uses. Without the build flag binius reverts to a portable
// u128 scalar mul, which is ~5-10x slower; see `PARALLELISM.md`.
#[cfg(not(target_arch = "aarch64"))]
use binius_field::Field;

#[cfg(target_arch = "aarch64")]
const GF128_POLY: u64 = 0x87;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn gf2_128_reduce_u128(mut t0: u128, t1: u128) -> u128 {
    t0 ^= t1 << 64;
    let t1_hi = (t1 >> 64) as u64;
    let correction: u128 = unsafe { vmull_p64(t1_hi, GF128_POLY) };
    t0 ^= correction;
    t0
}

#[cfg(target_arch = "aarch64")]
pub(super) struct GF128Accum {
    low: uint64x2_t,
    mid: uint64x2_t,
    high: uint64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl GF128Accum {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        unsafe {
            Self {
                low: vdupq_n_u64(0),
                mid: vdupq_n_u64(0),
                high: vdupq_n_u64(0),
            }
        }
    }

    #[inline(always)]
    pub(super) fn fmadd(&mut self, a: GF128, b: GF128) {
        unsafe {
            let a_val = a.val();
            let b_val = b.val();
            let a_lo = a_val as u64;
            let a_hi = (a_val >> 64) as u64;
            let b_lo = b_val as u64;
            let b_hi = (b_val >> 64) as u64;

            let t0: uint64x2_t = std::mem::transmute(vmull_p64(a_lo, b_lo));
            let t1a: uint64x2_t = std::mem::transmute(vmull_p64(a_hi, b_lo));
            let t1b: uint64x2_t = std::mem::transmute(vmull_p64(a_lo, b_hi));
            let t2: uint64x2_t = std::mem::transmute(vmull_p64(a_hi, b_hi));

            self.low = veorq_u64(self.low, t0);
            self.mid = veorq_u64(veorq_u64(self.mid, t1a), t1b);
            self.high = veorq_u64(self.high, t2);
        }
    }

    pub(super) fn reduce(self) -> GF128 {
        unsafe {
            let mid: u128 = std::mem::transmute(self.mid);
            let high: u128 = std::mem::transmute(self.high);
            let low: u128 = std::mem::transmute(self.low);
            let mid_reduced = gf2_128_reduce_u128(mid, high);
            let result = gf2_128_reduce_u128(low, mid_reduced);
            GF128::new(result)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(super) struct GF128Accum {
    sum: GF128,
}

#[cfg(not(target_arch = "aarch64"))]
impl GF128Accum {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self { sum: GF128::ZERO }
    }

    #[inline(always)]
    pub(super) fn fmadd(&mut self, a: GF128, b: GF128) {
        self.sum += a * b;
    }

    pub(super) fn reduce(self) -> GF128 {
        self.sum
    }
}

pub fn sumcheck_deg2_delayed_gf128(f: &mut Vec<GF128>, g: &mut Vec<GF128>, challenges: &[GF128]) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = GF128Accum::zero();
        let mut h1 = GF128Accum::zero();
        let mut h_inf = GF128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            let df = f1 - f0;
            let dg = g1 - g0;

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(df, dg);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            f[j] = f0 + r * (f1 - f0);
            g[j] = g0 + r * (g1 - g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

/// Same sum-check relation as [`sumcheck_deg2_delayed_gf128`], with
/// bind-of-previous-round and reduce-of-current-round fused into one
/// pass. Mirrors the jolt-cpp GPU `bind_eval_roundN_kernel` shape;
/// see `docs/plans/sumcheck-cpu-platform.md` Phase 1 for the
/// derivation and the Fp128 twin (`sumcheck_deg2_delayed_fp128_fused`)
/// for the rationale.
pub fn sumcheck_deg2_delayed_gf128_fused(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
) {
    let n_rounds = challenges.len();
    if n_rounds == 0 {
        return;
    }

    {
        let half = f.len() / 2;
        let mut h0 = GF128Accum::zero();
        let mut h1 = GF128Accum::zero();
        let mut h_inf = GF128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(f1 - f0, g1 - g0);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    for r_idx in 1..n_rounds {
        let r_prev = challenges[r_idx - 1];
        let new_half = f.len() / 4;

        let mut h0 = GF128Accum::zero();
        let mut h1 = GF128Accum::zero();
        let mut h_inf = GF128Accum::zero();

        for j in 0..new_half {
            let f00 = f[4 * j];
            let f01 = f[4 * j + 1];
            let f10 = f[4 * j + 2];
            let f11 = f[4 * j + 3];
            let g00 = g[4 * j];
            let g01 = g[4 * j + 1];
            let g10 = g[4 * j + 2];
            let g11 = g[4 * j + 3];

            let f0 = f00 + r_prev * (f01 - f00);
            let f1 = f10 + r_prev * (f11 - f10);
            let g0 = g00 + r_prev * (g01 - g00);
            let g1 = g10 + r_prev * (g11 - g10);

            // In-place write is safe: iteration j reads positions
            // [4j, 4j+4) and writes [2j, 2j+2); 4j > 2j+1 for j >= 1,
            // and for j = 0 both reads complete before either write.
            f[2 * j] = f0;
            f[2 * j + 1] = f1;
            g[2 * j] = g0;
            g[2 * j + 1] = g1;

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(f1 - f0, g1 - g0);
        }

        f.truncate(2 * new_half);
        g.truncate(2 * new_half);

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    let r_last = challenges[n_rounds - 1];
    let half = f.len() / 2;
    for j in 0..half {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let g0 = g[2 * j];
        let g1 = g[2 * j + 1];
        f[j] = f0 + r_last * (f1 - f0);
        g[j] = g0 + r_last * (g1 - g0);
    }
    f.truncate(half);
    g.truncate(half);
}

pub fn sumcheck_deg2_eq_delayed_gf128(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    suffix_eq: &[Vec<GF128>],
    challenges: &[GF128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = GF128Accum::zero();
        let mut q_inf = GF128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            let ew = eq_rest[j];
            let df = f1 - f0;
            let dg = g1 - g0;

            q1.fmadd(f1 * g1, ew);
            q_inf.fmadd(df * dg, ew);
        }

        black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            f[j] = f0 + r * (f1 - f0);
            g[j] = g0 + r * (g1 - g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_delayed_gf128(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    challenges: &[GF128],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = GF128Accum::zero();
        let mut h1 = GF128Accum::zero();
        let mut h_inf = GF128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            h0.fmadd(f0, g0);
            h_inf.fmadd(fi, gi);
            let sf = f0 + fi;
            let sg = g0 + gi;
            h1.fmadd(sf, sg);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = f0 + r * fi;
            g[j] = g0 + r * gi;
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_delayed_gf128(
    f: &mut Vec<GF128>,
    g: &mut Vec<GF128>,
    suffix_eq: &[Vec<GF128>],
    challenges: &[GF128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = GF128Accum::zero();
        let mut q_inf = GF128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q_inf.fmadd(fi * gi, ew);
            let sf = f0 + fi;
            let sg = g0 + gi;
            q1.fmadd(sf * sg, ew);
        }

        black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = f0 + r * fi;
            g[j] = g0 + r * gi;
        }
        f.truncate(half);
        g.truncate(half);
    }
}
