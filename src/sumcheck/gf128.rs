use super::*;

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
