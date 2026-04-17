use super::*;

pub fn sumcheck_deg2_projective_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
    zero: Fp128,
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = zero;
        let mut h1 = zero;
        let mut h_inf = zero;

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            h0 += f0 * g0;
            h_inf += fi * gi;
            let sf = f0 + fi;
            let sg = g0 + gi;
            h1 += sf * sg;
        }

        black_box((h0, h1, h_inf));

        let r = challenges[round];

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_1inf_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
    zero: Fp128,
) {
    let one = Fp128::one();
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = zero;
        let mut h1 = zero;
        let mut h_inf = zero;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let f0 = f1 - fi;
            let g0 = g1 - gi;

            h0 += f0 * g0;
            h1 += f1 * g1;
            h_inf += fi * gi;
        }

        black_box((h0, h1, h_inf));

        let r = challenges[round];
        let r_minus_one = r - one;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

#[inline(always)]
fn sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(
    f: &[Fp128],
    g: &[Fp128],
    eq_rest: &[Fp128],
    zero: Fp128,
) -> (Fp128, Fp128) {
    let half = f.len() / 2;
    let mut q0 = zero;
    let mut q1 = zero;

    for j in 0..half {
        let f1 = f[2 * j];
        let fi = f[2 * j + 1];
        let g1 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];
        let f0 = f1 - fi;
        let g0 = g1 - gi;

        q0 += f0 * g0 * ew;
        q1 += f1 * g1 * ew;
    }

    (q0, q1)
}

pub fn init_sumcheck_deg2_eq_gruen_projective_1inf_fp128_claim(
    f: &[Fp128],
    g: &[Fp128],
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    zero: Fp128,
) -> Fp128 {
    let (q0, q1) =
        sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, &suffix_eq[1], zero);
    let w = eq_point[0];
    let one_minus_w = Fp128::one() - w;
    q1.mul_add(w, one_minus_w * q0)
}

pub fn sumcheck_deg2_eq_gruen_projective_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
    zero: Fp128,
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = zero;
        let mut q_inf = zero;

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q_inf += fi * gi * ew;
            let sf = f0 + fi;
            let sg = g0 + gi;
            q1 += sf * sg * ew;
        }

        black_box((q1, q_inf));

        let r = challenges[round];

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_gruen_projective_1inf_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    challenges: &[Fp128],
    initial_claim: Fp128,
    zero: Fp128,
) {
    let one = Fp128::one();
    let mut claim = initial_claim;
    let mut current_scalar = one;
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = zero;
        let mut q_inf = zero;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q1 += f1 * g1 * ew;
            q_inf += fi * gi * ew;
        }

        let w = eq_point[round];
        let one_minus_w = one - w;
        let q0 = if one_minus_w == zero || current_scalar == zero {
            sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, eq_rest, zero).0
        } else {
            let normalized_claim = claim * current_scalar.inv_or_zero();
            (normalized_claim - w * q1) * one_minus_w.inv_or_zero()
        };

        black_box((q0, q1, q_inf));

        let r = challenges[round];
        let r_minus_one = r - one;
        let r_times_r_minus_one = r * r_minus_one;
        let q_r = q_inf.mul_add(r_times_r_minus_one, (q1 - q0).mul_add(r, q0));
        let eq_eval = (w - one_minus_w).mul_add(r, one_minus_w);

        current_scalar = current_scalar * eq_eval;
        claim = current_scalar * q_r;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

/// Compute `a + (b - a) * r` for the fused bind+reduce loop.
///
/// Currently just forwards to [`Fp128::mul_add`]. A pure-Rust
/// `mul_wide + solinas_reduce` variant was tried (the "delayed
/// reduction" shape that [`Fp128Accum::fmadd`] uses for the reduce
/// side) to give LLVM's scheduler freedom to interleave widening
/// muls across iterations. It measured 5 - 15 % slower than the
/// `mul_add_raw_aarch64` asm block because the asm block saves
/// ~17 instructions via fused carry chains and the `ccmp`
/// canonicalize trick, which outweighs the scheduling flexibility.
/// Kept as a helper so any future field backend can override the
/// bind shape (e.g. SIMD Fp128 via NEON would want to diverge from
/// the scalar `mul_add`).
///
/// See `docs/notes/fused-bind-eval-ab.md` for the A/B analysis.
#[inline(always)]
pub(super) fn fp128_bind(a: Fp128, b: Fp128, r: Fp128) -> Fp128 {
    (b - a).mul_add(r, a)
}

pub(super) struct Fp128Accum([u128; 4]);

impl Fp128Accum {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self([0u128; 4])
    }

    #[inline(always)]
    pub(super) fn fmadd(&mut self, a: Fp128, b: Fp128) {
        let product = a.mul_wide(b);
        for i in 0..4 {
            self.0[i] += product[i] as u128;
        }
    }

    pub(super) fn reduce(self) -> Fp128 {
        let mut limbs = [0u64; 5];
        let mut carry: u128 = 0;
        for i in 0..4 {
            let sum = self.0[i] + carry;
            limbs[i] = sum as u64;
            carry = sum >> 64;
        }
        limbs[4] = carry as u64;
        Fp128::solinas_reduce(&limbs)
    }
}

pub fn sumcheck_deg2_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

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
            f[j] = (f1 - f0).mul_add(r, f0);
            g[j] = (g1 - g0).mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

/// Same sum-check relation as [`sumcheck_deg2_delayed_fp128`], with
/// bind-of-previous-round and reduce-of-current-round fused into one
/// pass. The classic jolt-cpp GPU shape (see
/// `bind_eval_roundN_kernel`): each fused iteration reads 4
/// round-(r-1) elements per output pair, binds them with
/// `challenges[r - 1]` to produce 2 round-r elements, writes them in
/// place, and accumulates the round-r eval partial from the
/// just-bound pair before moving on. Round 0 stays a pure reduce
/// pass (no prior challenge), and the last challenge binds without a
/// following reduce.
///
/// Saves one read per round-r pair (the unfused path would re-read
/// the freshly bound round-r state during its separate reduce pass).
/// See `docs/plans/sumcheck-cpu-platform.md` Phase 1 for the derivation.
pub fn sumcheck_deg2_delayed_fp128_fused(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let n_rounds = challenges.len();
    if n_rounds == 0 {
        return;
    }

    {
        let half = f.len() / 2;
        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

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

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..new_half {
            let f00 = f[4 * j];
            let f01 = f[4 * j + 1];
            let f10 = f[4 * j + 2];
            let f11 = f[4 * j + 3];
            let g00 = g[4 * j];
            let g01 = g[4 * j + 1];
            let g10 = g[4 * j + 2];
            let g11 = g[4 * j + 3];

            let f0 = fp128_bind(f00, f01, r_prev);
            let f1 = fp128_bind(f10, f11, r_prev);
            let g0 = fp128_bind(g00, g01, r_prev);
            let g1 = fp128_bind(g10, g11, r_prev);

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
        f[j] = fp128_bind(f[2 * j], f[2 * j + 1], r_last);
        g[j] = fp128_bind(g[2 * j], g[2 * j + 1], r_last);
    }
    f.truncate(half);
    g.truncate(half);
}

pub fn sumcheck_deg2_eq_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

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
            f[j] = (f1 - f0).mul_add(r, f0);
            g[j] = (g1 - g0).mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

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
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_1inf_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let one = Fp128::one();
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let f0 = f1 - fi;
            let g0 = g1 - gi;

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(fi, gi);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        let r_minus_one = r - one;
        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

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
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_1inf_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    challenges: &[Fp128],
    initial_claim: Fp128,
) {
    let one = Fp128::one();
    let mut claim = initial_claim;
    let mut current_scalar = one;
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q1.fmadd(f1 * g1, ew);
            q_inf.fmadd(fi * gi, ew);
        }

        let q1 = q1.reduce();
        let q_inf = q_inf.reduce();
        let w = eq_point[round];
        let one_minus_w = one - w;
        let q0 = if one_minus_w == Fp128::ZERO || current_scalar == Fp128::ZERO {
            sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, eq_rest, Fp128::ZERO).0
        } else {
            let normalized_claim = claim * current_scalar.inv_or_zero();
            (normalized_claim - w * q1) * one_minus_w.inv_or_zero()
        };

        black_box((q0, q1, q_inf));

        let r = challenges[round];
        let r_minus_one = r - one;
        let r_times_r_minus_one = r * r_minus_one;
        let q_r = q_inf.mul_add(r_times_r_minus_one, (q1 - q0).mul_add(r, q0));
        let eq_eval = (w - one_minus_w).mul_add(r, one_minus_w);

        current_scalar = current_scalar * eq_eval;
        claim = current_scalar * q_r;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }
        f.truncate(half);
        g.truncate(half);
    }
}
