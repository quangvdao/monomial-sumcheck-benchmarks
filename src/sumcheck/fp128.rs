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

struct Fp128Accum([u128; 4]);

impl Fp128Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self([0u128; 4])
    }

    #[inline(always)]
    fn fmadd(&mut self, a: Fp128, b: Fp128) {
        let product = a.mul_wide(b);
        for i in 0..4 {
            self.0[i] += product[i] as u128;
        }
    }

    fn reduce(self) -> Fp128 {
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
