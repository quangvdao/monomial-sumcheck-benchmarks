use super::*;

const BN254_MODULUS: [u64; 4] = <FrConfig as MontConfig<4>>::MODULUS.0;
const BN254_INV: u64 = <FrConfig as MontConfig<4>>::INV;
const BN254_R2: [u64; 4] = <FrConfig as MontConfig<4>>::R2.0;

#[inline(always)]
fn mac_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128) * (c as u128) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

#[inline(always)]
fn adc(a: &mut u64, b: u64, carry: u64) -> u64 {
    let tmp = (*a as u128) + (b as u128) + (carry as u128);
    *a = tmp as u64;
    (tmp >> 64) as u64
}

#[inline(always)]
fn sbb(a: &mut u64, b: u64, borrow: u64) -> u64 {
    let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
    *a = tmp as u64;
    u64::from(tmp >> 64 == 0)
}

fn cmp4(a: [u64; 4], b: [u64; 4]) -> std::cmp::Ordering {
    let mut i = 4;
    while i > 0 {
        i -= 1;
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    std::cmp::Ordering::Equal
}

fn sub4(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut r = a;
    let mut borrow = 0u64;
    borrow = sbb(&mut r[0], b[0], borrow);
    borrow = sbb(&mut r[1], b[1], borrow);
    borrow = sbb(&mut r[2], b[2], borrow);
    let _ = sbb(&mut r[3], b[3], borrow);
    r
}

const _: () = assert!(std::mem::size_of::<BN254Fr>() == 32);

#[inline(always)]
fn bn254_to_limbs(x: BN254Fr) -> [u64; 4] {
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bn254_from_limbs(limbs: [u64; 4]) -> BN254Fr {
    unsafe { std::mem::transmute(limbs) }
}

fn montgomery_reduce_8(t: &[u64; 8]) -> BN254Fr {
    let mut buf = *t;
    let mut carry2 = 0u64;
    for i in 0..4 {
        let m = buf[i].wrapping_mul(BN254_INV);
        let mut carry = 0u64;
        let _ = mac_carry(buf[i], m, BN254_MODULUS[0], &mut carry);
        for j in 1..4 {
            buf[i + j] = mac_carry(buf[i + j], m, BN254_MODULUS[j], &mut carry);
        }
        carry2 = adc(&mut buf[i + 4], carry, carry2);
    }
    let mut result = [buf[4], buf[5], buf[6], buf[7]];
    let needs_sub = if BN254_MODULUS[3] >> 63 == 0 {
        cmp4(result, BN254_MODULUS) != std::cmp::Ordering::Less
    } else {
        carry2 != 0 || cmp4(result, BN254_MODULUS) != std::cmp::Ordering::Less
    };
    if needs_sub {
        result = sub4(result, BN254_MODULUS);
    }
    bn254_from_limbs(result)
}

struct BN254Accum([u128; 8]);

impl BN254Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self([0u128; 8])
    }

    #[inline(always)]
    fn fmadd(&mut self, a: BN254Fr, b: BN254Fr) {
        let a = bn254_to_limbs(a);
        let b = bn254_to_limbs(b);
        for i in 0..4 {
            for j in 0..4 {
                let p = (a[i] as u128) * (b[j] as u128);
                self.0[i + j] += (p as u64) as u128;
                self.0[i + j + 1] += (p >> 64) as u128;
            }
        }
    }

    fn reduce(self) -> BN254Fr {
        let mut limbs = [0u64; 9];
        let mut carry: u128 = 0;
        for i in 0..8 {
            let sum = self.0[i] + carry;
            limbs[i] = sum as u64;
            carry = sum >> 64;
        }
        limbs[8] = carry as u64;

        let mut buf = [0u64; 8];
        buf.copy_from_slice(&limbs[..8]);
        let result = montgomery_reduce_8(&buf);

        if limbs[8] != 0 {
            let r_field = bn254_from_limbs(BN254_R2);
            result + BN254Fr::from(limbs[8]) * r_field
        } else {
            result
        }
    }
}

pub trait Bn254UpperLimbMul {
    fn mul_by_hi_2limbs(self, limb_lo: u64, limb_hi: u64) -> Self;
}

#[inline(always)]
fn bn254_sparse_montgomery_step(r: &mut [u64; 4], a: [u64; 4], limb: u64) {
    let mut carry1 = 0u64;
    r[0] = mac_carry(r[0], a[0], limb, &mut carry1);

    let k = r[0].wrapping_mul(BN254_INV);
    let mut carry2 = 0u64;
    let _ = mac_carry(r[0], k, BN254_MODULUS[0], &mut carry2);

    let new_r1 = mac_carry(r[1], a[1], limb, &mut carry1);
    r[0] = mac_carry(new_r1, k, BN254_MODULUS[1], &mut carry2);
    r[1] = new_r1;

    let new_r2 = mac_carry(r[2], a[2], limb, &mut carry1);
    r[1] = mac_carry(new_r2, k, BN254_MODULUS[2], &mut carry2);
    r[2] = new_r2;

    let new_r3 = mac_carry(r[3], a[3], limb, &mut carry1);
    r[2] = mac_carry(new_r3, k, BN254_MODULUS[3], &mut carry2);
    r[3] = carry1.wrapping_add(carry2);
}

impl Bn254UpperLimbMul for BN254Fr {
    #[inline(always)]
    fn mul_by_hi_2limbs(self, limb_lo: u64, limb_hi: u64) -> Self {
        let a = bn254_to_limbs(self);
        let mut r = [0u64; 4];
        bn254_sparse_montgomery_step(&mut r, a, limb_lo);
        bn254_sparse_montgomery_step(&mut r, a, limb_hi);

        if cmp4(r, BN254_MODULUS) != std::cmp::Ordering::Less {
            r = sub4(r, BN254_MODULUS);
        }
        bn254_from_limbs(r)
    }
}

pub fn sumcheck_deg2_delayed_bn254(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenges: &[BN254Fr],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Accum::zero();
        let mut h1 = BN254Accum::zero();
        let mut h_inf = BN254Accum::zero();

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

pub fn sumcheck_deg2_eq_delayed_bn254(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenges: &[BN254Fr],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Accum::zero();
        let mut q_inf = BN254Accum::zero();

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

pub fn sumcheck_deg2_projective_delayed_bn254(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenges: &[BN254Fr],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Accum::zero();
        let mut h1 = BN254Accum::zero();
        let mut h_inf = BN254Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            let sf = f0 + fi;
            let sg = g0 + gi;
            h0.fmadd(f0, g0);
            h1.fmadd(sf, sg);
            h_inf.fmadd(fi, gi);
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

pub fn sumcheck_deg2_eq_projective_delayed_bn254(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenges: &[BN254Fr],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Accum::zero();
        let mut q_inf = BN254Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            let sf = f0 + fi;
            let sg = g0 + gi;
            q1.fmadd(sf * sg, ew);
            q_inf.fmadd(fi * gi, ew);
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

pub fn sumcheck_deg2_boolean_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenge_limbs: &[(u64, u64)],
) {
    for round in 0..challenge_limbs.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Fr::ZERO;
        let mut h1 = BN254Fr::ZERO;
        let mut h_inf = BN254Fr::ZERO;

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            let df = f1 - f0;
            let dg = g1 - g0;

            h0 += f0 * g0;
            h1 += f1 * g1;
            h_inf += df * dg;
        }

        black_box((h0, h1, h_inf));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            f[j] = f0 + (f1 - f0).mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + (g1 - g0).mul_by_hi_2limbs(r_lo, r_hi);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenge_limbs: &[(u64, u64)],
) {
    for round in 0..challenge_limbs.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Fr::ZERO;
        let mut h1 = BN254Fr::ZERO;
        let mut h_inf = BN254Fr::ZERO;

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            h0 += f0 * g0;
            h_inf += fi * gi;
            h1 += (f0 + fi) * (g0 + gi);
        }

        black_box((h0, h1, h_inf));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = f0 + fi.mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + gi.mul_by_hi_2limbs(r_lo, r_hi);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_gruen_boolean_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenge_limbs: &[(u64, u64)],
) {
    let n = challenge_limbs.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Fr::ZERO;
        let mut q_inf = BN254Fr::ZERO;

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            let ew = eq_rest[j];
            let df = f1 - f0;
            let dg = g1 - g0;

            q1 += f1 * g1 * ew;
            q_inf += df * dg * ew;
        }

        black_box((q1, q_inf));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            f[j] = f0 + (f1 - f0).mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + (g1 - g0).mul_by_hi_2limbs(r_lo, r_hi);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_gruen_projective_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenge_limbs: &[(u64, u64)],
) {
    let n = challenge_limbs.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Fr::ZERO;
        let mut q_inf = BN254Fr::ZERO;

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q1 += (f0 + fi) * (g0 + gi) * ew;
            q_inf += fi * gi * ew;
        }

        black_box((q1, q_inf));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = f0 + fi.mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + gi.mul_by_hi_2limbs(r_lo, r_hi);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_delayed_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenge_limbs: &[(u64, u64)],
) {
    for round in 0..challenge_limbs.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Accum::zero();
        let mut h1 = BN254Accum::zero();
        let mut h_inf = BN254Accum::zero();

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

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            f[j] = f0 + (f1 - f0).mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + (g1 - g0).mul_by_hi_2limbs(r_lo, r_hi);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_delayed_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    challenge_limbs: &[(u64, u64)],
) {
    for round in 0..challenge_limbs.len() {
        let half = f.len() / 2;

        let mut h0 = BN254Accum::zero();
        let mut h1 = BN254Accum::zero();
        let mut h_inf = BN254Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            let sf = f0 + fi;
            let sg = g0 + gi;
            h0.fmadd(f0, g0);
            h1.fmadd(sf, sg);
            h_inf.fmadd(fi, gi);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = f0 + fi.mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + gi.mul_by_hi_2limbs(r_lo, r_hi);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_delayed_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenge_limbs: &[(u64, u64)],
) {
    let n = challenge_limbs.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Accum::zero();
        let mut q_inf = BN254Accum::zero();

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

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            f[j] = f0 + (f1 - f0).mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + (g1 - g0).mul_by_hi_2limbs(r_lo, r_hi);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_delayed_bn254_upper(
    f: &mut Vec<BN254Fr>,
    g: &mut Vec<BN254Fr>,
    suffix_eq: &[Vec<BN254Fr>],
    challenge_limbs: &[(u64, u64)],
) {
    let n = challenge_limbs.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = BN254Accum::zero();
        let mut q_inf = BN254Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            let sf = f0 + fi;
            let sg = g0 + gi;
            q1.fmadd(sf * sg, ew);
            q_inf.fmadd(fi * gi, ew);
        }

        black_box((q1.reduce(), q_inf.reduce()));

        let (r_lo, r_hi) = challenge_limbs[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = f0 + fi.mul_by_hi_2limbs(r_lo, r_hi);
            g[j] = g0 + gi.mul_by_hi_2limbs(r_lo, r_hi);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_limb_mul_matches_full_multiplication() {
        let xs = [
            BN254Fr::from(0u64),
            BN254Fr::from(1u64),
            BN254Fr::from(2u64),
            BN254Fr::from(u64::MAX),
            BN254Fr::from(0x1234_5678_9abc_def0u64),
        ];
        let limbs = [
            (0u64, 0u64),
            (1u64, 0u64),
            (0u64, 1u64),
            (0xabcdef01_23456789u64, 0x01234567_89abcdefu64),
            (u64::MAX, (1u64 << 61) - 1),
        ];

        for x in xs {
            for (lo, hi) in limbs {
                let challenge = BN254Fr::new_unchecked(ark_ff::BigInt([0, 0, lo, hi]));
                assert_eq!(x.mul_by_hi_2limbs(lo, hi), x * challenge);
            }
        }
    }
}
