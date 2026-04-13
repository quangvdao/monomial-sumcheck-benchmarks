#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vmull_p64};
use std::ops::{Add, AddAssign, Mul, Sub};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::{Fr as BN254Fr, FrConfig};
use ark_ff::{AdditiveGroup, MontConfig};
use binius_field::BinaryField128bGhash as GF128;
use binius_field::Field as BiniusField;
use hachi_pcs::algebra::Prime128Offset275;
use hachi_pcs::{AdditiveGroup as HachiAdditiveGroup, CanonicalField, FieldCore};
use p3_baby_bear::BabyBear;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_baby_bear::PackedBabyBearNeon;
use p3_field::extension::BinomialExtensionField;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_field::PackedValue;
use p3_field::PrimeCharacteristicRing;

type BB4 = BinomialExtensionField<BabyBear, 4>;
type BB5 = BinomialExtensionField<BabyBear, 5>;
type Fp128 = Prime128Offset275;

// ===========================================================================
// Element generation helpers
// ===========================================================================

fn make_u64s(n: usize) -> Vec<u64> {
    let mut vals = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        vals.push(state);
    }
    vals
}

fn make_bn254(n: usize) -> Vec<BN254Fr> {
    make_u64s(n).iter().map(|&v| BN254Fr::from(v)).collect()
}

fn make_bb4(n: usize) -> Vec<BB4> {
    let raw = make_u64s(n * 4);
    raw.chunks(4)
        .map(|chunk| {
            let base: [BabyBear; 4] =
                std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB4::new(base)
        })
        .collect()
}

fn make_bb5(n: usize) -> Vec<BB5> {
    let raw = make_u64s(n * 5);
    raw.chunks(5)
        .map(|chunk| {
            let base: [BabyBear; 5] =
                std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB5::new(base)
        })
        .collect()
}

fn make_fp128(n: usize) -> Vec<Fp128> {
    let raw = make_u64s(n * 2);
    raw.chunks(2)
        .map(|chunk| {
            let v = (chunk[0] as u128) | ((chunk[1] as u128) << 64);
            Fp128::from_canonical_u128_reduced(v)
        })
        .collect()
}

fn make_gf128(n: usize) -> Vec<GF128> {
    let raw = make_u64s(n * 2);
    raw.chunks(2)
        .map(|chunk| {
            let v = (chunk[0] as u128) | ((chunk[1] as u128) << 64);
            GF128::new(v)
        })
        .collect()
}

// ===========================================================================
// Suffix eq tables for Gruen split-eq
// ===========================================================================

/// Build suffix eq tables: tables[k] = eq(w[k..n], ·) of size 2^{n-k}.
/// At round k the prover needs eq_rest = tables[k+1] (size 2^{n-k-1}).
fn build_suffix_eq_tables<F>(w: &[F], one: F) -> Vec<Vec<F>>
where
    F: Copy + Add<Output = F> + Mul<Output = F> + Sub<Output = F>,
{
    let n = w.len();
    let mut tables: Vec<Vec<F>> = Vec::with_capacity(n + 1);
    tables.resize_with(n + 1, Vec::new);
    tables[n] = vec![one];
    for k in (0..n).rev() {
        let prev_len = tables[k + 1].len();
        let mut cur = Vec::with_capacity(prev_len * 2);
        for i in 0..prev_len {
            let s = tables[k + 1][i];
            cur.push(s * (one - w[k]));
            cur.push(s * w[k]);
        }
        tables[k] = cur;
    }
    tables
}

// ===========================================================================
// Generic degree-2 sumcheck: prove sum_x f(x)*g(x) = v
// Univariate per round is degree 2 => 3 evaluation points {0, 1, inf}
//
// Each round: (1) evaluate, (2) receive challenge, (3) bind.
// ===========================================================================

/// Boolean: evaluations on {0,1}^n.
/// Evaluate: 3 mul, 3 add, 2 sub per pair.
/// Bind:     2 mul, 2 add, 2 sub per pair.
fn sumcheck_deg2_boolean<F>(f: &mut Vec<F>, g: &mut Vec<F>, challenges: &[F], zero: F)
where
    F: Copy + Add<Output = F> + AddAssign + Sub<Output = F> + Mul<Output = F>,
{
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = zero;
        let mut h1 = zero;
        let mut h_inf = zero;

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

/// Projective: monomial coefficients on {0,inf}^n.
/// Evaluate: 3 mul, 5 add per pair.
/// Bind:     2 mul, 2 add per pair.
fn sumcheck_deg2_projective<F>(f: &mut Vec<F>, g: &mut Vec<F>, challenges: &[F], zero: F)
where
    F: Copy + Add<Output = F> + AddAssign + Mul<Output = F>,
{
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
            h1 += (f0 + fi) * (g0 + gi);
            h_inf += fi * gi;
        }

        black_box((h0, h1, h_inf));

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

// ===========================================================================
// Generic degree-2-times-eq sumcheck with Gruen split-eq optimization.
//
// Factor h(t) = eq_1(w_k, t) * current_scalar * q(t), where the quotient
// q(t) = sum_{x'} f(t,x') g(t,x') eq_rest(x') is degree 2.
// Evaluate q at {1, inf} per pair; derive q(0) from the claim (O(1)/round).
// No eq table clone or bind.
//
// Each round: (1) evaluate, (2) receive challenge, (3) bind f and g only.
// ===========================================================================

/// Boolean + Gruen.
/// Evaluate: 4 mul, 2 add, 2 sub per pair.
/// Bind:     2 mul, 2 add, 2 sub per pair.
fn sumcheck_deg2_eq_gruen_boolean<F>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    suffix_eq: &[Vec<F>],
    challenges: &[F],
    zero: F,
) where
    F: Copy + Add<Output = F> + AddAssign + Sub<Output = F> + Mul<Output = F>,
{
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = zero;
        let mut q_inf = zero;

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

/// Projective + Gruen.
/// Evaluate: 4 mul, 4 add per pair.
/// Bind:     2 mul, 2 add per pair.
fn sumcheck_deg2_eq_gruen_projective<F>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    suffix_eq: &[Vec<F>],
    challenges: &[F],
    zero: F,
) where
    F: Copy + Add<Output = F> + AddAssign + Mul<Output = F>,
{
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

            q1 += (f0 + fi) * (g0 + gi) * ew;
            q_inf += fi * gi * ew;
        }

        black_box((q1, q_inf));

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

fn sumcheck_deg2_projective_fp128(
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

            let sf = f0 + fi;
            let sg = g0 + gi;

            h0 += f0 * g0;
            h1 += sf * sg;
            h_inf += fi * gi;
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

fn sumcheck_deg2_eq_gruen_projective_fp128(
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

            let sf = f0 + fi;
            let sg = g0 + gi;

            q1 += sf * sg * ew;
            q_inf += fi * gi * ew;
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

// ===========================================================================
// Delayed reduction accumulators
// (speeding-up-sumcheck baseline: defer modular reduction in evaluation loop)
// ===========================================================================

// --- BN254 Montgomery reduction primitives ---

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

/// Wide accumulator for BN254 Fr: 8 folded u128 slots for deferred Montgomery reduction.
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

/// Wide accumulator for Fp128: 4 folded u128 slots for deferred Solinas reduction.
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

// --- Degree-2 sumcheck with delayed reduction (Boolean basis) ---

fn sumcheck_deg2_delayed_bn254(
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

fn sumcheck_deg2_delayed_fp128(
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

// --- Degree-2 x eq (Gruen split-eq) with delayed reduction ---

fn sumcheck_deg2_eq_delayed_bn254(
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

fn sumcheck_deg2_eq_delayed_fp128(
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

// --- Degree-2 sumcheck with delayed reduction (Projective basis) ---

fn sumcheck_deg2_projective_delayed_bn254(
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

fn sumcheck_deg2_projective_delayed_fp128(
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
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

// --- Degree-2 x eq (Gruen split-eq) with delayed reduction (Projective basis) ---

fn sumcheck_deg2_eq_projective_delayed_bn254(
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

fn sumcheck_deg2_eq_projective_delayed_fp128(
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
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

// ===========================================================================
// BabyBear extension delayed reduction accumulators
// Defers both base-field (Montgomery) reduction and polynomial reduction
// by accumulating schoolbook product coefficients in u128.
// ===========================================================================

const BB_PRIME: u64 = 0x78000001;
const BB_MONTY_MU: u64 = 0x88000001;
const BB_MONTY_INV: u64 = BB_PRIME - ((BB_PRIME.wrapping_mul(BB_MONTY_MU) - 1) >> 32);
const BB_W4: u128 = 11; // X^4 - 11 is irreducible over BabyBear
const BB_W5: u128 = 2;  // X^5 - 2 is irreducible over BabyBear

#[inline(always)]
fn bb_monty_reduce_u128(x: u128) -> u32 {
    ((x.wrapping_mul(BB_MONTY_INV as u128)) % (BB_PRIME as u128)) as u32
}

#[inline(always)]
fn bb4_to_limbs(x: BB4) -> [u32; 4] {
    const _: () = assert!(std::mem::size_of::<BB4>() == std::mem::size_of::<[u32; 4]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb4_to_coeffs(x: BB4) -> [BabyBear; 4] {
    const _: () = assert!(std::mem::size_of::<BB4>() == std::mem::size_of::<[BabyBear; 4]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb4_from_limbs(limbs: [u32; 4]) -> BB4 {
    unsafe { std::mem::transmute(limbs) }
}

#[inline(always)]
fn bb5_to_limbs(x: BB5) -> [u32; 5] {
    const _: () = assert!(std::mem::size_of::<BB5>() == std::mem::size_of::<[u32; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb5_to_coeffs(x: BB5) -> [BabyBear; 5] {
    const _: () = assert!(std::mem::size_of::<BB5>() == std::mem::size_of::<[BabyBear; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb5_from_limbs(limbs: [u32; 5]) -> BB5 {
    unsafe { std::mem::transmute(limbs) }
}

// --- BB4 accumulator: 7 u128 slots for unreduced degree-6 polynomial ---

struct BB4Accum {
    c: [u128; 7],
}

impl BB4Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self { c: [0u128; 7] }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: BB4, b: BB4) {
        let a = bb4_to_limbs(a);
        let b = bb4_to_limbs(b);
        let (a0, a1, a2, a3) = (a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64);
        let (b0, b1, b2, b3) = (b[0] as u64, b[1] as u64, b[2] as u64, b[3] as u64);

        self.c[0] += (a0 * b0) as u128;
        self.c[1] += (a0 * b1 + a1 * b0) as u128;
        self.c[2] += (a0 * b2 + a1 * b1 + a2 * b0) as u128;
        self.c[3] += (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) as u128;
        self.c[4] += (a1 * b3 + a2 * b2 + a3 * b1) as u128;
        self.c[5] += (a2 * b3 + a3 * b2) as u128;
        self.c[6] += (a3 * b3) as u128;
    }

    fn reduce(self) -> BB4 {
        let d0 = self.c[0] + BB_W4 * self.c[4];
        let d1 = self.c[1] + BB_W4 * self.c[5];
        let d2 = self.c[2] + BB_W4 * self.c[6];
        let d3 = self.c[3];
        bb4_from_limbs([
            bb_monty_reduce_u128(d0),
            bb_monty_reduce_u128(d1),
            bb_monty_reduce_u128(d2),
            bb_monty_reduce_u128(d3),
        ])
    }
}

// --- BB5 accumulator: 9 u128 slots for unreduced degree-8 polynomial ---

struct BB5Accum {
    c: [u128; 9],
}

impl BB5Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self { c: [0u128; 9] }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: BB5, b: BB5) {
        let a = bb5_to_limbs(a);
        let b = bb5_to_limbs(b);
        let (a0, a1, a2, a3, a4) = (a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64, a[4] as u64);
        let (b0, b1, b2, b3, b4) = (b[0] as u64, b[1] as u64, b[2] as u64, b[3] as u64, b[4] as u64);

        self.c[0] += (a0 * b0) as u128;
        self.c[1] += (a0 * b1 + a1 * b0) as u128;
        self.c[2] += (a0 * b2 + a1 * b1 + a2 * b0) as u128;
        self.c[3] += (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) as u128;
        // c[4] has 5 terms; only 4 fit in u64 without overflow
        self.c[4] += (a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1) as u128;
        self.c[4] += (a4 * b0) as u128;
        self.c[5] += (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1) as u128;
        self.c[6] += (a2 * b4 + a3 * b3 + a4 * b2) as u128;
        self.c[7] += (a3 * b4 + a4 * b3) as u128;
        self.c[8] += (a4 * b4) as u128;
    }

    fn reduce(self) -> BB5 {
        let d0 = self.c[0] + BB_W5 * self.c[5];
        let d1 = self.c[1] + BB_W5 * self.c[6];
        let d2 = self.c[2] + BB_W5 * self.c[7];
        let d3 = self.c[3] + BB_W5 * self.c[8];
        let d4 = self.c[4];
        bb5_from_limbs([
            bb_monty_reduce_u128(d0),
            bb_monty_reduce_u128(d1),
            bb_monty_reduce_u128(d2),
            bb_monty_reduce_u128(d3),
            bb_monty_reduce_u128(d4),
        ])
    }
}

struct BB4MulByConst {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    r: BB4,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    cols: [PackedBabyBearNeon; 4],
}

impl BB4MulByConst {
    #[inline(always)]
    fn new(r: BB4) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let [r0, r1, r2, r3] = bb4_to_coeffs(r);
            let w = BabyBear::from_u32(BB_W4 as u32);
            let wr1 = w * r1;
            let wr2 = w * r2;
            let wr3 = w * r3;
            let cols = [
                PackedBabyBearNeon::from_fn(|i| [r0, r1, r2, r3][i]),
                PackedBabyBearNeon::from_fn(|i| [wr3, r0, r1, r2][i]),
                PackedBabyBearNeon::from_fn(|i| [wr2, wr3, r0, r1][i]),
                PackedBabyBearNeon::from_fn(|i| [wr1, wr2, wr3, r0][i]),
            ];
            Self { cols }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self { r }
        }
    }

    #[inline(always)]
    fn apply(&self, x: BB4) -> BB4 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let [x0, x1, x2, x3] = bb4_to_coeffs(x);
            let lhs = [x0.into(), x1.into(), x2.into(), x3.into()];
            let out = PackedBabyBearNeon::dot_product(&lhs, &self.cols).0;
            BB4::new(out)
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.r * x
        }
    }
}

struct BB5MulByConst {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    r: BB5,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    rhs: [PackedBabyBearNeon; 5],
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    tail: [BabyBear; 5],
}

impl BB5MulByConst {
    #[inline(always)]
    fn new(r: BB5) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let [r0, r1, r2, r3, r4] = bb5_to_coeffs(r);
            let w = BabyBear::from_u32(BB_W5 as u32);
            let wr1 = w * r1;
            let wr2 = w * r2;
            let wr3 = w * r3;
            let wr4 = w * r4;
            let rhs = [
                PackedBabyBearNeon::from_fn(|i| [r0, r1, r2, r3][i]),
                PackedBabyBearNeon::from_fn(|i| [wr4, r0, r1, r2][i]),
                PackedBabyBearNeon::from_fn(|i| [wr3, wr4, r0, r1][i]),
                PackedBabyBearNeon::from_fn(|i| [wr2, wr3, wr4, r0][i]),
                PackedBabyBearNeon::from_fn(|i| [wr1, wr2, wr3, wr4][i]),
            ];
            let tail = [r4, r3, r2, r1, r0];
            Self { rhs, tail }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self { r }
        }
    }

    #[inline(always)]
    fn apply(&self, x: BB5) -> BB5 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let [x0, x1, x2, x3, x4] = bb5_to_coeffs(x);
            let lhs = [x0.into(), x1.into(), x2.into(), x3.into(), x4.into()];
            let dot = PackedBabyBearNeon::dot_product(&lhs, &self.rhs).0;
            let tail = BabyBear::dot_product::<5>(&[x0, x1, x2, x3, x4], &self.tail);
            BB5::new([dot[0], dot[1], dot[2], dot[3], tail])
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.r * x
        }
    }
}


// --- Degree-2 sumcheck with delayed reduction for BB4/BB5 (Boolean basis) ---

macro_rules! delayed_sumcheck_fns {
    ($field:ty, $accum:ty, $mul_by_const:ty, $suffix:ident) => {
        paste::paste! {
            fn [<sumcheck_deg2_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                challenges: &[$field],
            ) {
                for round in 0..challenges.len() {
                    let half = f.len() / 2;
                    let mut h0 = <$accum>::zero();
                    let mut h1 = <$accum>::zero();
                    let mut h_inf = <$accum>::zero();
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
                    let _ = black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let f1 = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let g1 = g[2 * j + 1];
                        f[j] = f0 + mul_r.apply(f1 - f0);
                        g[j] = g0 + mul_r.apply(g1 - g0);
                    }
                    f.truncate(half);
                    g.truncate(half);
                }
            }

            fn [<sumcheck_deg2_eq_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                suffix_eq: &[Vec<$field>],
                challenges: &[$field],
            ) {
                let n = challenges.len();
                for round in 0..n {
                    let half = f.len() / 2;
                    let eq_rest = &suffix_eq[round + 1];
                    let mut q1 = <$accum>::zero();
                    let mut q_inf = <$accum>::zero();
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
                    let _ = black_box((q1.reduce(), q_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let f1 = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let g1 = g[2 * j + 1];
                        f[j] = f0 + mul_r.apply(f1 - f0);
                        g[j] = g0 + mul_r.apply(g1 - g0);
                    }
                    f.truncate(half);
                    g.truncate(half);
                }
            }

            fn [<sumcheck_deg2_projective_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                challenges: &[$field],
            ) {
                for round in 0..challenges.len() {
                    let half = f.len() / 2;
                    let mut h0 = <$accum>::zero();
                    let mut h1 = <$accum>::zero();
                    let mut h_inf = <$accum>::zero();
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
                    let _ = black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let fi = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let gi = g[2 * j + 1];
                        f[j] = f0 + mul_r.apply(fi);
                        g[j] = g0 + mul_r.apply(gi);
                    }
                    f.truncate(half);
                    g.truncate(half);
                }
            }

            fn [<sumcheck_deg2_eq_projective_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                suffix_eq: &[Vec<$field>],
                challenges: &[$field],
            ) {
                let n = challenges.len();
                for round in 0..n {
                    let half = f.len() / 2;
                    let eq_rest = &suffix_eq[round + 1];
                    let mut q1 = <$accum>::zero();
                    let mut q_inf = <$accum>::zero();
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
                    let _ = black_box((q1.reduce(), q_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let fi = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let gi = g[2 * j + 1];
                        f[j] = f0 + mul_r.apply(fi);
                        g[j] = g0 + mul_r.apply(gi);
                    }
                    f.truncate(half);
                    g.truncate(half);
                }
            }
        }
    };
}

delayed_sumcheck_fns!(BB4, BB4Accum, BB4MulByConst, bb4);
delayed_sumcheck_fns!(BB5, BB5Accum, BB5MulByConst, bb5);

// ===========================================================================
// GF(2^128) delayed reduction accumulator
// Defers the gf2_128_reduce step by accumulating unreduced 256-bit products
// via XOR (which has no carry in binary fields).
// ===========================================================================

// --- aarch64 NEON path: accumulate in NEON registers, reduce via PMULL ---

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
struct GF128Accum {
    low: uint64x2_t,
    mid: uint64x2_t,
    high: uint64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl GF128Accum {
    #[inline(always)]
    fn zero() -> Self {
        unsafe {
            Self {
                low: vdupq_n_u64(0),
                mid: vdupq_n_u64(0),
                high: vdupq_n_u64(0),
            }
        }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: GF128, b: GF128) {
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

    fn reduce(self) -> GF128 {
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

// --- Portable fallback: multiply-then-add, no deferred reduction ---

#[cfg(not(target_arch = "aarch64"))]
struct GF128Accum {
    sum: GF128,
}

#[cfg(not(target_arch = "aarch64"))]
impl GF128Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self { sum: GF128::ZERO }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: GF128, b: GF128) {
        self.sum += a * b;
    }

    fn reduce(self) -> GF128 {
        self.sum
    }
}

// --- Degree-2 sumcheck with delayed reduction for GF128 (Boolean basis) ---

fn sumcheck_deg2_delayed_gf128(
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

fn sumcheck_deg2_eq_delayed_gf128(
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

// --- Degree-2 sumcheck with delayed reduction for GF128 (Projective basis) ---

fn sumcheck_deg2_projective_delayed_gf128(
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

fn sumcheck_deg2_eq_projective_delayed_gf128(
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

// ===========================================================================
// Benchmarks
// ===========================================================================

macro_rules! bench_field {
    ($field_label:expr, $make_elems:expr, $make_chals:expr,
     $zero:expr, $one:expr, $ns:expr, $c:expr) => {{
        // Degree-2
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make_elems(1usize << n_usize);
                let g_orig = $make_elems(1usize << n_usize);
                let challenges = $make_chals(n_usize);

                group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_boolean(&mut f, &mut g, &challenges, $zero);
                    })
                });

                group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_projective(&mut f, &mut g, &challenges, $zero);
                    })
                });
            }
            group.finish();
        }
        // Degree-2 x eq (Gruen split-eq)
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2_eq/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make_elems(1usize << n_usize);
                let g_orig = $make_elems(1usize << n_usize);
                let challenges = $make_chals(n_usize);
                let eq_point = $make_chals(n_usize);
                let suffix_eq = build_suffix_eq_tables(&eq_point, $one);

                group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_eq_gruen_boolean(
                            &mut f, &mut g, &suffix_eq, &challenges, $zero,
                        );
                    })
                });

                group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_eq_gruen_projective(
                            &mut f, &mut g, &suffix_eq, &challenges, $zero,
                        );
                    })
                });
            }
            group.finish();
        }
    }};
}

fn bench_bn254(c: &mut Criterion) {
    let ns = [16u32, 20, 24];

    {
        let mut group = c.benchmark_group("sumcheck_deg2/BN254");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let challenges = make_bn254(n_usize);

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_boolean(&mut f, &mut g, &challenges, BN254Fr::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_delayed_bn254(&mut f, &mut g, &challenges);
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective(&mut f, &mut g, &challenges, BN254Fr::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective_delayed_bn254(&mut f, &mut g, &challenges);
                })
            });
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/BN254");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let challenges = make_bn254(n_usize);
            let eq_point = make_bn254(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, BN254Fr::from(1u64));

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_boolean(
                        &mut f, &mut g, &suffix_eq, &challenges, BN254Fr::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_delayed_bn254(
                        &mut f, &mut g, &suffix_eq, &challenges,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_projective(
                        &mut f, &mut g, &suffix_eq, &challenges, BN254Fr::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_projective_delayed_bn254(
                        &mut f, &mut g, &suffix_eq, &challenges,
                    );
                })
            });
        }
        group.finish();
    }
}

macro_rules! bench_bb_field {
    ($field_label:expr, $field:ty, $make:ident, $zero:expr, $one:expr,
     $delayed_fn:ident, $delayed_eq_fn:ident,
     $proj_delayed_fn:ident, $proj_delayed_eq_fn:ident,
     $ns:expr, $c:expr) => {{
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make(1usize << n_usize);
                let g_orig = $make(1usize << n_usize);
                let challenges = $make(n_usize);

                group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_boolean(&mut f, &mut g, &challenges, $zero);
                    })
                });

                group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        $delayed_fn(&mut f, &mut g, &challenges);
                    })
                });

                group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_projective(&mut f, &mut g, &challenges, $zero);
                    })
                });

                group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        $proj_delayed_fn(&mut f, &mut g, &challenges);
                    })
                });
            }
            group.finish();
        }
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2_eq/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make(1usize << n_usize);
                let g_orig = $make(1usize << n_usize);
                let challenges = $make(n_usize);
                let eq_point = $make(n_usize);
                let suffix_eq = build_suffix_eq_tables(&eq_point, $one);

                group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_eq_gruen_boolean(
                            &mut f, &mut g, &suffix_eq, &challenges, $zero,
                        );
                    })
                });

                group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        $delayed_eq_fn(&mut f, &mut g, &suffix_eq, &challenges);
                    })
                });

                group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        sumcheck_deg2_eq_gruen_projective(
                            &mut f, &mut g, &suffix_eq, &challenges, $zero,
                        );
                    })
                });

                group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                    b.iter(|| {
                        let mut f = f_orig.clone();
                        let mut g = g_orig.clone();
                        $proj_delayed_eq_fn(&mut f, &mut g, &suffix_eq, &challenges);
                    })
                });
            }
            group.finish();
        }
    }};
}

fn bench_bb4(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_bb_field!(
        "BB4", BB4, make_bb4, BB4::ZERO, BB4::ONE,
        sumcheck_deg2_delayed_bb4, sumcheck_deg2_eq_delayed_bb4,
        sumcheck_deg2_projective_delayed_bb4, sumcheck_deg2_eq_projective_delayed_bb4,
        ns, c
    );
}

fn bench_bb5(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_bb_field!(
        "BB5", BB5, make_bb5, BB5::ZERO, BB5::ONE,
        sumcheck_deg2_delayed_bb5, sumcheck_deg2_eq_delayed_bb5,
        sumcheck_deg2_projective_delayed_bb5, sumcheck_deg2_eq_projective_delayed_bb5,
        ns, c
    );
}

fn bench_fp128(c: &mut Criterion) {
    let ns = [16u32, 20];
    {
        let mut group = c.benchmark_group("sumcheck_deg2/Fp128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_fp128(1usize << n_usize);
            let g_orig = make_fp128(1usize << n_usize);
            let challenges = make_fp128(n_usize);

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_boolean(&mut f, &mut g, &challenges, Fp128::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_delayed_fp128(&mut f, &mut g, &challenges);
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective_fp128(&mut f, &mut g, &challenges, Fp128::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective_delayed_fp128(&mut f, &mut g, &challenges);
                })
            });
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/Fp128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_fp128(1usize << n_usize);
            let g_orig = make_fp128(1usize << n_usize);
            let challenges = make_fp128(n_usize);
            let eq_point = make_fp128(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, Fp128::one());

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_boolean(
                        &mut f,
                        &mut g,
                        &suffix_eq,
                        &challenges,
                        Fp128::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_delayed_fp128(
                        &mut f,
                        &mut g,
                        &suffix_eq,
                        &challenges,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_projective_fp128(
                        &mut f,
                        &mut g,
                        &suffix_eq,
                        &challenges,
                        Fp128::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_projective_delayed_fp128(
                        &mut f,
                        &mut g,
                        &suffix_eq,
                        &challenges,
                    );
                })
            });
        }
        group.finish();
    }
}

fn bench_gf128(c: &mut Criterion) {
    let ns = [16u32, 20];
    {
        let mut group = c.benchmark_group("sumcheck_deg2/GF128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_gf128(1usize << n_usize);
            let g_orig = make_gf128(1usize << n_usize);
            let challenges = make_gf128(n_usize);

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_boolean(&mut f, &mut g, &challenges, GF128::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_delayed_gf128(&mut f, &mut g, &challenges);
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective(&mut f, &mut g, &challenges, GF128::ZERO);
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_projective_delayed_gf128(&mut f, &mut g, &challenges);
                })
            });
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/GF128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_gf128(1usize << n_usize);
            let g_orig = make_gf128(1usize << n_usize);
            let challenges = make_gf128(n_usize);
            let eq_point = make_gf128(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, GF128::ONE);

            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_boolean(
                        &mut f, &mut g, &suffix_eq, &challenges, GF128::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_delayed_gf128(
                        &mut f, &mut g, &suffix_eq, &challenges,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_gruen_projective(
                        &mut f, &mut g, &suffix_eq, &challenges, GF128::ZERO,
                    );
                })
            });

            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                b.iter(|| {
                    let mut f = f_orig.clone();
                    let mut g = g_orig.clone();
                    sumcheck_deg2_eq_projective_delayed_gf128(
                        &mut f, &mut g, &suffix_eq, &challenges,
                    );
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_bn254, bench_bb4, bench_bb5, bench_fp128, bench_gf128);
criterion_main!(benches);
