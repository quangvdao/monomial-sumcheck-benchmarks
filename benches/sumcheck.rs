use std::ops::{Add, AddAssign, Mul, Sub};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::AdditiveGroup;
use binius_field::BinaryField128bGhash as GF128;
use binius_field::Field as BiniusField;
use hachi_pcs::algebra::Prime128Offset275;
use hachi_pcs::{AdditiveGroup as HachiAdditiveGroup, CanonicalField, FieldCore};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
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
    bench_field!(
        "BN254",
        make_bn254, make_bn254, BN254Fr::ZERO, BN254Fr::from(1u64),
        ns, c
    );
}

fn bench_bb4(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_field!(
        "BB4",
        make_bb4, make_bb4, BB4::ZERO, BB4::ONE,
        ns, c
    );
}

fn bench_bb5(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_field!(
        "BB5",
        make_bb5, make_bb5, BB5::ZERO, BB5::ONE,
        ns, c
    );
}

fn bench_fp128(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_field!(
        "Fp128",
        make_fp128, make_fp128, Fp128::ZERO, Fp128::one(),
        ns, c
    );
}

fn bench_gf128(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_field!(
        "GF128",
        make_gf128, make_gf128, GF128::ZERO, GF128::ONE,
        ns, c
    );
}

criterion_group!(benches, bench_bn254, bench_bb4, bench_bb5, bench_fp128, bench_gf128);
criterion_main!(benches);
