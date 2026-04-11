use std::iter::zip;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::AdditiveGroup;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeCharacteristicRing;

type BB4 = BinomialExtensionField<BabyBear, 4>;

fn make_u64s(n: usize) -> Vec<u64> {
    let mut vals = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        vals.push(state);
    }
    vals
}

fn make_bn254_challenges(n: usize) -> Vec<BN254Fr> {
    make_u64s(n).iter().map(|&v| BN254Fr::from(v)).collect()
}

fn make_bb4_challenges(n: usize) -> Vec<BB4> {
    let raw = make_u64s(n * 4);
    raw.chunks(4)
        .map(|chunk| {
            let base: [BabyBear; 4] =
                std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB4::new(base)
        })
        .collect()
}

// ===========================================================================
// 3a. Full table build via iterative doubling
// ===========================================================================

// ---------------------------------------------------------------------------
// EQ table: Boolean  y = x * r; x -= y  (1 mul + 1 sub per entry per round)
// ---------------------------------------------------------------------------
fn eq_evals_boolean<F: Copy + std::ops::Mul<Output = F> + std::ops::Sub<Output = F>>(
    r: &[F],
    one: F,
) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![one; 1 << n];
    let mut size = 1;
    for j in 0..n {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = evals[i / 2];
            evals[i] = scalar * r[j];
            evals[i - 1] = scalar - evals[i];
        }
    }
    evals
}

// ---------------------------------------------------------------------------
// EQ table: Projective  y = x * r; left half unchanged  (1 mul per entry per round)
// ---------------------------------------------------------------------------
fn eq_evals_projective<F: Copy + std::ops::Mul<Output = F>>(r: &[F], one: F) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![one; 1 << n];
    let mut size = 1;
    for j in 0..n {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = evals[i / 2];
            evals[i] = scalar * r[j];
            evals[i - 1] = scalar;
        }
    }
    evals
}

// ---------------------------------------------------------------------------
// LT table: Boolean  y = x * r; x += r - y  (1 mul + 1 add + 1 sub per entry per round)
// ---------------------------------------------------------------------------
fn lt_evals_boolean<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F> + std::ops::Sub<Output = F>>(
    r: &[F],
    zero: F,
) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![zero; 1 << n];
    for (i, r_i) in r.iter().rev().enumerate() {
        let (left, right) = evals.split_at_mut(1 << i);
        zip(left, right).for_each(|(x, y)| {
            *y = *x * *r_i;
            *x = *x + *r_i - *y;
        });
    }
    evals
}

// ---------------------------------------------------------------------------
// LT table: Projective  y = x * r; x += r * P  (1 mul + 1 add per entry per round)
// ---------------------------------------------------------------------------
fn lt_evals_projective<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F>>(
    r: &[F],
    zero: F,
    one: F,
) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![zero; 1 << n];
    let mut suffix_omega = one;
    for (i, r_i) in r.iter().rev().enumerate() {
        let r_times_omega = *r_i * suffix_omega;
        let (left, right) = evals.split_at_mut(1 << i);
        zip(left, right).for_each(|(x, y)| {
            *y = *x * *r_i;
            *x = *x + r_times_omega;
        });
        suffix_omega = suffix_omega * (one + *r_i);
    }
    evals
}

fn bench_eq_table_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("eq_table_build");

    for n in [16, 20, 24] {
        let r_bn = make_bn254_challenges(n);

        group.bench_with_input(BenchmarkId::new("BN254_boolean", n), &n, |b, _| {
            b.iter(|| black_box(eq_evals_boolean(black_box(&r_bn), BN254Fr::from(1u64))))
        });

        group.bench_with_input(BenchmarkId::new("BN254_projective", n), &n, |b, _| {
            b.iter(|| black_box(eq_evals_projective(black_box(&r_bn), BN254Fr::from(1u64))))
        });
    }

    for n in [16, 20, 24] {
        let r_bb = make_bb4_challenges(n);

        group.bench_with_input(BenchmarkId::new("BB4_boolean", n), &n, |b, _| {
            b.iter(|| black_box(eq_evals_boolean(black_box(&r_bb), BB4::ONE)))
        });

        group.bench_with_input(BenchmarkId::new("BB4_projective", n), &n, |b, _| {
            b.iter(|| black_box(eq_evals_projective(black_box(&r_bb), BB4::ONE)))
        });
    }

    group.finish();
}

fn bench_lt_table_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("lt_table_build");

    for n in [16, 20, 24] {
        let r_bn = make_bn254_challenges(n);

        group.bench_with_input(BenchmarkId::new("BN254_boolean", n), &n, |b, _| {
            b.iter(|| black_box(lt_evals_boolean(black_box(&r_bn), BN254Fr::ZERO)))
        });

        group.bench_with_input(BenchmarkId::new("BN254_projective", n), &n, |b, _| {
            b.iter(|| {
                black_box(lt_evals_projective(
                    black_box(&r_bn),
                    BN254Fr::ZERO,
                    BN254Fr::from(1u64),
                ))
            })
        });
    }

    for n in [16, 20, 24] {
        let r_bb = make_bb4_challenges(n);

        group.bench_with_input(BenchmarkId::new("BB4_boolean", n), &n, |b, _| {
            b.iter(|| black_box(lt_evals_boolean(black_box(&r_bb), BB4::ZERO)))
        });

        group.bench_with_input(BenchmarkId::new("BB4_projective", n), &n, |b, _| {
            b.iter(|| {
                black_box(lt_evals_projective(
                    black_box(&r_bb),
                    BB4::ZERO,
                    BB4::ONE,
                ))
            })
        });
    }

    group.finish();
}

// ===========================================================================
// 3b. Single-point evaluate_mle
// ===========================================================================

// ---------------------------------------------------------------------------
// EQ evaluate_mle: Boolean  prod_i(x_i*y_i + (1-x_i)*(1-y_i))
// ---------------------------------------------------------------------------
fn eq_mle_boolean<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F> + std::ops::Sub<Output = F>>(
    x: &[F],
    y: &[F],
    one: F,
) -> F {
    assert_eq!(x.len(), y.len());
    let mut result = one;
    for i in 0..x.len() {
        result = result * (x[i] * y[i] + (one - x[i]) * (one - y[i]));
    }
    result
}

// ---------------------------------------------------------------------------
// EQ evaluate_mle: Projective  prod_i(1 + x_i*y_i)
// ---------------------------------------------------------------------------
fn eq_mle_projective<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F>>(
    x: &[F],
    y: &[F],
    one: F,
) -> F {
    assert_eq!(x.len(), y.len());
    let mut result = one;
    for i in 0..x.len() {
        result = result * (one + x[i] * y[i]);
    }
    result
}

// ---------------------------------------------------------------------------
// LT evaluate_mle: Boolean
//   result += (1-x_i)*y_i * eq_term
//   eq_term *= x_i*y_i + (1-x_i)*(1-y_i)
// ---------------------------------------------------------------------------
fn lt_mle_boolean<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F> + std::ops::Sub<Output = F>>(
    x: &[F],
    y: &[F],
    zero: F,
    one: F,
) -> F {
    assert_eq!(x.len(), y.len());
    let mut result = zero;
    let mut eq_term = one;
    for i in 0..x.len() {
        result = result + (one - x[i]) * y[i] * eq_term;
        eq_term = eq_term * (x[i] * y[i] + (one - x[i]) * (one - y[i]));
    }
    result
}

// ---------------------------------------------------------------------------
// LT evaluate_mle: Projective
//   result += y_i * eq_prefix * omega_suffix
//   eq_prefix *= (1 + x_i*y_i)
//   omega_suffix is precomputed as suffix product of (1+x_k)*(1+y_k)
// ---------------------------------------------------------------------------
fn lt_mle_projective<F: Copy + std::ops::Mul<Output = F> + std::ops::Add<Output = F>>(
    x: &[F],
    y: &[F],
    zero: F,
    one: F,
) -> F {
    assert_eq!(x.len(), y.len());
    let w = x.len();

    let mut omega_suffix = vec![one; w + 1];
    for i in (0..w).rev() {
        omega_suffix[i] = omega_suffix[i + 1] * (one + x[i]) * (one + y[i]);
    }

    let mut result = zero;
    let mut eq_prefix = one;
    for i in 0..w {
        result = result + y[i] * eq_prefix * omega_suffix[i + 1];
        eq_prefix = eq_prefix * (one + x[i] * y[i]);
    }
    result
}

fn bench_eq_mle(c: &mut Criterion) {
    let mut group = c.benchmark_group("eq_mle");

    for w in [8, 32] {
        let x_bn = make_bn254_challenges(w);
        let y_bn = make_bn254_challenges(w);

        group.bench_with_input(BenchmarkId::new("BN254_boolean", w), &w, |b, _| {
            b.iter(|| {
                black_box(eq_mle_boolean(
                    black_box(&x_bn),
                    black_box(&y_bn),
                    BN254Fr::from(1u64),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("BN254_projective", w), &w, |b, _| {
            b.iter(|| {
                black_box(eq_mle_projective(
                    black_box(&x_bn),
                    black_box(&y_bn),
                    BN254Fr::from(1u64),
                ))
            })
        });
    }

    for w in [8, 32] {
        let x_bb = make_bb4_challenges(w);
        let y_bb = make_bb4_challenges(w);

        group.bench_with_input(BenchmarkId::new("BB4_boolean", w), &w, |b, _| {
            b.iter(|| black_box(eq_mle_boolean(black_box(&x_bb), black_box(&y_bb), BB4::ONE)))
        });

        group.bench_with_input(BenchmarkId::new("BB4_projective", w), &w, |b, _| {
            b.iter(|| {
                black_box(eq_mle_projective(
                    black_box(&x_bb),
                    black_box(&y_bb),
                    BB4::ONE,
                ))
            })
        });
    }

    group.finish();
}

fn bench_lt_mle(c: &mut Criterion) {
    let mut group = c.benchmark_group("lt_mle");

    for w in [8, 32] {
        let x_bn = make_bn254_challenges(w);
        let y_bn = make_bn254_challenges(w);

        group.bench_with_input(BenchmarkId::new("BN254_boolean", w), &w, |b, _| {
            b.iter(|| {
                black_box(lt_mle_boolean(
                    black_box(&x_bn),
                    black_box(&y_bn),
                    BN254Fr::ZERO,
                    BN254Fr::from(1u64),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("BN254_projective", w), &w, |b, _| {
            b.iter(|| {
                black_box(lt_mle_projective(
                    black_box(&x_bn),
                    black_box(&y_bn),
                    BN254Fr::ZERO,
                    BN254Fr::from(1u64),
                ))
            })
        });
    }

    for w in [8, 32] {
        let x_bb = make_bb4_challenges(w);
        let y_bb = make_bb4_challenges(w);

        group.bench_with_input(BenchmarkId::new("BB4_boolean", w), &w, |b, _| {
            b.iter(|| {
                black_box(lt_mle_boolean(
                    black_box(&x_bb),
                    black_box(&y_bb),
                    BB4::ZERO,
                    BB4::ONE,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("BB4_projective", w), &w, |b, _| {
            b.iter(|| {
                black_box(lt_mle_projective(
                    black_box(&x_bb),
                    black_box(&y_bb),
                    BB4::ZERO,
                    BB4::ONE,
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_eq_table_build,
    bench_lt_table_build,
    bench_eq_mle,
    bench_lt_mle
);
criterion_main!(benches);
