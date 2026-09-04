use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::AdditiveGroup;
use monomial_sumcheck_benchmarks::{benchmark_config, lookup_tables::*};
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
            let base: [BabyBear; 4] = std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB4::new(base)
        })
        .collect()
}

// ===========================================================================
// 3a. Full table build via iterative doubling
// ===========================================================================

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
            b.iter(|| black_box(lt_evals_projective(black_box(&r_bb), BB4::ZERO, BB4::ONE)))
        });
    }

    group.finish();
}

// ===========================================================================
// 3b. Single-point evaluate_mle
// ===========================================================================

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

criterion_group! {
    name = benches;
    config = benchmark_config();
    targets = bench_eq_table_build, bench_lt_table_build, bench_eq_mle, bench_lt_mle
}
criterion_main!(benches);
