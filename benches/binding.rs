use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::seq::SliceRandom;

use ark_bn254::Fr as BN254Fr;
use monomial_sumcheck_benchmarks::{
    benchmark_config,
    sumcheck::{Bn254BranchFree, Bn254UpperLimbMul},
};

const N: usize = 1 << 20;

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

// ---------------------------------------------------------------------------
// Upper-limb challenge multiplication (Section 6.3)
// Challenge is a 125-bit value stored in the upper two limbs of Montgomery form:
// BigInt = [0, 0, lo, hi]
// ---------------------------------------------------------------------------

fn make_upper_limb_challenge() -> (BN254Fr, u64, u64) {
    let lo: u64 = 0xabcdef0123456789;
    let hi: u64 = 0x1234567890abcdef >> 3; // 125 bits total (top 3 bits zeroed)
    let challenge = BN254Fr::new_unchecked(ark_ff::BigInt([0, 0, lo, hi]));
    (challenge, lo, hi)
}

fn bench_upper_limb_mul(c: &mut Criterion) {
    let a = make_bn254(4096);
    let (challenge_full, lo, hi) = make_upper_limb_challenge();

    c.bench_function("BN254/chained_mul_standard", |b| {
        b.iter(|| {
            let mut acc = a[0];
            for _ in 0..4096 {
                acc *= challenge_full;
            }
            black_box(acc)
        })
    });

    c.bench_function("BN254/chained_mul_upper_limb", |b| {
        b.iter(|| {
            let mut acc = a[0];
            for _ in 0..4096 {
                acc = acc.mul_by_hi_2limbs(lo, hi);
            }
            black_box(acc)
        })
    });
}

// ---------------------------------------------------------------------------
// Combined benchmark (Section 6.4)
// Faithful to Jolt's DensePolynomial::bound_poly_var_top: in-place binding
// on a single interleaved array Z, where left = Z[..N], right = Z[N..].
//
// Criterion prepares a fresh working buffer outside the timed routine. This
// avoids subtracting a separately measured memcpy point estimate.
//
// The `_bf` cases repeat the full-challenge cases with branch-free addition
// and subtraction (`Bn254BranchFree`).
// ---------------------------------------------------------------------------

fn bench_combined(c: &mut Criterion) {
    let template = make_bn254(2 * N);
    let r_full = BN254Fr::from(0x123456789abcdef0u64);
    let (_, lo, hi) = make_upper_limb_challenge();
    let template_bf: Vec<Bn254BranchFree> = template.iter().copied().map(Bn254BranchFree).collect();
    let r_full_bf = Bn254BranchFree(r_full);
    let mut order = [0usize, 1, 2, 3, 4, 5];
    order.shuffle(&mut rand::thread_rng());

    for variant in order {
        match variant {
            0 => c.bench_function("BN254/combined_boolean_full", |b| {
                b.iter_batched(
                    || template.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            *a += r_full * (*b_val - *a);
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            1 => c.bench_function("BN254/combined_boolean_upper", |b| {
                b.iter_batched(
                    || template.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            let diff = *b_val - *a;
                            *a += diff.mul_by_hi_2limbs(lo, hi);
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            2 => c.bench_function("BN254/combined_projective_full", |b| {
                b.iter_batched(
                    || template.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            *a += r_full * *b_val;
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            3 => c.bench_function("BN254/combined_projective_upper", |b| {
                b.iter_batched(
                    || template.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            *a += b_val.mul_by_hi_2limbs(lo, hi);
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            4 => c.bench_function("BN254/combined_boolean_bf", |b| {
                b.iter_batched(
                    || template_bf.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            *a += r_full_bf * (*b_val - *a);
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            5 => c.bench_function("BN254/combined_projective_bf", |b| {
                b.iter_batched(
                    || template_bf.clone(),
                    |mut work| {
                        let (left, right) = work.split_at_mut(N);
                        for (a, b_val) in left.iter_mut().zip(right.iter()) {
                            *a += r_full_bf * *b_val;
                        }
                        black_box(left[0]);
                    },
                    BatchSize::LargeInput,
                )
            }),
            _ => unreachable!(),
        };
    }
}

criterion_group! {
    name = benches;
    config = benchmark_config();
    targets = bench_upper_limb_mul, bench_combined
}
criterion_main!(benches);
