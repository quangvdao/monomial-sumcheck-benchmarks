use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::AdditiveGroup;

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
// Boolean-hypercube binding: out[i] = p0[i] + r * (p1[i] - p0[i])
// 1 mul + 1 add + 1 sub per element
// ---------------------------------------------------------------------------

fn bind_boolean_loop(p0: &[BN254Fr], p1: &[BN254Fr], r: BN254Fr, out: &mut [BN254Fr]) {
    for i in 0..p0.len() {
        out[i] = p0[i] + r * (p1[i] - p0[i]);
    }
}

// ---------------------------------------------------------------------------
// Projective binding: out[i] = p0[i] + r * p_inf[i]
// 1 mul + 1 add per element (no subtraction)
// ---------------------------------------------------------------------------

fn bind_projective_loop(p0: &[BN254Fr], p_inf: &[BN254Fr], r: BN254Fr, out: &mut [BN254Fr]) {
    for i in 0..p0.len() {
        out[i] = p0[i] + r * p_inf[i];
    }
}

// ---------------------------------------------------------------------------
// Full-loop benchmarks (throughput over 2^20 elements)
// ---------------------------------------------------------------------------

fn bench_binding_full_loop(c: &mut Criterion) {
    let p0 = make_bn254(N);
    let p1 = make_bn254(N);
    let p_inf = make_bn254(N);
    let r = BN254Fr::from(0x123456789abcdef0u64);
    let mut out = vec![BN254Fr::ZERO; N];

    c.bench_function("BN254/bind_boolean_full", |b| {
        b.iter(|| {
            bind_boolean_loop(
                black_box(&p0),
                black_box(&p1),
                black_box(r),
                &mut out,
            );
            black_box(&out);
        })
    });

    c.bench_function("BN254/bind_projective_full", |b| {
        b.iter(|| {
            bind_projective_loop(
                black_box(&p0),
                black_box(&p_inf),
                black_box(r),
                &mut out,
            );
            black_box(&out);
        })
    });
}

// ---------------------------------------------------------------------------
// Per-element latency benchmarks (serial dependency chain)
// ---------------------------------------------------------------------------

fn bench_binding_latency(c: &mut Criterion) {
    let p0 = make_bn254(4096);
    let p1 = make_bn254(4096);
    let p_inf = make_bn254(4096);
    let r = BN254Fr::from(0x123456789abcdef0u64);

    c.bench_function("BN254/bind_boolean_lat", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ZERO;
            for i in 0..p0.len() {
                acc += p0[i] + r * (p1[i] - p0[i]);
            }
            black_box(acc)
        })
    });

    c.bench_function("BN254/bind_projective_lat", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ZERO;
            for i in 0..p0.len() {
                acc += p0[i] + r * p_inf[i];
            }
            black_box(acc)
        })
    });
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
// Binding with upper-limb challenge (Section 6.3)
// ---------------------------------------------------------------------------

fn bind_projective_upper_limb_loop(
    p0: &[BN254Fr],
    p_inf: &[BN254Fr],
    r_lo: u64,
    r_hi: u64,
    out: &mut [BN254Fr],
) {
    for i in 0..p0.len() {
        out[i] = p0[i] + p_inf[i].mul_by_hi_2limbs(r_lo, r_hi);
    }
}

fn bench_upper_limb_binding(c: &mut Criterion) {
    let p0 = make_bn254(N);
    let p_inf = make_bn254(N);
    let (_, lo, hi) = make_upper_limb_challenge();
    let mut out = vec![BN254Fr::ZERO; N];

    c.bench_function("BN254/bind_proj_upper_limb_full", |b| {
        b.iter(|| {
            bind_projective_upper_limb_loop(
                black_box(&p0),
                black_box(&p_inf),
                black_box(lo),
                black_box(hi),
                &mut out,
            );
            black_box(&out);
        })
    });
}

// ---------------------------------------------------------------------------
// Combined benchmark (Section 6.4)
// Baseline: Boolean + full-field challenge
// Optimized: Projective + upper-limb challenge
// ---------------------------------------------------------------------------

fn bench_combined(c: &mut Criterion) {
    let p0 = make_bn254(N);
    let p1 = make_bn254(N);
    let p_inf = make_bn254(N);
    let r_full = BN254Fr::from(0x123456789abcdef0u64);
    let (_, lo, hi) = make_upper_limb_challenge();
    let mut out = vec![BN254Fr::ZERO; N];

    c.bench_function("BN254/combined_baseline", |b| {
        b.iter(|| {
            bind_boolean_loop(
                black_box(&p0),
                black_box(&p1),
                black_box(r_full),
                &mut out,
            );
            black_box(&out);
        })
    });

    c.bench_function("BN254/combined_optimized", |b| {
        b.iter(|| {
            bind_projective_upper_limb_loop(
                black_box(&p0),
                black_box(&p_inf),
                black_box(lo),
                black_box(hi),
                &mut out,
            );
            black_box(&out);
        })
    });
}

criterion_group!(
    benches,
    bench_binding_full_loop,
    bench_binding_latency,
    bench_upper_limb_mul,
    bench_upper_limb_binding,
    bench_combined
);
criterion_main!(benches);
