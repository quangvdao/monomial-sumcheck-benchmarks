use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup as ArkAdditiveGroup, Field as ArkField};
use hachi_pcs::algebra::{Fp128Packing, PackedField, PackedValue, Prime128Offset275};
use hachi_pcs::{AdditiveGroup as HachiAdditiveGroup, CanonicalField, FieldCore};
use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear;

type BB4 = BinomialExtensionField<BabyBear, 4>;
type BB5 = BinomialExtensionField<BabyBear, 5>;
type KB5 = QuinticTrinomialExtensionField<KoalaBear>;
type PackedFp128 = Fp128Packing<0xfffffffffffffffffffffffffffffeedu128>;

const N: usize = 4096;

fn make_u64s(n: usize) -> Vec<u64> {
    let mut vals = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        vals.push(state);
    }
    vals
}

// ---------------------------------------------------------------------------
// Latency: serial dependency chain  acc = acc OP x[i]
// ---------------------------------------------------------------------------

fn lat_add<F: Copy + AddAssign>(name: &str, zero: F, elems: &[F], c: &mut Criterion) {
    c.bench_function(&format!("{name}/lat_add"), |b| {
        b.iter(|| {
            let mut acc = zero;
            for x in elems.iter() {
                acc += *x;
            }
            black_box(acc)
        })
    });
}

fn lat_sub<F: Copy + SubAssign>(name: &str, zero: F, elems: &[F], c: &mut Criterion) {
    c.bench_function(&format!("{name}/lat_sub"), |b| {
        b.iter(|| {
            let mut acc = zero;
            for x in elems.iter() {
                acc -= *x;
            }
            black_box(acc)
        })
    });
}

fn lat_mul<F: Copy + MulAssign>(name: &str, one: F, elems: &[F], c: &mut Criterion) {
    c.bench_function(&format!("{name}/lat_mul"), |b| {
        b.iter(|| {
            let mut acc = one;
            for x in elems.iter() {
                acc *= *x;
            }
            black_box(acc)
        })
    });
}

// ---------------------------------------------------------------------------
// Throughput: independent pairwise ops  out[i] = a[i] OP b[i]
// No serial dependency between iterations, only the target operation.
// ---------------------------------------------------------------------------

fn thr_add<F: Copy + Add<Output = F>>(
    name: &str,
    a: &[F],
    b: &[F],
    out: &mut [F],
    c: &mut Criterion,
) {
    c.bench_function(&format!("{name}/thr_add"), |bench| {
        bench.iter(|| {
            for i in 0..a.len() {
                out[i] = a[i] + b[i];
            }
            black_box(&*out);
        })
    });
}

fn thr_sub<F: Copy + Sub<Output = F>>(
    name: &str,
    a: &[F],
    b: &[F],
    out: &mut [F],
    c: &mut Criterion,
) {
    c.bench_function(&format!("{name}/thr_sub"), |bench| {
        bench.iter(|| {
            for i in 0..a.len() {
                out[i] = a[i] - b[i];
            }
            black_box(&*out);
        })
    });
}

fn thr_mul<F: Copy + Mul<Output = F>>(
    name: &str,
    a: &[F],
    b: &[F],
    out: &mut [F],
    c: &mut Criterion,
) {
    c.bench_function(&format!("{name}/thr_mul"), |bench| {
        bench.iter(|| {
            for i in 0..a.len() {
                out[i] = a[i] * b[i];
            }
            black_box(&*out);
        })
    });
}

// ---------------------------------------------------------------------------
// Element generation
// ---------------------------------------------------------------------------

fn make_babybear(n: usize) -> Vec<BabyBear> {
    make_u64s(n)
        .iter()
        .map(|&v| BabyBear::from_u32(v as u32))
        .collect()
}

fn make_bb_ext<const D: usize>(n: usize) -> Vec<BinomialExtensionField<BabyBear, D>> {
    let raw = make_u64s(n * D);
    raw.chunks(D)
        .map(|chunk| {
            let base: [BabyBear; D] =
                std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BinomialExtensionField::new(base)
        })
        .collect()
}

fn make_koalabear(n: usize) -> Vec<KoalaBear> {
    make_u64s(n)
        .iter()
        .map(|&v| KoalaBear::from_u32(v as u32))
        .collect()
}

fn make_kb5(n: usize) -> Vec<KB5> {
    let raw = make_u64s(n * 5);
    raw.chunks(5)
        .map(|chunk| {
            let base: [KoalaBear; 5] =
                std::array::from_fn(|i| KoalaBear::from_u32(chunk[i] as u32));
            KB5::new(base)
        })
        .collect()
}

fn make_fp128(n: usize) -> Vec<Prime128Offset275> {
    let raw = make_u64s(n * 2);
    raw.chunks(2)
        .map(|chunk| {
            let v = (chunk[0] as u128) | ((chunk[1] as u128) << 64);
            Prime128Offset275::from_canonical_u128_reduced(v)
        })
        .collect()
}

fn make_fp128_packed(n: usize) -> Vec<PackedFp128> {
    let scalars = make_fp128(n);
    PackedFp128::pack_slice(&scalars)
}

fn make_bn254(n: usize) -> Vec<BN254Fr> {
    make_u64s(n).iter().map(|&v| BN254Fr::from(v)).collect()
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_babybear(c: &mut Criterion) {
    let a = make_babybear(N);
    let b = make_babybear(N);
    let mut out = vec![BabyBear::ZERO; N];
    let name = "BabyBear";

    lat_add(name, BabyBear::ZERO, &a, c);
    lat_sub(name, BabyBear::ZERO, &a, c);
    lat_mul(name, BabyBear::ONE, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_babybear_ext4(c: &mut Criterion) {
    let a = make_bb_ext::<4>(N);
    let b = make_bb_ext::<4>(N);
    let mut out = vec![BB4::ZERO; N];
    let name = "BabyBear_Ext4";

    lat_add(name, BB4::ZERO, &a, c);
    lat_sub(name, BB4::ZERO, &a, c);
    lat_mul(name, BB4::ONE, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_babybear_ext5(c: &mut Criterion) {
    let a = make_bb_ext::<5>(N);
    let b = make_bb_ext::<5>(N);
    let mut out = vec![BB5::ZERO; N];
    let name = "BabyBear_Ext5";

    lat_add(name, BB5::ZERO, &a, c);
    lat_sub(name, BB5::ZERO, &a, c);
    lat_mul(name, BB5::ONE, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_koalabear_ext5(c: &mut Criterion) {
    let a = make_kb5(N);
    let b = make_kb5(N);
    let mut out = vec![KB5::ZERO; N];
    let name = "KoalaBear_Ext5";

    lat_add(name, KB5::ZERO, &a, c);
    lat_sub(name, KB5::ZERO, &a, c);
    lat_mul(name, KB5::ONE, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_fp128(c: &mut Criterion) {
    type F = Prime128Offset275;
    let a = make_fp128(N);
    let b = make_fp128(N);
    let mut out = vec![F::ZERO; N];
    let name = "Fp128";

    lat_add(name, F::ZERO, &a, c);
    lat_sub(name, F::ZERO, &a, c);
    lat_mul(name, F::one(), &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_fp128_packed(c: &mut Criterion) {
    let a = make_fp128_packed(N);
    let b = make_fp128_packed(N);
    let zero = PackedFp128::broadcast(Prime128Offset275::ZERO);
    let one = PackedFp128::broadcast(Prime128Offset275::one());
    let mut out = vec![zero; a.len()];
    let name = "Fp128_Packed";

    lat_add(name, zero, &a, c);
    lat_sub(name, zero, &a, c);
    lat_mul(name, one, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

fn bench_bn254(c: &mut Criterion) {
    type F = BN254Fr;
    let a = make_bn254(N);
    let b = make_bn254(N);
    let mut out = vec![F::ZERO; N];
    let name = "BN254_Fr";

    lat_add(name, F::ZERO, &a, c);
    lat_sub(name, F::ZERO, &a, c);
    lat_mul(name, F::ONE, &a, c);
    thr_add(name, &a, &b, &mut out, c);
    thr_sub(name, &a, &b, &mut out, c);
    thr_mul(name, &a, &b, &mut out, c);
}

criterion_group!(
    benches,
    bench_babybear,
    bench_babybear_ext4,
    bench_babybear_ext5,
    bench_koalabear_ext5,
    bench_fp128,
    bench_fp128_packed,
    bench_bn254
);
criterion_main!(benches);
