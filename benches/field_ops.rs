use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup as ArkAdditiveGroup, Field as ArkField, UniformRand};
use binius_field::BinaryField128bGhash;
use binius_field::Field as BiniusField;
use binius_field::PackedBinaryGhash1x128b as GF128Packed;
use binius_field::PackedField as BiniusPackedField;
use hachi_pcs::algebra::{Fp128Packing, PackedField, PackedValue, Prime128Offset275};
use hachi_pcs::{AdditiveGroup as HachiAdditiveGroup, CanonicalField, FieldCore};
use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type BB4 = BinomialExtensionField<BabyBear, 4>;
type BB5 = BinomialExtensionField<BabyBear, 5>;
type KB5 = QuinticTrinomialExtensionField<KoalaBear>;
type PackedFp128 = Fp128Packing<0xfffffffffffffffffffffffffffffeedu128>;

const N: usize = 4096;

fn rng() -> StdRng {
    StdRng::seed_from_u64(0x12345678deadbeef)
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
// Throughput: in-register batch of BATCH independent ops per pass.
// Uses stride BATCH/2 so the compiler can pipeline without serial deps.
// ---------------------------------------------------------------------------

const BATCH: usize = 16;
const PASSES: usize = 64;

fn thr_add<F: Copy + Add<Output = F>>(name: &str, seed: &[F], c: &mut Criterion) {
    let mut batch: [F; BATCH] = std::array::from_fn(|i| seed[i % seed.len()]);
    c.bench_function(&format!("{name}/thr_add"), |b| {
        b.iter(|| {
            for _ in 0..PASSES {
                for i in 0..BATCH {
                    batch[i] = batch[i] + batch[(i + BATCH / 2) % BATCH];
                }
            }
            black_box(batch)
        })
    });
}

fn thr_sub<F: Copy + Sub<Output = F>>(name: &str, seed: &[F], c: &mut Criterion) {
    let mut batch: [F; BATCH] = std::array::from_fn(|i| seed[i % seed.len()]);
    c.bench_function(&format!("{name}/thr_sub"), |b| {
        b.iter(|| {
            for _ in 0..PASSES {
                for i in 0..BATCH {
                    batch[i] = batch[i] - batch[(i + BATCH / 2) % BATCH];
                }
            }
            black_box(batch)
        })
    });
}

fn thr_mul<F: Copy + Mul<Output = F>>(name: &str, seed: &[F], c: &mut Criterion) {
    let mut batch: [F; BATCH] = std::array::from_fn(|i| seed[i % seed.len()]);
    c.bench_function(&format!("{name}/thr_mul"), |b| {
        b.iter(|| {
            for _ in 0..PASSES {
                for i in 0..BATCH {
                    batch[i] = batch[i] * batch[(i + BATCH / 2) % BATCH];
                }
            }
            black_box(batch)
        })
    });
}

// ---------------------------------------------------------------------------
// Element generation: proper random field elements via each library's RNG
// ---------------------------------------------------------------------------

fn make_babybear(n: usize) -> Vec<BabyBear> {
    let mut r = rng();
    (0..n)
        .map(|_| BabyBear::from_u32(r.gen::<u32>() % ((1u64 << 31) - (1u64 << 27) + 1) as u32))
        .collect()
}

fn make_bb_ext<const D: usize>(n: usize) -> Vec<BinomialExtensionField<BabyBear, D>> {
    let mut r = rng();
    let p = ((1u64 << 31) - (1u64 << 27) + 1) as u32;
    (0..n)
        .map(|_| {
            let base: [BabyBear; D] = std::array::from_fn(|_| BabyBear::from_u32(r.gen::<u32>() % p));
            BinomialExtensionField::new(base)
        })
        .collect()
}

fn make_kb5(n: usize) -> Vec<KB5> {
    let mut r = rng();
    let p = ((1u64 << 31) - (1u64 << 24) + 1) as u32;
    (0..n)
        .map(|_| {
            let base: [KoalaBear; 5] =
                std::array::from_fn(|_| KoalaBear::from_u32(r.gen::<u32>() % p));
            KB5::new(base)
        })
        .collect()
}

fn make_fp128(n: usize) -> Vec<Prime128Offset275> {
    let mut r = rng();
    (0..n)
        .map(|_| Prime128Offset275::from_canonical_u128_reduced(r.gen::<u128>()))
        .collect()
}

fn make_fp128_packed(n: usize) -> Vec<PackedFp128> {
    let scalars = make_fp128(n);
    PackedFp128::pack_slice(&scalars)
}

fn make_bn254(n: usize) -> Vec<BN254Fr> {
    let mut r = rng();
    (0..n).map(|_| BN254Fr::rand(&mut r)).collect()
}

fn make_gf128(n: usize) -> Vec<GF128Packed> {
    let mut r = rng();
    (0..n)
        .map(|_| GF128Packed::set_single(BinaryField128bGhash::new(r.gen::<u128>())))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_babybear(c: &mut Criterion) {
    let a = make_babybear(N);
    let name = "BabyBear";

    lat_add(name, BabyBear::ZERO, &a, c);
    lat_sub(name, BabyBear::ZERO, &a, c);
    lat_mul(name, BabyBear::ONE, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_babybear_ext4(c: &mut Criterion) {
    let a = make_bb_ext::<4>(N);
    let name = "BabyBear_Ext4";

    lat_add(name, BB4::ZERO, &a, c);
    lat_sub(name, BB4::ZERO, &a, c);
    lat_mul(name, BB4::ONE, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_babybear_ext5(c: &mut Criterion) {
    let a = make_bb_ext::<5>(N);
    let name = "BabyBear_Ext5";

    lat_add(name, BB5::ZERO, &a, c);
    lat_sub(name, BB5::ZERO, &a, c);
    lat_mul(name, BB5::ONE, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_koalabear_ext5(c: &mut Criterion) {
    let a = make_kb5(N);
    let name = "KoalaBear_Ext5";

    lat_add(name, KB5::ZERO, &a, c);
    lat_sub(name, KB5::ZERO, &a, c);
    lat_mul(name, KB5::ONE, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_fp128(c: &mut Criterion) {
    type F = Prime128Offset275;
    let a = make_fp128(N);
    let name = "Fp128";

    lat_add(name, F::ZERO, &a, c);
    lat_sub(name, F::ZERO, &a, c);
    lat_mul(name, F::one(), &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_fp128_packed(c: &mut Criterion) {
    let a = make_fp128_packed(N);
    let zero = PackedFp128::broadcast(Prime128Offset275::ZERO);
    let one = PackedFp128::broadcast(Prime128Offset275::one());
    let name = "Fp128_Packed";

    lat_add(name, zero, &a, c);
    lat_sub(name, zero, &a, c);
    lat_mul(name, one, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_bn254(c: &mut Criterion) {
    type F = BN254Fr;
    let a = make_bn254(N);
    let name = "BN254_Fr";

    lat_add(name, F::ZERO, &a, c);
    lat_sub(name, F::ZERO, &a, c);
    lat_mul(name, F::ONE, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

fn bench_gf128(c: &mut Criterion) {
    let a = make_gf128(N);
    let zero = GF128Packed::set_single(BinaryField128bGhash::ZERO);
    let one = GF128Packed::set_single(BinaryField128bGhash::ONE);
    let name = "GF128_Ghash";

    lat_add(name, zero, &a, c);
    lat_sub(name, zero, &a, c);
    lat_mul(name, one, &a, c);
    thr_add(name, &a, c);
    thr_sub(name, &a, c);
    thr_mul(name, &a, c);
}

criterion_group!(
    benches,
    bench_babybear,
    bench_babybear_ext4,
    bench_babybear_ext5,
    bench_koalabear_ext5,
    bench_fp128,
    bench_fp128_packed,
    bench_bn254,
    bench_gf128
);
criterion_main!(benches);
