#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    uint32x2_t, uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vaddvq_u64, vandq_u32, vcgtq_u32,
    vcreate_u32, vdup_n_u32, vdupq_n_u32, vdupq_n_u64, veorq_u64, vget_high_u32, vget_low_u32,
    vld1q_u32, vminq_u32, vmlal_lane_u32, vmull_high_u32, vmull_lane_u32, vmull_p64, vmull_u32,
    vmulq_u32, vreinterpretq_u32_u64, vsubq_u32, vuzp1q_u32, vuzp2q_u32,
};
use std::ops::{Add, AddAssign, Mul, Sub};

use criterion::black_box;

pub use ark_bn254::Fr as BN254Fr;
use ark_bn254::FrConfig;
use ark_ff::{AdditiveGroup, MontConfig};
pub use binius_field::BinaryField128bGhash as GF128;
use hachi_pcs::algebra::Prime128Offset275;
use hachi_pcs::{AdditiveGroup as HachiAdditiveGroup, CanonicalField, FieldCore, Invertible};
use p3_baby_bear::BabyBear;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_baby_bear::PackedBabyBearNeon;
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_koala_bear::KoalaBear;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use p3_koala_bear::PackedKoalaBearNeon;

pub type BB4 = BinomialExtensionField<BabyBear, 4>;
pub type BB5 = BinomialExtensionField<BabyBear, 5>;
pub type KB5 = QuinticTrinomialExtensionField<KoalaBear>;
pub type Fp128 = Prime128Offset275;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
type PackedBB5 = <BB5 as ExtensionField<BabyBear>>::ExtensionPacking;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const BB5_PACK_WIDTH: usize = 4;

mod bb5_packed;
mod bb_ext;
mod bn254;
mod data;
mod fp128;
mod generic;
mod gf128;

pub use bb5_packed::*;
pub use bb_ext::*;
pub use bn254::*;
pub use data::*;
pub use fp128::*;
pub use generic::*;
pub use gf128::*;
