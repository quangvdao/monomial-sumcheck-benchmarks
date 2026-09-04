use ark_ff::BigInt;
use monomial_sumcheck_benchmarks::sumcheck::{
    bn254_upper_limb_challenge_from_bytes, make_bn254, BN254Fr, Bn254UpperLimbMul,
};

#[test]
fn transcript_mapping_zeroes_low_limbs_and_top_three_bits() {
    let bytes = [0xff; 16];
    let (challenge, (limb_lo, limb_hi)) = bn254_upper_limb_challenge_from_bytes(bytes);

    assert_eq!(limb_lo, u64::MAX);
    assert_eq!(limb_hi, (1u64 << 61) - 1);
    assert_eq!(
        challenge,
        BN254Fr::new_unchecked(BigInt([0, 0, limb_lo, limb_hi]))
    );
}

#[test]
fn transcript_mapping_is_little_endian() {
    let bytes = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    let (_, (limb_lo, limb_hi)) = bn254_upper_limb_challenge_from_bytes(bytes);

    assert_eq!(limb_lo, 0x0706_0504_0302_0100);
    assert_eq!(limb_hi, 0x0f0e_0d0c_0b0a_0908 & ((1u64 << 61) - 1));
}

#[test]
fn mapped_challenge_matches_standard_multiplication() {
    let bytes = [0xa5; 16];
    let (challenge, (limb_lo, limb_hi)) = bn254_upper_limb_challenge_from_bytes(bytes);

    for value in make_bn254(32) {
        assert_eq!(value * challenge, value.mul_by_hi_2limbs(limb_lo, limb_hi));
    }
}
