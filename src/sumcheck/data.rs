use super::*;

pub fn make_u64s(n: usize) -> Vec<u64> {
    let mut vals = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef12345678;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        vals.push(state);
    }
    vals
}

pub fn make_bn254(n: usize) -> Vec<BN254Fr> {
    make_u64s(n).iter().map(|&v| BN254Fr::from(v)).collect()
}

pub fn make_bb4(n: usize) -> Vec<BB4> {
    let raw = make_u64s(n * 4);
    raw.chunks(4)
        .map(|chunk| {
            let base: [BabyBear; 4] = std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB4::new(base)
        })
        .collect()
}

pub fn make_bb5(n: usize) -> Vec<BB5> {
    let raw = make_u64s(n * 5);
    raw.chunks(5)
        .map(|chunk| {
            let base: [BabyBear; 5] = std::array::from_fn(|i| BabyBear::from_u32(chunk[i] as u32));
            BB5::new(base)
        })
        .collect()
}

pub fn make_kb5(n: usize) -> Vec<KB5> {
    let raw = make_u64s(n * 5);
    raw.chunks(5)
        .map(|chunk| KB5::from_basis_coefficients_fn(|i| KoalaBear::from_u32(chunk[i] as u32)))
        .collect()
}

pub fn make_fp128(n: usize) -> Vec<Fp128> {
    let raw = make_u64s(n * 2);
    raw.chunks(2)
        .map(|chunk| {
            let v = (chunk[0] as u128) | ((chunk[1] as u128) << 64);
            Fp128::from_canonical_u128_reduced(v)
        })
        .collect()
}

pub fn make_gf128(n: usize) -> Vec<GF128> {
    let raw = make_u64s(n * 2);
    raw.chunks(2)
        .map(|chunk| {
            let v = (chunk[0] as u128) | ((chunk[1] as u128) << 64);
            GF128::new(v)
        })
        .collect()
}

pub fn make_bn254_upper_limb_challenges(n: usize) -> (Vec<BN254Fr>, Vec<(u64, u64)>) {
    let raw = make_u64s(n * 2);
    let mut challenges = Vec::with_capacity(n);
    let mut limbs = Vec::with_capacity(n);
    for chunk in raw.chunks(2) {
        let bytes = ((chunk[0] as u128) | ((chunk[1] as u128) << 64)).to_le_bytes();
        let (challenge, challenge_limbs) = bn254_upper_limb_challenge_from_bytes(bytes);
        challenges.push(challenge);
        limbs.push(challenge_limbs);
    }
    (challenges, limbs)
}

/// Map 128 uniformly distributed transcript bits to the 125-bit BN254
/// upper-limb challenge set used by the specialized multiplication kernel.
///
/// The three most significant input bits are discarded. The returned field
/// element is already in Montgomery representation with two zero low limbs.
pub fn bn254_upper_limb_challenge_from_bytes(bytes: [u8; 16]) -> (BN254Fr, (u64, u64)) {
    let limb_lo = u64::from_le_bytes(bytes[..8].try_into().expect("slice length is fixed"));
    let limb_hi = u64::from_le_bytes(bytes[8..].try_into().expect("slice length is fixed"))
        & ((1u64 << 61) - 1);
    (
        BN254Fr::new_unchecked(ark_ff::BigInt([0, 0, limb_lo, limb_hi])),
        (limb_lo, limb_hi),
    )
}

/// Build suffix eq tables: tables[k] = eq(w[k..n], ·) of size 2^{n-k}.
/// At round k the prover needs eq_rest = tables[k+1] (size 2^{n-k-1}).
pub fn build_suffix_eq_tables<F>(w: &[F], one: F) -> Vec<Vec<F>>
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

/// Build coefficient tables for the projective equality polynomial
/// `prod_i (1 + w_i X_i)`.
///
/// `tables[k]` contains the monomial coefficients for variables `k..n`, in
/// the same low-variable-first layout consumed by the projective kernels.
pub fn build_suffix_eq_tables_projective<F>(w: &[F], one: F) -> Vec<Vec<F>>
where
    F: Copy + Mul<Output = F>,
{
    let n = w.len();
    let mut tables: Vec<Vec<F>> = Vec::with_capacity(n + 1);
    tables.resize_with(n + 1, Vec::new);
    tables[n] = vec![one];
    for k in (0..n).rev() {
        let prev_len = tables[k + 1].len();
        let mut cur = Vec::with_capacity(prev_len * 2);
        for i in 0..prev_len {
            let coefficient = tables[k + 1][i];
            cur.push(coefficient);
            cur.push(coefficient * w[k]);
        }
        tables[k] = cur;
    }
    tables
}
