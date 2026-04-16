use super::bb_ext::{BB5Accum, BB5MulByConst};
use super::*;

#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
pub fn bb5_eq_gruen_boolean_eval_ref(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5, BB5) {
    let half = f.len() / 2;
    let mut q1 = BB5::ZERO;
    let mut q_inf = BB5::ZERO;

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

    (q1, q_inf)
}

#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
pub fn bb5_eq_gruen_projective_eval_ref(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5, BB5) {
    let half = f.len() / 2;
    let mut q1 = BB5::ZERO;
    let mut q_inf = BB5::ZERO;

    for j in 0..half {
        let f0 = f[2 * j];
        let fi = f[2 * j + 1];
        let g0 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];

        q_inf += fi * gi * ew;
        let sf = f0 + fi;
        let sg = g0 + gi;
        q1 += sf * sg * ew;
    }

    (q1, q_inf)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn pack_bb5_chunk(values: [BB5; BB5_PACK_WIDTH]) -> PackedBB5 {
    PackedBB5::from_ext_slice(&values)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn sum_packed_bb5(acc: PackedBB5) -> BB5 {
    let mut sum = BB5::ZERO;
    for lane in 0..BB5_PACK_WIDTH {
        sum += <PackedBB5 as PackedFieldExtension<BabyBear, BB5>>::extract(&acc, lane);
    }
    sum
}

#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
pub fn bb5_eq_delayed_eval_ref(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5Accum, BB5Accum) {
    let half = f.len() / 2;
    let mut q1 = BB5Accum::zero();
    let mut q_inf = BB5Accum::zero();

    for j in 0..half {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let g0 = g[2 * j];
        let g1 = g[2 * j + 1];
        let ew = eq_rest[j];
        let df = f1 - f0;
        let dg = g1 - g0;

        q1.fmadd(f1 * g1, ew);
        q_inf.fmadd(df * dg, ew);
    }

    (q1, q_inf)
}

#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
pub fn bb5_eq_projective_delayed_eval_ref(
    f: &[BB5],
    g: &[BB5],
    eq_rest: &[BB5],
) -> (BB5Accum, BB5Accum) {
    let half = f.len() / 2;
    let mut q1 = BB5Accum::zero();
    let mut q_inf = BB5Accum::zero();

    for j in 0..half {
        let f0 = f[2 * j];
        let fi = f[2 * j + 1];
        let g0 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];

        q_inf.fmadd(fi * gi, ew);
        let sf = f0 + fi;
        let sg = g0 + gi;
        q1.fmadd(sf * sg, ew);
    }

    (q1, q_inf)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn bb_neon_sum_mul_4(a: PackedBabyBearNeon, b: PackedBabyBearNeon) -> u128 {
    let a_vec = vld1q_u32(a.as_slice().as_ptr() as *const u32);
    let b_vec = vld1q_u32(b.as_slice().as_ptr() as *const u32);
    let lo = vmull_u32(vget_low_u32(a_vec), vget_low_u32(b_vec));
    let hi = vmull_high_u32(a_vec, b_vec);
    vaddvq_u64(vaddq_u64(lo, hi)) as u128
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn bb5_accum_fmadd_packed(acc: &mut BB5Accum, a: PackedBB5, b: PackedBB5) {
    let a = a.as_basis_coefficients_slice();
    let b = b.as_basis_coefficients_slice();
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];

    unsafe {
        acc.c[0] += bb_neon_sum_mul_4(a0, b0);
        acc.c[1] += bb_neon_sum_mul_4(a0, b1) + bb_neon_sum_mul_4(a1, b0);
        acc.c[2] += bb_neon_sum_mul_4(a0, b2) + bb_neon_sum_mul_4(a1, b1) + bb_neon_sum_mul_4(a2, b0);
        acc.c[3] += bb_neon_sum_mul_4(a0, b3)
            + bb_neon_sum_mul_4(a1, b2)
            + bb_neon_sum_mul_4(a2, b1)
            + bb_neon_sum_mul_4(a3, b0);
        acc.c[4] += bb_neon_sum_mul_4(a0, b4)
            + bb_neon_sum_mul_4(a1, b3)
            + bb_neon_sum_mul_4(a2, b2)
            + bb_neon_sum_mul_4(a3, b1)
            + bb_neon_sum_mul_4(a4, b0);
        acc.c[5] += bb_neon_sum_mul_4(a1, b4)
            + bb_neon_sum_mul_4(a2, b3)
            + bb_neon_sum_mul_4(a3, b2)
            + bb_neon_sum_mul_4(a4, b1);
        acc.c[6] += bb_neon_sum_mul_4(a2, b4) + bb_neon_sum_mul_4(a3, b3) + bb_neon_sum_mul_4(a4, b2);
        acc.c[7] += bb_neon_sum_mul_4(a3, b4) + bb_neon_sum_mul_4(a4, b3);
        acc.c[8] += bb_neon_sum_mul_4(a4, b4);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(never)]
fn bb5_accum_fmadd_packed_outlined(acc: &mut BB5Accum, a: PackedBB5, b: PackedBB5) {
    let a = a.as_basis_coefficients_slice();
    let b = b.as_basis_coefficients_slice();
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];

    unsafe {
        acc.c[0] += bb_neon_sum_mul_4(a0, b0);
        acc.c[1] += bb_neon_sum_mul_4(a0, b1) + bb_neon_sum_mul_4(a1, b0);
        acc.c[2] += bb_neon_sum_mul_4(a0, b2) + bb_neon_sum_mul_4(a1, b1) + bb_neon_sum_mul_4(a2, b0);
        acc.c[3] += bb_neon_sum_mul_4(a0, b3)
            + bb_neon_sum_mul_4(a1, b2)
            + bb_neon_sum_mul_4(a2, b1)
            + bb_neon_sum_mul_4(a3, b0);
        acc.c[4] += bb_neon_sum_mul_4(a0, b4)
            + bb_neon_sum_mul_4(a1, b3)
            + bb_neon_sum_mul_4(a2, b2)
            + bb_neon_sum_mul_4(a3, b1)
            + bb_neon_sum_mul_4(a4, b0);
        acc.c[5] += bb_neon_sum_mul_4(a1, b4)
            + bb_neon_sum_mul_4(a2, b3)
            + bb_neon_sum_mul_4(a3, b2)
            + bb_neon_sum_mul_4(a4, b1);
        acc.c[6] += bb_neon_sum_mul_4(a2, b4) + bb_neon_sum_mul_4(a3, b3) + bb_neon_sum_mul_4(a4, b2);
        acc.c[7] += bb_neon_sum_mul_4(a3, b4) + bb_neon_sum_mul_4(a4, b3);
        acc.c[8] += bb_neon_sum_mul_4(a4, b4);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn bb5_eq_gruen_boolean_eval_packed(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5, BB5) {
    let half = f.len() / 2;
    let packed_end = half / BB5_PACK_WIDTH * BB5_PACK_WIDTH;
    let mut q1_packed = PackedBB5::ZERO;
    let mut q_inf_packed = PackedBB5::ZERO;
    let mut q1_tail = BB5::ZERO;
    let mut q_inf_tail = BB5::ZERO;

    for j in (0..packed_end).step_by(BB5_PACK_WIDTH) {
        let f0 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane)]));
        let f1 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane) + 1]));
        let g0 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane)]));
        let g1 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane) + 1]));
        let ew = PackedBB5::from_ext_slice(&eq_rest[j..j + BB5_PACK_WIDTH]);
        let df = f1 - f0;
        let dg = g1 - g0;

        q1_packed += (f1 * g1) * ew;
        q_inf_packed += (df * dg) * ew;
    }

    for j in packed_end..half {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let g0 = g[2 * j];
        let g1 = g[2 * j + 1];
        let ew = eq_rest[j];
        let df = f1 - f0;
        let dg = g1 - g0;

        q1_tail += f1 * g1 * ew;
        q_inf_tail += df * dg * ew;
    }

    (
        sum_packed_bb5(q1_packed) + q1_tail,
        sum_packed_bb5(q_inf_packed) + q_inf_tail,
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn bb5_eq_delayed_eval_packed(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5Accum, BB5Accum) {
    let half = f.len() / 2;
    let packed_end = half / BB5_PACK_WIDTH * BB5_PACK_WIDTH;
    let mut q1 = BB5Accum::zero();
    let mut q_inf = BB5Accum::zero();

    for j in (0..packed_end).step_by(BB5_PACK_WIDTH) {
        let f0 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane)]));
        let f1 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane) + 1]));
        let g0 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane)]));
        let g1 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane) + 1]));
        let ew = PackedBB5::from_ext_slice(&eq_rest[j..j + BB5_PACK_WIDTH]);
        let df = f1 - f0;
        let dg = g1 - g0;

        bb5_accum_fmadd_packed(&mut q1, f1 * g1, ew);
        bb5_accum_fmadd_packed(&mut q_inf, df * dg, ew);
    }

    for j in packed_end..half {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let g0 = g[2 * j];
        let g1 = g[2 * j + 1];
        let ew = eq_rest[j];
        let df = f1 - f0;
        let dg = g1 - g0;

        q1.fmadd(f1 * g1, ew);
        q_inf.fmadd(df * dg, ew);
    }

    (q1, q_inf)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn bb5_eq_gruen_projective_eval_packed(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5, BB5) {
    let half = f.len() / 2;
    let packed_end = half / BB5_PACK_WIDTH * BB5_PACK_WIDTH;
    let mut q1_packed = PackedBB5::ZERO;
    let mut q_inf_packed = PackedBB5::ZERO;
    let mut q1_tail = BB5::ZERO;
    let mut q_inf_tail = BB5::ZERO;

    for j in (0..packed_end).step_by(BB5_PACK_WIDTH) {
        let f0 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane)]));
        let fi = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane) + 1]));
        let g0 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane)]));
        let gi = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane) + 1]));
        let ew = PackedBB5::from_ext_slice(&eq_rest[j..j + BB5_PACK_WIDTH]);
        let sf = f0 + fi;
        let sg = g0 + gi;

        q_inf_packed += (fi * gi) * ew;
        q1_packed += (sf * sg) * ew;
    }

    for j in packed_end..half {
        let f0 = f[2 * j];
        let fi = f[2 * j + 1];
        let g0 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];
        let sf = f0 + fi;
        let sg = g0 + gi;

        q_inf_tail += fi * gi * ew;
        q1_tail += sf * sg * ew;
    }

    (
        sum_packed_bb5(q1_packed) + q1_tail,
        sum_packed_bb5(q_inf_packed) + q_inf_tail,
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn bb5_eq_projective_delayed_eval_packed(
    f: &[BB5],
    g: &[BB5],
    eq_rest: &[BB5],
) -> (BB5Accum, BB5Accum) {
    let half = f.len() / 2;
    let packed_end = half / BB5_PACK_WIDTH * BB5_PACK_WIDTH;
    let mut q1 = BB5Accum::zero();
    let mut q_inf = BB5Accum::zero();

    for j in (0..packed_end).step_by(BB5_PACK_WIDTH) {
        let fi = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane) + 1]));
        let gi = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane) + 1]));
        let ew = PackedBB5::from_ext_slice(&eq_rest[j..j + BB5_PACK_WIDTH]);

        bb5_accum_fmadd_packed(&mut q_inf, fi * gi, ew);
        let f0 = pack_bb5_chunk(core::array::from_fn(|lane| f[2 * (j + lane)]));
        let g0 = pack_bb5_chunk(core::array::from_fn(|lane| g[2 * (j + lane)]));
        let ew = PackedBB5::from_ext_slice(&eq_rest[j..j + BB5_PACK_WIDTH]);
        let sf = f0 + fi;
        let sg = g0 + gi;
        bb5_accum_fmadd_packed_outlined(&mut q1, sf * sg, ew);
    }

    for j in packed_end..half {
        let f0 = f[2 * j];
        let fi = f[2 * j + 1];
        let g0 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];
        let sf = f0 + fi;
        let sg = g0 + gi;

        q1.fmadd(sf * sg, ew);
        q_inf.fmadd(fi * gi, ew);
    }

    (q1, q_inf)
}

#[inline(always)]
pub fn bb5_eq_gruen_boolean_eval(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5, BB5) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        bb5_eq_gruen_boolean_eval_packed(f, g, eq_rest)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        bb5_eq_gruen_boolean_eval_ref(f, g, eq_rest)
    }
}

#[inline(always)]
pub fn bb5_eq_delayed_eval(f: &[BB5], g: &[BB5], eq_rest: &[BB5]) -> (BB5Accum, BB5Accum) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        bb5_eq_delayed_eval_packed(f, g, eq_rest)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        bb5_eq_delayed_eval_ref(f, g, eq_rest)
    }
}

#[inline(always)]
pub fn bb5_eq_projective_delayed_eval(
    f: &[BB5],
    g: &[BB5],
    eq_rest: &[BB5],
) -> (BB5Accum, BB5Accum) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        bb5_eq_projective_delayed_eval_packed(f, g, eq_rest)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        bb5_eq_projective_delayed_eval_ref(f, g, eq_rest)
    }
}

pub fn sumcheck_deg2_eq_gruen_boolean_bb5(
    f: &mut Vec<BB5>,
    g: &mut Vec<BB5>,
    suffix_eq: &[Vec<BB5>],
    challenges: &[BB5],
    zero: BB5,
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];
        let (q1, q_inf) = bb5_eq_gruen_boolean_eval(f, g, eq_rest);
        let _ = black_box((q1, q_inf));

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

    let _ = black_box(zero);
}

pub fn sumcheck_deg2_eq_delayed_bb5_packed(
    f: &mut Vec<BB5>,
    g: &mut Vec<BB5>,
    suffix_eq: &[Vec<BB5>],
    challenges: &[BB5],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];
        let (q1, q_inf) = bb5_eq_delayed_eval(f, g, eq_rest);
        let _ = black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        let mul_r = BB5MulByConst::new(r);
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            f[j] = f0 + mul_r.apply(f1 - f0);
            g[j] = g0 + mul_r.apply(g1 - g0);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_gruen_projective_bb5(
    f: &mut Vec<BB5>,
    g: &mut Vec<BB5>,
    suffix_eq: &[Vec<BB5>],
    challenges: &[BB5],
    zero: BB5,
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        let (q1, q_inf) = bb5_eq_gruen_projective_eval_packed(f, g, eq_rest);
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        let (q1, q_inf) = bb5_eq_gruen_projective_eval_ref(f, g, eq_rest);
        let _ = black_box((q1, q_inf));

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

    let _ = black_box(zero);
}

pub fn sumcheck_deg2_eq_projective_delayed_bb5_packed(
    f: &mut Vec<BB5>,
    g: &mut Vec<BB5>,
    suffix_eq: &[Vec<BB5>],
    challenges: &[BB5],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];
        let (q1, q_inf) = bb5_eq_projective_delayed_eval(f, g, eq_rest);
        let _ = black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        let mul_r = BB5MulByConst::new(r);
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = f0 + mul_r.apply(fi);
            g[j] = g0 + mul_r.apply(gi);
        }

        f.truncate(half);
        g.truncate(half);
    }
}
