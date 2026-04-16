use super::*;

const BB_PRIME: u64 = 0x78000001;
const BB_MONTY_MU: u64 = 0x88000001;
const BB_MONTY_INV: u64 = BB_PRIME - ((BB_PRIME.wrapping_mul(BB_MONTY_MU) - 1) >> 32);
const BB_W4: u128 = 11;
const BB_W5: u128 = 2;

#[inline(always)]
fn bb_monty_reduce_u128(x: u128) -> u32 {
    ((x.wrapping_mul(BB_MONTY_INV as u128)) % (BB_PRIME as u128)) as u32
}

#[inline(always)]
fn bb4_to_limbs(x: BB4) -> [u32; 4] {
    const _: () = assert!(std::mem::size_of::<BB4>() == std::mem::size_of::<[u32; 4]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb4_to_coeffs(x: BB4) -> [BabyBear; 4] {
    const _: () = assert!(std::mem::size_of::<BB4>() == std::mem::size_of::<[BabyBear; 4]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb4_from_limbs(limbs: [u32; 4]) -> BB4 {
    unsafe { std::mem::transmute(limbs) }
}

#[inline(always)]
fn bb5_to_limbs(x: BB5) -> [u32; 5] {
    const _: () = assert!(std::mem::size_of::<BB5>() == std::mem::size_of::<[u32; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb5_to_coeffs(x: BB5) -> [BabyBear; 5] {
    const _: () = assert!(std::mem::size_of::<BB5>() == std::mem::size_of::<[BabyBear; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn bb5_from_limbs(limbs: [u32; 5]) -> BB5 {
    unsafe { std::mem::transmute(limbs) }
}

struct BB4Accum {
    c: [u128; 7],
}

impl BB4Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self { c: [0u128; 7] }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: BB4, b: BB4) {
        let a = bb4_to_limbs(a);
        let b = bb4_to_limbs(b);
        let (a0, a1, a2, a3) = (a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64);
        let (b0, b1, b2, b3) = (b[0] as u64, b[1] as u64, b[2] as u64, b[3] as u64);

        self.c[0] += (a0 * b0) as u128;
        self.c[1] += (a0 * b1 + a1 * b0) as u128;
        self.c[2] += (a0 * b2 + a1 * b1 + a2 * b0) as u128;
        self.c[3] += (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) as u128;
        self.c[4] += (a1 * b3 + a2 * b2 + a3 * b1) as u128;
        self.c[5] += (a2 * b3 + a3 * b2) as u128;
        self.c[6] += (a3 * b3) as u128;
    }

    fn reduce(self) -> BB4 {
        let d0 = self.c[0] + BB_W4 * self.c[4];
        let d1 = self.c[1] + BB_W4 * self.c[5];
        let d2 = self.c[2] + BB_W4 * self.c[6];
        let d3 = self.c[3];
        bb4_from_limbs([
            bb_monty_reduce_u128(d0),
            bb_monty_reduce_u128(d1),
            bb_monty_reduce_u128(d2),
            bb_monty_reduce_u128(d3),
        ])
    }
}

pub struct BB5Accum {
    pub(crate) c: [u128; 9],
}

impl BB5Accum {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self { c: [0u128; 9] }
    }

    #[inline(always)]
    pub(crate) fn fmadd(&mut self, a: BB5, b: BB5) {
        let a = bb5_to_limbs(a);
        let b = bb5_to_limbs(b);
        let (a0, a1, a2, a3, a4) = (a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64, a[4] as u64);
        let (b0, b1, b2, b3, b4) = (b[0] as u64, b[1] as u64, b[2] as u64, b[3] as u64, b[4] as u64);

        self.c[0] += (a0 * b0) as u128;
        self.c[1] += (a0 * b1 + a1 * b0) as u128;
        self.c[2] += (a0 * b2 + a1 * b1 + a2 * b0) as u128;
        self.c[3] += (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) as u128;
        self.c[4] += (a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1) as u128;
        self.c[4] += (a4 * b0) as u128;
        self.c[5] += (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1) as u128;
        self.c[6] += (a2 * b4 + a3 * b3 + a4 * b2) as u128;
        self.c[7] += (a3 * b4 + a4 * b3) as u128;
        self.c[8] += (a4 * b4) as u128;
    }

    pub fn reduce(self) -> BB5 {
        let d0 = self.c[0] + BB_W5 * self.c[5];
        let d1 = self.c[1] + BB_W5 * self.c[6];
        let d2 = self.c[2] + BB_W5 * self.c[7];
        let d3 = self.c[3] + BB_W5 * self.c[8];
        let d4 = self.c[4];
        bb5_from_limbs([
            bb_monty_reduce_u128(d0),
            bb_monty_reduce_u128(d1),
            bb_monty_reduce_u128(d2),
            bb_monty_reduce_u128(d3),
            bb_monty_reduce_u128(d4),
        ])
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn bb_neon_monty_reduce(res_lo: uint64x2_t, res_hi: uint64x2_t) -> uint32x4_t {
    let lo = vuzp1q_u32(vreinterpretq_u32_u64(res_lo), vreinterpretq_u32_u64(res_hi));
    let hi = vuzp2q_u32(vreinterpretq_u32_u64(res_lo), vreinterpretq_u32_u64(res_hi));
    let neg_p = vdupq_n_u32(0x87FFFFFFu32);
    let hi_reduced = vminq_u32(hi, vaddq_u32(hi, neg_p));
    let mu = vdupq_n_u32(BB_MONTY_MU as u32);
    let m = vmulq_u32(lo, mu);
    let p = vdupq_n_u32(BB_PRIME as u32);
    let monty = vuzp2q_u32(
        vreinterpretq_u32_u64(vmull_u32(vget_low_u32(m), vget_low_u32(p))),
        vreinterpretq_u32_u64(vmull_high_u32(m, p)),
    );
    let cmp = vcgtq_u32(monty, hi_reduced);
    vaddq_u32(vsubq_u32(hi_reduced, monty), vandq_u32(cmp, p))
}

struct BB4MulByConst {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    r: BB4,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    cols_lo: [uint32x2_t; 4],
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    cols_hi: [uint32x2_t; 4],
}

impl BB4MulByConst {
    #[inline(always)]
    fn new(r: BB4) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let [r0, r1, r2, r3] = bb4_to_coeffs(r);
            let w = BabyBear::from_u32(BB_W4 as u32);
            let wr1: u32 = unsafe { std::mem::transmute(w * r1) };
            let wr2: u32 = unsafe { std::mem::transmute(w * r2) };
            let wr3: u32 = unsafe { std::mem::transmute(w * r3) };
            let r0: u32 = unsafe { std::mem::transmute(r0) };
            let r1: u32 = unsafe { std::mem::transmute(r1) };
            let r2: u32 = unsafe { std::mem::transmute(r2) };
            let r3: u32 = unsafe { std::mem::transmute(r3) };
            unsafe {
                let cols_lo = [
                    vcreate_u32((r0 as u64) | ((r1 as u64) << 32)),
                    vcreate_u32((wr3 as u64) | ((r0 as u64) << 32)),
                    vcreate_u32((wr2 as u64) | ((wr3 as u64) << 32)),
                    vcreate_u32((wr1 as u64) | ((wr2 as u64) << 32)),
                ];
                let cols_hi = [
                    vcreate_u32((r2 as u64) | ((r3 as u64) << 32)),
                    vcreate_u32((r1 as u64) | ((r2 as u64) << 32)),
                    vcreate_u32((r0 as u64) | ((r1 as u64) << 32)),
                    vcreate_u32((wr3 as u64) | ((r0 as u64) << 32)),
                ];
                Self { cols_lo, cols_hi }
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self { r }
        }
    }

    #[inline(always)]
    fn apply(&self, x: BB4) -> BB4 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                let x_raw: uint32x4_t = vld1q_u32(&x as *const BB4 as *const u32);
                let x_lo = vget_low_u32(x_raw);
                let x_hi = vget_high_u32(x_raw);
                let mut lo = vmull_lane_u32::<0>(self.cols_lo[0], x_lo);
                lo = vmlal_lane_u32::<1>(lo, self.cols_lo[1], x_lo);
                lo = vmlal_lane_u32::<0>(lo, self.cols_lo[2], x_hi);
                lo = vmlal_lane_u32::<1>(lo, self.cols_lo[3], x_hi);
                let mut hi = vmull_lane_u32::<0>(self.cols_hi[0], x_lo);
                hi = vmlal_lane_u32::<1>(hi, self.cols_hi[1], x_lo);
                hi = vmlal_lane_u32::<0>(hi, self.cols_hi[2], x_hi);
                hi = vmlal_lane_u32::<1>(hi, self.cols_hi[3], x_hi);
                std::mem::transmute(bb_neon_monty_reduce(lo, hi))
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.r * x
        }
    }
}

pub(crate) struct BB5MulByConst {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    r: BB5,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    rhs_lo: [uint32x2_t; 5],
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    rhs_hi: [uint32x2_t; 5],
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    tail: [BabyBear; 5],
}

impl BB5MulByConst {
    #[inline(always)]
    pub(crate) fn new(r: BB5) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let coeffs = bb5_to_coeffs(r);
            let w = BabyBear::from_u32(BB_W5 as u32);
            let wr1: u32 = unsafe { std::mem::transmute(w * coeffs[1]) };
            let wr2: u32 = unsafe { std::mem::transmute(w * coeffs[2]) };
            let wr3: u32 = unsafe { std::mem::transmute(w * coeffs[3]) };
            let wr4: u32 = unsafe { std::mem::transmute(w * coeffs[4]) };
            let [r0, r1, r2, r3, _] = bb5_to_limbs(r);
            unsafe {
                let rhs_lo = [
                    vcreate_u32((r0 as u64) | ((r1 as u64) << 32)),
                    vcreate_u32((wr4 as u64) | ((r0 as u64) << 32)),
                    vcreate_u32((wr3 as u64) | ((wr4 as u64) << 32)),
                    vcreate_u32((wr2 as u64) | ((wr3 as u64) << 32)),
                    vcreate_u32((wr1 as u64) | ((wr2 as u64) << 32)),
                ];
                let rhs_hi = [
                    vcreate_u32((r2 as u64) | ((r3 as u64) << 32)),
                    vcreate_u32((r1 as u64) | ((r2 as u64) << 32)),
                    vcreate_u32((r0 as u64) | ((r1 as u64) << 32)),
                    vcreate_u32((wr4 as u64) | ((r0 as u64) << 32)),
                    vcreate_u32((wr3 as u64) | ((wr4 as u64) << 32)),
                ];
                let tail = [coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]];
                Self {
                    rhs_lo,
                    rhs_hi,
                    tail,
                }
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self { r }
        }
    }

    #[inline(always)]
    pub(crate) fn apply(&self, x: BB5) -> BB5 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                let x_raw: uint32x4_t = vld1q_u32(&x as *const BB5 as *const u32);
                let x_lo = vget_low_u32(x_raw);
                let x_hi = vget_high_u32(x_raw);
                let x4 = vdup_n_u32(*(&x as *const BB5 as *const u32).add(4));
                let mut lo = vmull_lane_u32::<0>(self.rhs_lo[0], x_lo);
                lo = vmlal_lane_u32::<1>(lo, self.rhs_lo[1], x_lo);
                lo = vmlal_lane_u32::<0>(lo, self.rhs_lo[2], x_hi);
                lo = vmlal_lane_u32::<1>(lo, self.rhs_lo[3], x_hi);
                lo = vmlal_lane_u32::<0>(lo, self.rhs_lo[4], x4);
                let mut hi = vmull_lane_u32::<0>(self.rhs_hi[0], x_lo);
                hi = vmlal_lane_u32::<1>(hi, self.rhs_hi[1], x_lo);
                hi = vmlal_lane_u32::<0>(hi, self.rhs_hi[2], x_hi);
                hi = vmlal_lane_u32::<1>(hi, self.rhs_hi[3], x_hi);
                hi = vmlal_lane_u32::<0>(hi, self.rhs_hi[4], x4);
                let dot: [u32; 4] = std::mem::transmute(bb_neon_monty_reduce(lo, hi));
                let x_coeffs = bb5_to_coeffs(x);
                let tail: u32 = std::mem::transmute(BabyBear::dot_product::<5>(&x_coeffs, &self.tail));
                bb5_from_limbs([dot[0], dot[1], dot[2], dot[3], tail])
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.r * x
        }
    }
}

const KB_PRIME: u64 = 0x7f000001;
const KB_MONTY_MU: u64 = 0x81000001;
const KB_MONTY_INV: u64 = KB_PRIME - ((KB_PRIME.wrapping_mul(KB_MONTY_MU) - 1) >> 32);

#[inline(always)]
fn kb_monty_reduce_i128(x: i128) -> u32 {
    let x = x.rem_euclid(KB_PRIME as i128) as u128;
    ((x.wrapping_mul(KB_MONTY_INV as u128)) % (KB_PRIME as u128)) as u32
}

#[inline(always)]
fn kb5_to_limbs(x: KB5) -> [u32; 5] {
    const _: () = assert!(std::mem::size_of::<KB5>() == std::mem::size_of::<[u32; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn kb5_to_coeffs(x: KB5) -> [KoalaBear; 5] {
    const _: () = assert!(std::mem::size_of::<KB5>() == std::mem::size_of::<[KoalaBear; 5]>());
    unsafe { std::mem::transmute(x) }
}

#[inline(always)]
fn kb5_from_limbs(limbs: [u32; 5]) -> KB5 {
    unsafe { std::mem::transmute(limbs) }
}

struct KB5Accum {
    c: [u128; 9],
}

impl KB5Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self { c: [0u128; 9] }
    }

    #[inline(always)]
    fn fmadd(&mut self, a: KB5, b: KB5) {
        let a = kb5_to_limbs(a);
        let b = kb5_to_limbs(b);
        let (a0, a1, a2, a3, a4) = (a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64, a[4] as u64);
        let (b0, b1, b2, b3, b4) = (b[0] as u64, b[1] as u64, b[2] as u64, b[3] as u64, b[4] as u64);

        self.c[0] += (a0 * b0) as u128;
        self.c[1] += (a0 * b1 + a1 * b0) as u128;
        self.c[2] += (a0 * b2 + a1 * b1 + a2 * b0) as u128;
        self.c[3] += (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) as u128;
        self.c[4] += (a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1) as u128;
        self.c[4] += (a4 * b0) as u128;
        self.c[5] += (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1) as u128;
        self.c[6] += (a2 * b4 + a3 * b3 + a4 * b2) as u128;
        self.c[7] += (a3 * b4 + a4 * b3) as u128;
        self.c[8] += (a4 * b4) as u128;
    }

    fn reduce(self) -> KB5 {
        let d0 = self.c[0] as i128 + self.c[5] as i128 - self.c[8] as i128;
        let d1 = self.c[1] as i128 + self.c[6] as i128;
        let d2 = self.c[2] as i128 - self.c[5] as i128 + self.c[7] as i128 + self.c[8] as i128;
        let d3 = self.c[3] as i128 - self.c[6] as i128 + self.c[8] as i128;
        let d4 = self.c[4] as i128 - self.c[7] as i128;
        kb5_from_limbs([
            kb_monty_reduce_i128(d0),
            kb_monty_reduce_i128(d1),
            kb_monty_reduce_i128(d2),
            kb_monty_reduce_i128(d3),
            kb_monty_reduce_i128(d4),
        ])
    }
}

struct KB5MulByConst {
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    r: KB5,
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    rhs: [PackedKoalaBearNeon; 5],
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    tail: [KoalaBear; 5],
}

impl KB5MulByConst {
    #[inline(always)]
    fn new(r: KB5) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let b = kb5_to_coeffs(r);
            let b0_minus_b3 = b[0] - b[3];
            let b1_minus_b4 = b[1] - b[4];
            let b4_minus_b2 = b[4] - b[2];
            let b3_plus_b4_minus_b1 = b[3] - b1_minus_b4;
            let rhs = [
                unsafe { std::mem::transmute([b[0], b[1], b[2], b[3]]) },
                unsafe { std::mem::transmute([b[4], b[0], b1_minus_b4, b[2]]) },
                unsafe { std::mem::transmute([b[3], b[4], b0_minus_b3, b1_minus_b4]) },
                unsafe { std::mem::transmute([b[2], b[3], b4_minus_b2, b0_minus_b3]) },
                unsafe { std::mem::transmute([b1_minus_b4, b[2], b3_plus_b4_minus_b1, b4_minus_b2]) },
            ];
            let tail = [b[4], b[3], b[2], b1_minus_b4, b0_minus_b3];
            Self { rhs, tail }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self { r }
        }
    }

    #[inline(always)]
    fn apply(&self, x: KB5) -> KB5 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let x_coeffs = kb5_to_coeffs(x);
            let lhs = x_coeffs.map(Into::<PackedKoalaBearNeon>::into);
            let dot = PackedKoalaBearNeon::dot_product(&lhs, &self.rhs).0;
            KB5::from_basis_coefficients_fn(|i| {
                if i < 4 {
                    dot[i]
                } else {
                    KoalaBear::dot_product::<5>(&x_coeffs, &self.tail)
                }
            })
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.r * x
        }
    }
}

macro_rules! delayed_sumcheck_fns {
    ($field:ty, $accum:ty, $mul_by_const:ty, $suffix:ident) => {
        paste::paste! {
            pub fn [<sumcheck_deg2_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                challenges: &[$field],
            ) {
                for round in 0..challenges.len() {
                    let half = f.len() / 2;
                    let mut h0 = <$accum>::zero();
                    let mut h1 = <$accum>::zero();
                    let mut h_inf = <$accum>::zero();
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let f1 = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let g1 = g[2 * j + 1];
                        let df = f1 - f0;
                        let dg = g1 - g0;
                        h0.fmadd(f0, g0);
                        h1.fmadd(f1, g1);
                        h_inf.fmadd(df, dg);
                    }
                    let _ = black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
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

            #[allow(dead_code)]
            pub fn [<sumcheck_deg2_eq_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                suffix_eq: &[Vec<$field>],
                challenges: &[$field],
            ) {
                let n = challenges.len();
                for round in 0..n {
                    let half = f.len() / 2;
                    let eq_rest = &suffix_eq[round + 1];
                    let mut q1 = <$accum>::zero();
                    let mut q_inf = <$accum>::zero();
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
                    let _ = black_box((q1.reduce(), q_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
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

            pub fn [<sumcheck_deg2_projective_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                challenges: &[$field],
            ) {
                for round in 0..challenges.len() {
                    let half = f.len() / 2;
                    let mut h0 = <$accum>::zero();
                    let mut h1 = <$accum>::zero();
                    let mut h_inf = <$accum>::zero();
                    for j in 0..half {
                        let f0 = f[2 * j];
                        let fi = f[2 * j + 1];
                        let g0 = g[2 * j];
                        let gi = g[2 * j + 1];
                        h0.fmadd(f0, g0);
                        h_inf.fmadd(fi, gi);
                        let sf = f0 + fi;
                        let sg = g0 + gi;
                        h1.fmadd(sf, sg);
                    }
                    let _ = black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
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

            #[allow(dead_code)]
            pub fn [<sumcheck_deg2_eq_projective_delayed_ $suffix>](
                f: &mut Vec<$field>,
                g: &mut Vec<$field>,
                suffix_eq: &[Vec<$field>],
                challenges: &[$field],
            ) {
                let n = challenges.len();
                for round in 0..n {
                    let half = f.len() / 2;
                    let eq_rest = &suffix_eq[round + 1];
                    let mut q1 = <$accum>::zero();
                    let mut q_inf = <$accum>::zero();
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
                    let _ = black_box((q1.reduce(), q_inf.reduce()));
                    let r = challenges[round];
                    let mul_r = <$mul_by_const>::new(r);
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
        }
    };
}

delayed_sumcheck_fns!(BB4, BB4Accum, BB4MulByConst, bb4);
delayed_sumcheck_fns!(BB5, BB5Accum, BB5MulByConst, bb5);
delayed_sumcheck_fns!(KB5, KB5Accum, KB5MulByConst, kb5);
