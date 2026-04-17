use super::*;

pub fn sumcheck_deg2_projective_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
    zero: Fp128,
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = zero;
        let mut h1 = zero;
        let mut h_inf = zero;

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            h0 += f0 * g0;
            h_inf += fi * gi;
            let sf = f0 + fi;
            let sg = g0 + gi;
            h1 += sf * sg;
        }

        black_box((h0, h1, h_inf));

        let r = challenges[round];

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_1inf_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
    zero: Fp128,
) {
    let one = Fp128::one();
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = zero;
        let mut h1 = zero;
        let mut h_inf = zero;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let f0 = f1 - fi;
            let g0 = g1 - gi;

            h0 += f0 * g0;
            h1 += f1 * g1;
            h_inf += fi * gi;
        }

        black_box((h0, h1, h_inf));

        let r = challenges[round];
        let r_minus_one = r - one;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

#[inline(always)]
fn sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(
    f: &[Fp128],
    g: &[Fp128],
    eq_rest: &[Fp128],
    zero: Fp128,
) -> (Fp128, Fp128) {
    let half = f.len() / 2;
    let mut q0 = zero;
    let mut q1 = zero;

    for j in 0..half {
        let f1 = f[2 * j];
        let fi = f[2 * j + 1];
        let g1 = g[2 * j];
        let gi = g[2 * j + 1];
        let ew = eq_rest[j];
        let f0 = f1 - fi;
        let g0 = g1 - gi;

        q0 += f0 * g0 * ew;
        q1 += f1 * g1 * ew;
    }

    (q0, q1)
}

pub fn init_sumcheck_deg2_eq_gruen_projective_1inf_fp128_claim(
    f: &[Fp128],
    g: &[Fp128],
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    zero: Fp128,
) -> Fp128 {
    let (q0, q1) =
        sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, &suffix_eq[1], zero);
    let w = eq_point[0];
    let one_minus_w = Fp128::one() - w;
    q1.mul_add(w, one_minus_w * q0)
}

pub fn sumcheck_deg2_eq_gruen_projective_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
    zero: Fp128,
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = zero;
        let mut q_inf = zero;

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

        black_box((q1, q_inf));

        let r = challenges[round];

        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_gruen_projective_1inf_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    challenges: &[Fp128],
    initial_claim: Fp128,
    zero: Fp128,
) {
    let one = Fp128::one();
    let mut claim = initial_claim;
    let mut current_scalar = one;
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = zero;
        let mut q_inf = zero;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q1 += f1 * g1 * ew;
            q_inf += fi * gi * ew;
        }

        let w = eq_point[round];
        let one_minus_w = one - w;
        let q0 = if one_minus_w == zero || current_scalar == zero {
            sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, eq_rest, zero).0
        } else {
            let normalized_claim = claim * current_scalar.inv_or_zero();
            (normalized_claim - w * q1) * one_minus_w.inv_or_zero()
        };

        black_box((q0, q1, q_inf));

        let r = challenges[round];
        let r_minus_one = r - one;
        let r_times_r_minus_one = r * r_minus_one;
        let q_r = q_inf.mul_add(r_times_r_minus_one, (q1 - q0).mul_add(r, q0));
        let eq_eval = (w - one_minus_w).mul_add(r, one_minus_w);

        current_scalar = current_scalar * eq_eval;
        claim = current_scalar * q_r;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];

            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }

        f.truncate(half);
        g.truncate(half);
    }
}

/// Compute `a + (b - a) * r` for the fused bind+reduce loop.
///
/// Currently just forwards to [`Fp128::mul_add`]. A pure-Rust
/// `mul_wide + solinas_reduce` variant was tried (the "delayed
/// reduction" shape that [`Fp128Accum::fmadd`] uses for the reduce
/// side) to give LLVM's scheduler freedom to interleave widening
/// muls across iterations. It measured 5 - 15 % slower than the
/// `mul_add_raw_aarch64` asm block because the asm block saves
/// ~17 instructions via fused carry chains and the `ccmp`
/// canonicalize trick, which outweighs the scheduling flexibility.
/// Kept as a helper so any future field backend can override the
/// bind shape (e.g. SIMD Fp128 via NEON would want to diverge from
/// the scalar `mul_add`).
///
/// See `docs/notes/fused-bind-eval-ab.md` for the A/B analysis.
#[inline(always)]
pub(super) fn fp128_bind(a: Fp128, b: Fp128, r: Fp128) -> Fp128 {
    (b - a).mul_add(r, a)
}

// -----------------------------------------------------------------------------
// Relaxed / "delayed reduction" experiment (AArch64, Solinas p = 2^128 - 275)
// -----------------------------------------------------------------------------
//
// Hypothesis: drop the ≥p canonicalize tail on mul_add's output. The next
// round's ops must accept non-canonical inputs in [0, 2^128). For Solinas near
// 2^128 the non-canonical range is only [p, 2^128) (size c ≈ 2^8), but hachi's
// Sub silently mis-reduces there. So a matching "relaxed sub" is needed that
// iterates the -c correction until the 128-bit result is settled.
//
// Bookkeeping per fused iteration (ignoring shared loads):
//   4 * mul_add  -3 inst each  =  -12
//   6 * sub      +2 inst each  =  +12   (4 bind subs + 2 h_inf subs)
//
// Net zero at the algebra level. We still run it empirically to ground-truth
// that prediction: scheduler / register-pressure side-effects might push it a
// few % either way.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn sub_raw_aarch64_relaxed(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    use std::arch::asm;
    let out_lo: u64;
    let out_hi: u64;
    asm!(
        // Standard subs/sbcs then "+p if borrow" is hachi's Sub. It's correct
        // when both inputs are in [0, p) but off by c when an input lives in
        // the non-canonical sliver [p, 2^128) AND the other is small enough
        // that the post-correction low wraps below c. Iterating the "-c if
        // borrow" (`p = 2^128 - c`, so adding p ≡ subtracting c mod 2^128)
        // a second time covers the wrap case and leaves the result in
        // [0, 2^128), possibly still ≥ p (that's fine, downstream accepts it).
        "subs {out_lo}, {a_lo}, {b_lo}",
        "sbcs {out_hi}, {a_hi}, {b_hi}",
        "csel {c_tmp}, xzr, {c}, hs",
        "subs {out_lo}, {out_lo}, {c_tmp}",
        "sbcs {out_hi}, {out_hi}, xzr",
        "csel {c_tmp}, xzr, {c}, hs",
        "subs {out_lo}, {out_lo}, {c_tmp}",
        "sbc  {out_hi}, {out_hi}, xzr",
        c = in(reg) Fp128::C_LO,
        a_lo = in(reg) a[0],
        a_hi = in(reg) a[1],
        b_lo = in(reg) b[0],
        b_hi = in(reg) b[1],
        c_tmp = out(reg) _,
        out_lo = out(reg) out_lo,
        out_hi = out(reg) out_hi,
        options(pure, nomem, nostack),
    );
    [out_lo, out_hi]
}

/// Non-canonicalizing multiply-add. Output lives in `[0, 2^128)`, possibly
/// non-canonical by at most `c` (i.e. in `[p, 2^128)`). All downstream ops on
/// the relaxed-bind path must accept such inputs.
///
/// Saves 3 instructions vs [`Fp128::mul_add`]'s 35-inst asm: drops the
/// `ccmp`-based ≥p selector and folds the single fold-2 overflow bit via a
/// direct `csel / adds / adc` chain.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn mul_add_raw_aarch64_relaxed(
    a: [u64; 2],
    b: [u64; 2],
    addend: [u64; 2],
) -> [u64; 2] {
    use std::arch::asm;
    let out_lo: u64;
    let out_hi: u64;
    asm!(
        // Schoolbook 2×2 → 256-bit product (identical to canonical).
        "mul     {p00l}, {a0}, {b0}",
        "umulh   {p00h}, {a0}, {b0}",
        "mul     {p01l}, {a0}, {b1}",
        "umulh   {p01h}, {a0}, {b1}",
        "mul     {p10l}, {a1}, {b0}",
        "umulh   {p10h}, {a1}, {b0}",
        "mul     {p11l}, {a1}, {b1}",
        "umulh   {p11h}, {a1}, {b1}",
        // Carry accumulation.
        "adds   {p00h}, {p00h}, {p01l}",
        "cset   {p01l:w}, hs",
        "adds   {p01h}, {p01h}, {p10h}",
        "cset   {p10h:w}, hs",
        "adds   {p01h}, {p01h}, {p11l}",
        "cinc   {p10h}, {p10h}, hs",
        "adds   {p00h}, {p00h}, {p10l}",
        "adcs   {p01h}, {p01h}, {p01l}",
        "adc    {p11h}, {p11h}, {p10h}",
        // Fuse addend.
        "adds   {p00l}, {p00l}, {add_lo}",
        "adcs   {p00h}, {p00h}, {add_hi}",
        "adcs   {p01h}, {p01h}, xzr",
        "adc    {p11h}, {p11h}, xzr",
        // Fold-1: [lo] += C * [hi].
        "mul    {p01l}, {p01h}, {c}",
        "umulh  {p10l}, {p01h}, {c}",
        "mul    {p10h}, {p11h}, {c}",
        "umulh  {p11l}, {p11h}, {c}",
        "adds   {p00l}, {p00l}, {p01l}",
        "adcs   {p00h}, {p00h}, {p10l}",
        "cset   {p01h:w}, hs",
        "adds   {p00h}, {p00h}, {p10h}",
        "adc    {p11h}, {p11l}, {p01h}",
        // Fold-2 RELAXED: drop the ≥p `ccmp` canonicalization; keep only the
        // 128-bit overflow fold. p11h is 0 or 1 here, so c·p11h < 2^32 and the
        // single csel/adds/adc chain cannot produce a second-order overflow.
        "mul    {p01l}, {p11h}, {c}",
        "adds   {p00l}, {p00l}, {p01l}",
        "adcs   {p00h}, {p00h}, xzr",
        "csel   {p01l}, {c}, xzr, hs",
        "adds   {out_lo}, {p00l}, {p01l}",
        "adc    {out_hi}, {p00h}, xzr",
        a0 = in(reg) a[0],
        a1 = in(reg) a[1],
        b0 = in(reg) b[0],
        b1 = in(reg) b[1],
        add_lo = in(reg) addend[0],
        add_hi = in(reg) addend[1],
        c = in(reg) Fp128::C_LO,
        p00l = out(reg) _,
        p00h = out(reg) _,
        p01l = out(reg) _,
        p01h = out(reg) _,
        p10l = out(reg) _,
        p10h = out(reg) _,
        p11l = out(reg) _,
        p11h = out(reg) _,
        out_lo = lateout(reg) out_lo,
        out_hi = lateout(reg) out_hi,
        options(pure, nomem, nostack),
    );
    [out_lo, out_hi]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) fn fp128_bind_relaxed(a: [u64; 2], b: [u64; 2], r: [u64; 2]) -> [u64; 2] {
    unsafe {
        let diff = sub_raw_aarch64_relaxed(b, a);
        mul_add_raw_aarch64_relaxed(diff, r, a)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(super) fn fp128_bind_relaxed(a: [u64; 2], b: [u64; 2], r: [u64; 2]) -> [u64; 2] {
    let a_fp = Fp128::from_canonical_u128(to_u128_u64_pair(a));
    let b_fp = Fp128::from_canonical_u128(to_u128_u64_pair(b));
    let r_fp = Fp128::from_canonical_u128(to_u128_u64_pair(r));
    fp128_bind(a_fp, b_fp, r_fp).to_limbs()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn to_u128_u64_pair(limbs: [u64; 2]) -> u128 {
    (limbs[0] as u128) | ((limbs[1] as u128) << 64)
}

/// Canonicalize a relaxed Fp128 in `[0, 2^128)` to its member in `[0, p)`.
#[inline(always)]
pub(super) fn canonicalize_relaxed(limbs: [u64; 2]) -> Fp128 {
    let v = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
    Fp128::from_canonical_u128_reduced(v)
}

pub(super) struct Fp128Accum([u128; 4]);

impl Fp128Accum {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        Self([0u128; 4])
    }

    #[inline(always)]
    pub(super) fn fmadd(&mut self, a: Fp128, b: Fp128) {
        let product = a.mul_wide(b);
        for i in 0..4 {
            self.0[i] += product[i] as u128;
        }
    }

    pub(super) fn reduce(self) -> Fp128 {
        let mut limbs = [0u64; 5];
        let mut carry: u128 = 0;
        for i in 0..4 {
            let sum = self.0[i] + carry;
            limbs[i] = sum as u64;
            carry = sum >> 64;
        }
        limbs[4] = carry as u64;
        Fp128::solinas_reduce(&limbs)
    }
}

pub fn sumcheck_deg2_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

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

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            f[j] = (f1 - f0).mul_add(r, f0);
            g[j] = (g1 - g0).mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

/// Same sum-check relation as [`sumcheck_deg2_delayed_fp128`], with
/// bind-of-previous-round and reduce-of-current-round fused into one
/// pass. The classic jolt-cpp GPU shape (see
/// `bind_eval_roundN_kernel`): each fused iteration reads 4
/// round-(r-1) elements per output pair, binds them with
/// `challenges[r - 1]` to produce 2 round-r elements, writes them in
/// place, and accumulates the round-r eval partial from the
/// just-bound pair before moving on. Round 0 stays a pure reduce
/// pass (no prior challenge), and the last challenge binds without a
/// following reduce.
///
/// Saves one read per round-r pair (the unfused path would re-read
/// the freshly bound round-r state during its separate reduce pass).
/// See `docs/plans/sumcheck-cpu-platform.md` Phase 1 for the derivation.
pub fn sumcheck_deg2_delayed_fp128_fused(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let n_rounds = challenges.len();
    if n_rounds == 0 {
        return;
    }

    {
        let half = f.len() / 2;
        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(f1 - f0, g1 - g0);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    for r_idx in 1..n_rounds {
        let r_prev = challenges[r_idx - 1];
        let new_half = f.len() / 4;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..new_half {
            let f00 = f[4 * j];
            let f01 = f[4 * j + 1];
            let f10 = f[4 * j + 2];
            let f11 = f[4 * j + 3];
            let g00 = g[4 * j];
            let g01 = g[4 * j + 1];
            let g10 = g[4 * j + 2];
            let g11 = g[4 * j + 3];

            let f0 = fp128_bind(f00, f01, r_prev);
            let f1 = fp128_bind(f10, f11, r_prev);
            let g0 = fp128_bind(g00, g01, r_prev);
            let g1 = fp128_bind(g10, g11, r_prev);

            // In-place write is safe: iteration j reads positions
            // [4j, 4j+4) and writes [2j, 2j+2); 4j > 2j+1 for j >= 1,
            // and for j = 0 both reads complete before either write.
            f[2 * j] = f0;
            f[2 * j + 1] = f1;
            g[2 * j] = g0;
            g[2 * j + 1] = g1;

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(f1 - f0, g1 - g0);
        }

        f.truncate(2 * new_half);
        g.truncate(2 * new_half);

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    let r_last = challenges[n_rounds - 1];
    let half = f.len() / 2;
    for j in 0..half {
        f[j] = fp128_bind(f[2 * j], f[2 * j + 1], r_last);
        g[j] = fp128_bind(g[2 * j], g[2 * j + 1], r_last);
    }
    f.truncate(half);
    g.truncate(half);
}

/// Variant of [`sumcheck_deg2_delayed_fp128_fused`] that uses the relaxed
/// `mul_add` (no ≥p canonicalize) and a matching relaxed `sub` (iterates the
/// -c correction so non-canonical inputs are handled correctly). Intermediates
/// live in `[0, 2^128)` instead of `[0, p)`; the last round canonicalizes.
///
/// See `docs/notes/fused-bind-eval-ab.md` § "Delayed reduction on Fp128 Solinas"
/// for the algebra. This kernel exists to ground-truth the claim that
/// delayed reduction is neutral for Solinas near 2^128 (3 inst saved per
/// mul_add, 2 inst paid per sub; 4 binds + 2 h_inf subs per iter → net 0).
///
/// Storage: we keep the buffers as `Vec<Fp128>` for API compatibility, but
/// reinterpret the backing memory as `[u64; 2]` chunks via pointer casts
/// (valid because `Fp128` is a `#[repr(Rust)]` single-field tuple struct over
/// `[u64; 2]`, which is layout-compatible with `[u64; 2]`). A non-canonical
/// `[u64; 2]` stored in that slot is semantically invalid *as an* `Fp128`
/// (violates the documented `< p` invariant), but is fine as an internal
/// transient value: we canonicalize the whole buffer at the last round.
#[cfg(target_arch = "aarch64")]
pub fn sumcheck_deg2_delayed_fp128_fused_relaxed(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let n_rounds = challenges.len();
    if n_rounds == 0 {
        return;
    }

    #[inline(always)]
    fn fmadd_accum_relaxed(acc: &mut Fp128Accum, a: [u64; 2], b: [u64; 2]) {
        // `Fp128Accum::fmadd` calls `mul_wide`, which is a pure-Rust 2x2
        // schoolbook on the limb pair. It's correct for any `[u64; 2]` input
        // (no reduction dependence). Feed raw limbs through a transmute to
        // avoid the `from_canonical_u128` debug-assert.
        let a_fp: Fp128 = unsafe { std::mem::transmute(a) };
        let b_fp: Fp128 = unsafe { std::mem::transmute(b) };
        acc.fmadd(a_fp, b_fp);
    }

    // Helper: reinterpret the Vec<Fp128> backing buffer as *mut [u64; 2].
    // Safety: Fp128 is a single-field tuple struct over [u64; 2] with default
    // layout (size 16, align 8); no other fields, no padding. Layout matches.
    let f_ptr = f.as_mut_ptr() as *mut [u64; 2];
    let g_ptr = g.as_mut_ptr() as *mut [u64; 2];

    // Round 0: pure reduce (no prior challenge). Inputs are canonical.
    {
        let half = f.len() / 2;
        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(f1 - f0, g1 - g0);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    for r_idx in 1..n_rounds {
        let r_prev: [u64; 2] = challenges[r_idx - 1].to_limbs();
        let new_half = f.len() / 4;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..new_half {
            unsafe {
                let f00 = *f_ptr.add(4 * j);
                let f01 = *f_ptr.add(4 * j + 1);
                let f10 = *f_ptr.add(4 * j + 2);
                let f11 = *f_ptr.add(4 * j + 3);
                let g00 = *g_ptr.add(4 * j);
                let g01 = *g_ptr.add(4 * j + 1);
                let g10 = *g_ptr.add(4 * j + 2);
                let g11 = *g_ptr.add(4 * j + 3);

                let f0 = fp128_bind_relaxed(f00, f01, r_prev);
                let f1 = fp128_bind_relaxed(f10, f11, r_prev);
                let g0 = fp128_bind_relaxed(g00, g01, r_prev);
                let g1 = fp128_bind_relaxed(g10, g11, r_prev);

                // Store relaxed limbs back. The Vec<Fp128> now holds
                // non-canonical values in slots [0..2*new_half); next round
                // reads them as [u64; 2] via the same pointer cast.
                *f_ptr.add(2 * j) = f0;
                *f_ptr.add(2 * j + 1) = f1;
                *g_ptr.add(2 * j) = g0;
                *g_ptr.add(2 * j + 1) = g1;

                let df = sub_raw_aarch64_relaxed(f1, f0);
                let dg = sub_raw_aarch64_relaxed(g1, g0);

                fmadd_accum_relaxed(&mut h0, f0, g0);
                fmadd_accum_relaxed(&mut h1, f1, g1);
                fmadd_accum_relaxed(&mut h_inf, df, dg);
            }
        }

        f.truncate(2 * new_half);
        g.truncate(2 * new_half);

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));
    }

    // Last round: inputs in f/g are relaxed; canonicalize before handing
    // back to the canonical bind. Equivalent cost-wise to writing a "final
    // relaxed bind" + canonicalize; we just do it through standard ops.
    let r_last = challenges[n_rounds - 1];
    for v in f.iter_mut() {
        *v = canonicalize_relaxed(v.to_limbs());
    }
    for v in g.iter_mut() {
        *v = canonicalize_relaxed(v.to_limbs());
    }

    let half = f.len() / 2;
    for j in 0..half {
        f[j] = fp128_bind(f[2 * j], f[2 * j + 1], r_last);
        g[j] = fp128_bind(g[2 * j], g[2 * j + 1], r_last);
    }
    f.truncate(half);
    g.truncate(half);
}

pub fn sumcheck_deg2_eq_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

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

        black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let g0 = g[2 * j];
            let g1 = g[2 * j + 1];
            f[j] = (f1 - f0).mul_add(r, f0);
            g[j] = (g1 - g0).mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

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

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_projective_1inf_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    challenges: &[Fp128],
) {
    let one = Fp128::one();
    for round in 0..challenges.len() {
        let half = f.len() / 2;

        let mut h0 = Fp128Accum::zero();
        let mut h1 = Fp128Accum::zero();
        let mut h_inf = Fp128Accum::zero();

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let f0 = f1 - fi;
            let g0 = g1 - gi;

            h0.fmadd(f0, g0);
            h1.fmadd(f1, g1);
            h_inf.fmadd(fi, gi);
        }

        black_box((h0.reduce(), h1.reduce(), h_inf.reduce()));

        let r = challenges[round];
        let r_minus_one = r - one;
        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    challenges: &[Fp128],
) {
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

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

        black_box((q1.reduce(), q_inf.reduce()));

        let r = challenges[round];
        for j in 0..half {
            let f0 = f[2 * j];
            let fi = f[2 * j + 1];
            let g0 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r, f0);
            g[j] = gi.mul_add(r, g0);
        }
        f.truncate(half);
        g.truncate(half);
    }
}

pub fn sumcheck_deg2_eq_projective_1inf_delayed_fp128(
    f: &mut Vec<Fp128>,
    g: &mut Vec<Fp128>,
    suffix_eq: &[Vec<Fp128>],
    eq_point: &[Fp128],
    challenges: &[Fp128],
    initial_claim: Fp128,
) {
    let one = Fp128::one();
    let mut claim = initial_claim;
    let mut current_scalar = one;
    let n = challenges.len();
    for round in 0..n {
        let half = f.len() / 2;
        let eq_rest = &suffix_eq[round + 1];

        let mut q1 = Fp128Accum::zero();
        let mut q_inf = Fp128Accum::zero();

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            let ew = eq_rest[j];

            q1.fmadd(f1 * g1, ew);
            q_inf.fmadd(fi * gi, ew);
        }

        let q1 = q1.reduce();
        let q_inf = q_inf.reduce();
        let w = eq_point[round];
        let one_minus_w = one - w;
        let q0 = if one_minus_w == Fp128::ZERO || current_scalar == Fp128::ZERO {
            sumcheck_deg2_eq_gruen_projective_1inf_q0_q1_fp128(f, g, eq_rest, Fp128::ZERO).0
        } else {
            let normalized_claim = claim * current_scalar.inv_or_zero();
            (normalized_claim - w * q1) * one_minus_w.inv_or_zero()
        };

        black_box((q0, q1, q_inf));

        let r = challenges[round];
        let r_minus_one = r - one;
        let r_times_r_minus_one = r * r_minus_one;
        let q_r = q_inf.mul_add(r_times_r_minus_one, (q1 - q0).mul_add(r, q0));
        let eq_eval = (w - one_minus_w).mul_add(r, one_minus_w);

        current_scalar = current_scalar * eq_eval;
        claim = current_scalar * q_r;

        for j in 0..half {
            let f1 = f[2 * j];
            let fi = f[2 * j + 1];
            let g1 = g[2 * j];
            let gi = g[2 * j + 1];
            f[j] = fi.mul_add(r_minus_one, f1);
            g[j] = gi.mul_add(r_minus_one, g1);
        }
        f.truncate(half);
        g.truncate(half);
    }
}
