//! Bit-identity tests for the sequential fused bind+reduce kernels
//! against the sequential unfused baseline, for both GF128 and Fp128.
//!
//! The fused kernel is expected to produce **bit-identical** `(f, g)`
//! after all rounds: the arithmetic is associative+commutative per
//! pair and the fused path visits pairs in the same order as the
//! unfused path, so no reduction-order tolerance is needed.
//!
//! Covers `n = 1..=4` (to exercise the round-0-only and
//! round-0 + final-bind edge cases), `n = 8, 12` (typical sizes),
//! and `n = 14, 16` (closer to production, gated on release for
//! reasonable runtime). No `parallel` feature required; pure
//! sequential code under test.

use monomial_sumcheck_benchmarks::sumcheck::*;

fn check_gf128_n(n: usize) {
    let f = make_gf128(1 << n);
    let g = make_gf128(1 << n);
    let challenges = make_gf128(n);

    let mut f_ref = f.clone();
    let mut g_ref = g.clone();
    sumcheck_deg2_delayed_gf128(&mut f_ref, &mut g_ref, &challenges);

    let mut f_fused = f.clone();
    let mut g_fused = g.clone();
    sumcheck_deg2_delayed_gf128_fused(&mut f_fused, &mut g_fused, &challenges);

    assert_eq!(f_fused.len(), f_ref.len(), "GF128 fused f.len mismatch at n={n}");
    assert_eq!(g_fused.len(), g_ref.len(), "GF128 fused g.len mismatch at n={n}");
    assert_eq!(f_fused, f_ref, "GF128 fused f mismatch at n={n}");
    assert_eq!(g_fused, g_ref, "GF128 fused g mismatch at n={n}");
}

fn check_fp128_n(n: usize) {
    let f = make_fp128(1 << n);
    let g = make_fp128(1 << n);
    let challenges = make_fp128(n);

    let mut f_ref = f.clone();
    let mut g_ref = g.clone();
    sumcheck_deg2_delayed_fp128(&mut f_ref, &mut g_ref, &challenges);

    let mut f_fused = f.clone();
    let mut g_fused = g.clone();
    sumcheck_deg2_delayed_fp128_fused(&mut f_fused, &mut g_fused, &challenges);

    assert_eq!(f_fused.len(), f_ref.len(), "Fp128 fused f.len mismatch at n={n}");
    assert_eq!(g_fused.len(), g_ref.len(), "Fp128 fused g.len mismatch at n={n}");
    assert_eq!(f_fused, f_ref, "Fp128 fused f mismatch at n={n}");
    assert_eq!(g_fused, g_ref, "Fp128 fused g mismatch at n={n}");
}

#[test]
fn gf128_fused_edge_cases() {
    for n in 1..=4 {
        check_gf128_n(n);
    }
}

#[test]
fn gf128_fused_medium() {
    check_gf128_n(8);
    check_gf128_n(12);
}

#[test]
fn gf128_fused_large() {
    check_gf128_n(14);
    check_gf128_n(16);
}

#[test]
fn fp128_fused_edge_cases() {
    for n in 1..=4 {
        check_fp128_n(n);
    }
}

#[test]
fn fp128_fused_medium() {
    check_fp128_n(8);
    check_fp128_n(12);
}

#[test]
fn fp128_fused_large() {
    check_fp128_n(14);
    check_fp128_n(16);
}

#[test]
fn gf128_fused_n_zero_is_noop() {
    let f = make_gf128(16);
    let g = make_gf128(16);
    let challenges: Vec<_> = Vec::new();

    let mut f_fused = f.clone();
    let mut g_fused = g.clone();
    sumcheck_deg2_delayed_gf128_fused(&mut f_fused, &mut g_fused, &challenges);

    assert_eq!(f_fused, f, "GF128 fused should be a no-op with no challenges");
    assert_eq!(g_fused, g, "GF128 fused should be a no-op with no challenges");
}

#[test]
fn fp128_fused_n_zero_is_noop() {
    let f = make_fp128(16);
    let g = make_fp128(16);
    let challenges: Vec<_> = Vec::new();

    let mut f_fused = f.clone();
    let mut g_fused = g.clone();
    sumcheck_deg2_delayed_fp128_fused(&mut f_fused, &mut g_fused, &challenges);

    assert_eq!(f_fused, f, "Fp128 fused should be a no-op with no challenges");
    assert_eq!(g_fused, g, "Fp128 fused should be a no-op with no challenges");
}

// -----------------------------------------------------------------------------
// Relaxed / delayed-reduction Fp128 kernel correctness
// -----------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn check_fp128_relaxed_n(n: usize) {
    let f = make_fp128(1 << n);
    let g = make_fp128(1 << n);
    let challenges = make_fp128(n);

    let mut f_ref = f.clone();
    let mut g_ref = g.clone();
    sumcheck_deg2_delayed_fp128_fused(&mut f_ref, &mut g_ref, &challenges);

    let mut f_relax = f.clone();
    let mut g_relax = g.clone();
    sumcheck_deg2_delayed_fp128_fused_relaxed(&mut f_relax, &mut g_relax, &challenges);

    assert_eq!(
        f_relax.len(),
        f_ref.len(),
        "Fp128 relaxed f.len mismatch at n={n}"
    );
    assert_eq!(
        g_relax.len(),
        g_ref.len(),
        "Fp128 relaxed g.len mismatch at n={n}"
    );
    assert_eq!(f_relax, f_ref, "Fp128 relaxed f mismatch at n={n}");
    assert_eq!(g_relax, g_ref, "Fp128 relaxed g mismatch at n={n}");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn fp128_relaxed_edge_cases() {
    for n in 1..=4 {
        check_fp128_relaxed_n(n);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn fp128_relaxed_medium() {
    check_fp128_relaxed_n(8);
    check_fp128_relaxed_n(12);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn fp128_relaxed_large() {
    check_fp128_relaxed_n(14);
    check_fp128_relaxed_n(16);
}
