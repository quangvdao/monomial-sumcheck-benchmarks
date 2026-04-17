//! Correctness smoke tests for the parallel wrappers (approaches 1/2/3/4).
//!
//! The sequential kernels are already validated by `tests/bb5_packed_eq.rs`
//! (which covers the generic and field-specific delayed paths). These tests
//! confirm that each parallel wrapper in `src/sumcheck/parallel.rs` produces
//! bit-identical final `(f, g)` state as the sequential delayed kernel for
//! GF128 and Fp128, across a sweep of `n` that forces both the "fully inside
//! scope" case and the "cut over to sequential for the small tail" case of
//! the persistent-pool variant.
//!
//! Bit-identical f/g is the right invariant: the bind phase is deterministic
//! per-pair and doesn't depend on reduction order, so no reordering tolerance
//! is needed.

#![cfg(feature = "parallel")]

use monomial_sumcheck_benchmarks::sumcheck::*;

fn check_gf128(n: usize) {
    let f = make_gf128(1 << n);
    let g = make_gf128(1 << n);
    let challenges = make_gf128(n);

    let mut f_seq = f.clone();
    let mut g_seq = g.clone();
    sumcheck_deg2_delayed_gf128(&mut f_seq, &mut g_seq, &challenges);

    let mut f_scope = f.clone();
    let mut g_scope = g.clone();
    sumcheck_deg2_delayed_gf128_rayon_scope(&mut f_scope, &mut g_scope, &challenges);

    let mut f_pi = f.clone();
    let mut g_pi = g.clone();
    sumcheck_deg2_delayed_gf128_rayon_iter(&mut f_pi, &mut g_pi, &challenges);

    let mut f_p3 = f.clone();
    let mut g_p3 = g.clone();
    sumcheck_deg2_delayed_gf128_persistent(&mut f_p3, &mut g_p3, &challenges);

    for schedule in [Schedule::Static, Schedule::guided()] {
        let mut f_p4 = f.clone();
        let mut g_p4 = g.clone();
        sumcheck_deg2_delayed_gf128_pinned(&mut f_p4, &mut g_p4, &challenges, false, schedule);

        let mut f_p4_fused = f.clone();
        let mut g_p4_fused = g.clone();
        sumcheck_deg2_delayed_gf128_pinned(&mut f_p4_fused, &mut g_p4_fused, &challenges, true, schedule);

        assert_eq!(
            f_p4, f_seq,
            "GF128 pinned (unfused, {schedule:?}) f mismatch at n={n}"
        );
        assert_eq!(
            g_p4, g_seq,
            "GF128 pinned (unfused, {schedule:?}) g mismatch at n={n}"
        );
        assert_eq!(
            f_p4_fused, f_seq,
            "GF128 pinned (fused, {schedule:?}) f mismatch at n={n}"
        );
        assert_eq!(
            g_p4_fused, g_seq,
            "GF128 pinned (fused, {schedule:?}) g mismatch at n={n}"
        );
    }

    assert_eq!(f_scope, f_seq, "GF128 rayon_scope f mismatch at n={n}");
    assert_eq!(g_scope, g_seq, "GF128 rayon_scope g mismatch at n={n}");
    assert_eq!(f_pi, f_seq, "GF128 rayon_iter f mismatch at n={n}");
    assert_eq!(g_pi, g_seq, "GF128 rayon_iter g mismatch at n={n}");
    assert_eq!(f_p3, f_seq, "GF128 persistent f mismatch at n={n}");
    assert_eq!(g_p3, g_seq, "GF128 persistent g mismatch at n={n}");

    #[cfg(feature = "parallel_chili")]
    {
        for base in [8, 32, 128, 512].iter() {
            let mut f_ch = f.clone();
            let mut g_ch = g.clone();
            sumcheck_deg2_delayed_gf128_chili(&mut f_ch, &mut g_ch, &challenges, *base);
            assert_eq!(f_ch, f_seq, "GF128 chili (base={base}) f mismatch at n={n}");
            assert_eq!(g_ch, g_seq, "GF128 chili (base={base}) g mismatch at n={n}");
        }
    }
}

fn check_fp128(n: usize) {
    let f = make_fp128(1 << n);
    let g = make_fp128(1 << n);
    let challenges = make_fp128(n);

    let mut f_seq = f.clone();
    let mut g_seq = g.clone();
    sumcheck_deg2_delayed_fp128(&mut f_seq, &mut g_seq, &challenges);

    let mut f_scope = f.clone();
    let mut g_scope = g.clone();
    sumcheck_deg2_delayed_fp128_rayon_scope(&mut f_scope, &mut g_scope, &challenges);

    let mut f_pi = f.clone();
    let mut g_pi = g.clone();
    sumcheck_deg2_delayed_fp128_rayon_iter(&mut f_pi, &mut g_pi, &challenges);

    let mut f_p3 = f.clone();
    let mut g_p3 = g.clone();
    sumcheck_deg2_delayed_fp128_persistent(&mut f_p3, &mut g_p3, &challenges);

    for schedule in [Schedule::Static, Schedule::guided()] {
        let mut f_p4 = f.clone();
        let mut g_p4 = g.clone();
        sumcheck_deg2_delayed_fp128_pinned(&mut f_p4, &mut g_p4, &challenges, false, schedule);

        let mut f_p4_fused = f.clone();
        let mut g_p4_fused = g.clone();
        sumcheck_deg2_delayed_fp128_pinned(&mut f_p4_fused, &mut g_p4_fused, &challenges, true, schedule);

        assert_eq!(
            f_p4, f_seq,
            "Fp128 pinned (unfused, {schedule:?}) f mismatch at n={n}"
        );
        assert_eq!(
            g_p4, g_seq,
            "Fp128 pinned (unfused, {schedule:?}) g mismatch at n={n}"
        );
        assert_eq!(
            f_p4_fused, f_seq,
            "Fp128 pinned (fused, {schedule:?}) f mismatch at n={n}"
        );
        assert_eq!(
            g_p4_fused, g_seq,
            "Fp128 pinned (fused, {schedule:?}) g mismatch at n={n}"
        );
    }

    assert_eq!(f_scope, f_seq, "Fp128 rayon_scope f mismatch at n={n}");
    assert_eq!(g_scope, g_seq, "Fp128 rayon_scope g mismatch at n={n}");
    assert_eq!(f_pi, f_seq, "Fp128 rayon_iter f mismatch at n={n}");
    assert_eq!(g_pi, g_seq, "Fp128 rayon_iter g mismatch at n={n}");
    assert_eq!(f_p3, f_seq, "Fp128 persistent f mismatch at n={n}");
    assert_eq!(g_p3, g_seq, "Fp128 persistent g mismatch at n={n}");

    #[cfg(feature = "parallel_chili")]
    {
        for base in [8, 32, 128, 512].iter() {
            let mut f_ch = f.clone();
            let mut g_ch = g.clone();
            sumcheck_deg2_delayed_fp128_chili(&mut f_ch, &mut g_ch, &challenges, *base);
            assert_eq!(f_ch, f_seq, "Fp128 chili (base={base}) f mismatch at n={n}");
            assert_eq!(g_ch, g_seq, "Fp128 chili (base={base}) g mismatch at n={n}");
        }
    }
}

#[test]
fn gf128_delayed_parallel_matches_sequential_small() {
    for n in [4usize, 6, 8, 10] {
        check_gf128(n);
    }
}

#[test]
fn gf128_delayed_parallel_matches_sequential_medium() {
    check_gf128(12);
    check_gf128(14);
}

#[test]
fn fp128_delayed_parallel_matches_sequential_small() {
    for n in [4usize, 6, 8, 10] {
        check_fp128(n);
    }
}

#[test]
fn fp128_delayed_parallel_matches_sequential_medium() {
    check_fp128(12);
    check_fp128(14);
}
