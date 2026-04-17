//! Hang-repro: call `pinned` many times at n=12. Bench loops in
//! Criterion can hang if the per-phase pool state leaks between calls;
//! this test mirrors that pattern at smaller scale.

#![cfg(feature = "parallel")]

use monomial_sumcheck_benchmarks::sumcheck::*;

#[test]
fn gf128_pinned_loop_n12() {
    let n = 12usize;
    let f_orig = make_gf128(1usize << n);
    let g_orig = make_gf128(1usize << n);
    let challenges = make_gf128(n);

    // Alternate fused and schedule so we exercise all 4 shapes over
    // 200 iterations of the hot loop.
    for i in 0..200 {
        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        let fused = i & 1 == 1;
        let schedule = if i & 2 == 2 {
            Schedule::guided()
        } else {
            Schedule::Static
        };
        sumcheck_deg2_delayed_gf128_pinned(&mut f, &mut g, &challenges, fused, schedule);
        assert_eq!(
            f.len(),
            1,
            "iter {i} (fused={fused}, {schedule:?}): bad final length"
        );
    }
}

#[test]
fn fp128_pinned_loop_n12() {
    let n = 12usize;
    let f_orig = make_fp128(1usize << n);
    let g_orig = make_fp128(1usize << n);
    let challenges = make_fp128(n);

    for i in 0..200 {
        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        let fused = i & 1 == 1;
        let schedule = if i & 2 == 2 {
            Schedule::guided()
        } else {
            Schedule::Static
        };
        sumcheck_deg2_delayed_fp128_pinned(&mut f, &mut g, &challenges, fused, schedule);
        assert_eq!(
            f.len(),
            1,
            "iter {i} (fused={fused}, {schedule:?}): bad final length"
        );
    }
}
