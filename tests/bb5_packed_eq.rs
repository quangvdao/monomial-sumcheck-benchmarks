mod sumcheck_impl {
    #![allow(dead_code)]
    #![allow(unused_imports)]

    // Include the benchmark implementation directly so these tests exercise the
    // exact packed kernels against the scalar and generic reference paths.
    include!("../benches/sumcheck.rs");

    #[test]
    fn bb5_boolean_eval_matches_reference() {
        let n = 6usize;
        let f = make_bb5(1usize << n);
        let g = make_bb5(1usize << n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);
        let eq_rest = &suffix_eq[1];

        assert_eq!(
            bb5_eq_gruen_boolean_eval(&f, &g, eq_rest),
            bb5_eq_gruen_boolean_eval_ref(&f, &g, eq_rest),
        );
    }

    #[test]
    fn bb5_projective_eval_matches_reference() {
        let n = 6usize;
        let f = make_bb5(1usize << n);
        let g = make_bb5(1usize << n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);
        let eq_rest = &suffix_eq[1];
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        let got = bb5_eq_gruen_projective_eval_packed(&f, &g, eq_rest);
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        let got = bb5_eq_gruen_projective_eval_ref(&f, &g, eq_rest);

        assert_eq!(
            got,
            bb5_eq_gruen_projective_eval_ref(&f, &g, eq_rest),
        );
    }

    #[test]
    fn bb5_delayed_eval_matches_reference() {
        let n = 6usize;
        let f = make_bb5(1usize << n);
        let g = make_bb5(1usize << n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);
        let eq_rest = &suffix_eq[1];
        let (q1, q_inf) = bb5_eq_delayed_eval(&f, &g, eq_rest);
        let (q1_ref, q_inf_ref) = bb5_eq_delayed_eval_ref(&f, &g, eq_rest);

        assert_eq!(q1.reduce(), q1_ref.reduce());
        assert_eq!(q_inf.reduce(), q_inf_ref.reduce());
    }

    #[test]
    fn bb5_projective_delayed_eval_matches_reference() {
        let n = 6usize;
        let f = make_bb5(1usize << n);
        let g = make_bb5(1usize << n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);
        let eq_rest = &suffix_eq[1];
        let (q1, q_inf) = bb5_eq_projective_delayed_eval(&f, &g, eq_rest);
        let (q1_ref, q_inf_ref) = bb5_eq_projective_delayed_eval_ref(&f, &g, eq_rest);

        assert_eq!(q1.reduce(), q1_ref.reduce());
        assert_eq!(q_inf.reduce(), q_inf_ref.reduce());
    }

    #[test]
    fn bb5_boolean_wrapper_matches_generic() {
        let n = 6usize;
        let mut f_generic = make_bb5(1usize << n);
        let mut g_generic = make_bb5(1usize << n);
        let mut f_packed = f_generic.clone();
        let mut g_packed = g_generic.clone();
        let challenges = make_bb5(n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);

        sumcheck_deg2_eq_gruen_boolean(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
            BB5::ZERO,
        );
        sumcheck_deg2_eq_gruen_boolean_bb5(
            &mut f_packed,
            &mut g_packed,
            &suffix_eq,
            &challenges,
            BB5::ZERO,
        );

        assert_eq!(f_packed, f_generic);
        assert_eq!(g_packed, g_generic);
    }

    #[test]
    fn bb5_delayed_wrapper_matches_generic() {
        let n = 6usize;
        let mut f_generic = make_bb5(1usize << n);
        let mut g_generic = make_bb5(1usize << n);
        let mut f_packed = f_generic.clone();
        let mut g_packed = g_generic.clone();
        let challenges = make_bb5(n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);

        sumcheck_deg2_eq_delayed_bb5(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
        );
        sumcheck_deg2_eq_delayed_bb5_packed(
            &mut f_packed,
            &mut g_packed,
            &suffix_eq,
            &challenges,
        );

        assert_eq!(f_packed, f_generic);
        assert_eq!(g_packed, g_generic);
    }

    #[test]
    fn bb5_projective_wrapper_matches_generic() {
        let n = 6usize;
        let mut f_generic = make_bb5(1usize << n);
        let mut g_generic = make_bb5(1usize << n);
        let mut f_packed = f_generic.clone();
        let mut g_packed = g_generic.clone();
        let challenges = make_bb5(n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);

        sumcheck_deg2_eq_gruen_projective(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
            BB5::ZERO,
        );
        sumcheck_deg2_eq_gruen_projective_bb5(
            &mut f_packed,
            &mut g_packed,
            &suffix_eq,
            &challenges,
            BB5::ZERO,
        );

        assert_eq!(f_packed, f_generic);
        assert_eq!(g_packed, g_generic);
    }

    #[test]
    fn bb5_projective_delayed_wrapper_matches_generic() {
        let n = 6usize;
        let mut f_generic = make_bb5(1usize << n);
        let mut g_generic = make_bb5(1usize << n);
        let mut f_packed = f_generic.clone();
        let mut g_packed = g_generic.clone();
        let challenges = make_bb5(n);
        let eq_point = make_bb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, BB5::ONE);

        sumcheck_deg2_eq_projective_delayed_bb5(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
        );
        sumcheck_deg2_eq_projective_delayed_bb5_packed(
            &mut f_packed,
            &mut g_packed,
            &suffix_eq,
            &challenges,
        );

        assert_eq!(f_packed, f_generic);
        assert_eq!(g_packed, g_generic);
    }
}
