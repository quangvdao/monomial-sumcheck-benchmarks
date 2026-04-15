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

    #[test]
    fn bn254_upper_boolean_paths_match_standard() {
        let n = 6usize;
        let f_orig = make_bn254(1usize << n);
        let g_orig = make_bn254(1usize << n);
        let (challenges, challenge_limbs) = make_bn254_upper_limb_challenges(n);

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_upper = f_orig.clone();
        let mut g_upper = g_orig.clone();
        sumcheck_deg2_boolean(&mut f_generic, &mut g_generic, &challenges, BN254Fr::ZERO);
        sumcheck_deg2_boolean_bn254_upper(&mut f_upper, &mut g_upper, &challenge_limbs);
        assert_eq!(f_upper, f_generic);
        assert_eq!(g_upper, g_generic);

        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        let mut f_upper_delayed = f_orig;
        let mut g_upper_delayed = g_orig;
        sumcheck_deg2_delayed_bn254(&mut f_delayed, &mut g_delayed, &challenges);
        sumcheck_deg2_delayed_bn254_upper(&mut f_upper_delayed, &mut g_upper_delayed, &challenge_limbs);
        assert_eq!(f_upper_delayed, f_delayed);
        assert_eq!(g_upper_delayed, g_delayed);
    }

    #[test]
    fn bn254_upper_projective_paths_match_standard() {
        let n = 6usize;
        let f_orig = make_bn254(1usize << n);
        let g_orig = make_bn254(1usize << n);
        let (challenges, challenge_limbs) = make_bn254_upper_limb_challenges(n);

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_upper = f_orig.clone();
        let mut g_upper = g_orig.clone();
        sumcheck_deg2_projective(&mut f_generic, &mut g_generic, &challenges, BN254Fr::ZERO);
        sumcheck_deg2_projective_bn254_upper(&mut f_upper, &mut g_upper, &challenge_limbs);
        assert_eq!(f_upper, f_generic);
        assert_eq!(g_upper, g_generic);

        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        let mut f_upper_delayed = f_orig;
        let mut g_upper_delayed = g_orig;
        sumcheck_deg2_projective_delayed_bn254(&mut f_delayed, &mut g_delayed, &challenges);
        sumcheck_deg2_projective_delayed_bn254_upper(
            &mut f_upper_delayed,
            &mut g_upper_delayed,
            &challenge_limbs,
        );
        assert_eq!(f_upper_delayed, f_delayed);
        assert_eq!(g_upper_delayed, g_delayed);
    }

    #[test]
    fn bn254_upper_boolean_eq_paths_match_standard() {
        let n = 6usize;
        let f_orig = make_bn254(1usize << n);
        let g_orig = make_bn254(1usize << n);
        let (eq_point, challenge_limbs) = make_bn254_upper_limb_challenges(n);
        let challenges: Vec<_> = challenge_limbs
            .iter()
            .map(|&(lo, hi)| BN254Fr::new_unchecked(ark_ff::BigInt([0, 0, lo, hi])))
            .collect();
        let suffix_eq = build_suffix_eq_tables(&eq_point, BN254Fr::from(1u64));

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_upper = f_orig.clone();
        let mut g_upper = g_orig.clone();
        sumcheck_deg2_eq_gruen_boolean(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
            BN254Fr::ZERO,
        );
        sumcheck_deg2_eq_gruen_boolean_bn254_upper(
            &mut f_upper,
            &mut g_upper,
            &suffix_eq,
            &challenge_limbs,
        );
        assert_eq!(f_upper, f_generic);
        assert_eq!(g_upper, g_generic);

        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        let mut f_upper_delayed = f_orig;
        let mut g_upper_delayed = g_orig;
        sumcheck_deg2_eq_delayed_bn254(&mut f_delayed, &mut g_delayed, &suffix_eq, &challenges);
        sumcheck_deg2_eq_delayed_bn254_upper(
            &mut f_upper_delayed,
            &mut g_upper_delayed,
            &suffix_eq,
            &challenge_limbs,
        );
        assert_eq!(f_upper_delayed, f_delayed);
        assert_eq!(g_upper_delayed, g_delayed);
    }

    #[test]
    fn bn254_upper_projective_eq_paths_match_standard() {
        let n = 6usize;
        let f_orig = make_bn254(1usize << n);
        let g_orig = make_bn254(1usize << n);
        let (eq_point, challenge_limbs) = make_bn254_upper_limb_challenges(n);
        let challenges: Vec<_> = challenge_limbs
            .iter()
            .map(|&(lo, hi)| BN254Fr::new_unchecked(ark_ff::BigInt([0, 0, lo, hi])))
            .collect();
        let suffix_eq = build_suffix_eq_tables(&eq_point, BN254Fr::from(1u64));

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_upper = f_orig.clone();
        let mut g_upper = g_orig.clone();
        sumcheck_deg2_eq_gruen_projective(
            &mut f_generic,
            &mut g_generic,
            &suffix_eq,
            &challenges,
            BN254Fr::ZERO,
        );
        sumcheck_deg2_eq_gruen_projective_bn254_upper(
            &mut f_upper,
            &mut g_upper,
            &suffix_eq,
            &challenge_limbs,
        );
        assert_eq!(f_upper, f_generic);
        assert_eq!(g_upper, g_generic);

        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        let mut f_upper_delayed = f_orig;
        let mut g_upper_delayed = g_orig;
        sumcheck_deg2_eq_projective_delayed_bn254(
            &mut f_delayed,
            &mut g_delayed,
            &suffix_eq,
            &challenges,
        );
        sumcheck_deg2_eq_projective_delayed_bn254_upper(
            &mut f_upper_delayed,
            &mut g_upper_delayed,
            &suffix_eq,
            &challenge_limbs,
        );
        assert_eq!(f_upper_delayed, f_delayed);
        assert_eq!(g_upper_delayed, g_delayed);
    }

    #[test]
    fn kb5_delayed_boolean_paths_match_generic() {
        let n = 6usize;
        let f_orig = make_kb5(1usize << n);
        let g_orig = make_kb5(1usize << n);
        let challenges = make_kb5(n);

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        sumcheck_deg2_boolean(&mut f_generic, &mut g_generic, &challenges, KB5::ZERO);
        sumcheck_deg2_delayed_kb5(&mut f_delayed, &mut g_delayed, &challenges);
        assert_eq!(f_delayed, f_generic);
        assert_eq!(g_delayed, g_generic);

        let eq_point = make_kb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, KB5::ONE);
        let mut f_generic_eq = f_orig.clone();
        let mut g_generic_eq = g_orig.clone();
        let mut f_delayed_eq = f_orig;
        let mut g_delayed_eq = g_orig;
        sumcheck_deg2_eq_gruen_boolean(
            &mut f_generic_eq,
            &mut g_generic_eq,
            &suffix_eq,
            &challenges,
            KB5::ZERO,
        );
        sumcheck_deg2_eq_delayed_kb5(
            &mut f_delayed_eq,
            &mut g_delayed_eq,
            &suffix_eq,
            &challenges,
        );
        assert_eq!(f_delayed_eq, f_generic_eq);
        assert_eq!(g_delayed_eq, g_generic_eq);
    }

    #[test]
    fn kb5_delayed_projective_paths_match_generic() {
        let n = 6usize;
        let f_orig = make_kb5(1usize << n);
        let g_orig = make_kb5(1usize << n);
        let challenges = make_kb5(n);

        let mut f_generic = f_orig.clone();
        let mut g_generic = g_orig.clone();
        let mut f_delayed = f_orig.clone();
        let mut g_delayed = g_orig.clone();
        sumcheck_deg2_projective(&mut f_generic, &mut g_generic, &challenges, KB5::ZERO);
        sumcheck_deg2_projective_delayed_kb5(&mut f_delayed, &mut g_delayed, &challenges);
        assert_eq!(f_delayed, f_generic);
        assert_eq!(g_delayed, g_generic);

        let eq_point = make_kb5(n);
        let suffix_eq = build_suffix_eq_tables(&eq_point, KB5::ONE);
        let mut f_generic_eq = f_orig.clone();
        let mut g_generic_eq = g_orig.clone();
        let mut f_delayed_eq = f_orig;
        let mut g_delayed_eq = g_orig;
        sumcheck_deg2_eq_gruen_projective(
            &mut f_generic_eq,
            &mut g_generic_eq,
            &suffix_eq,
            &challenges,
            KB5::ZERO,
        );
        sumcheck_deg2_eq_projective_delayed_kb5(
            &mut f_delayed_eq,
            &mut g_delayed_eq,
            &suffix_eq,
            &challenges,
        );
        assert_eq!(f_delayed_eq, f_generic_eq);
        assert_eq!(g_delayed_eq, g_generic_eq);
    }
}
