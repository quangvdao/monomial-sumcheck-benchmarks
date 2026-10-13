//! Round messages of the delayed kernels against the eager kernels.
//!
//! The kernels hand their partial sums to `round_message`, which records them
//! in test builds. Every test compares all sums of all rounds.

use super::*;
use binius_field::Field as _;
use jolt_field::Zero as _;

const N: usize = 8;

/// Run a kernel on copies of the tables and return its round messages.
fn messages<F: Clone, T: Copy + 'static>(
    f: &[F],
    g: &[F],
    kernel: impl FnOnce(&mut Vec<F>, &mut Vec<F>),
) -> Vec<T> {
    let (mut f, mut g) = (f.to_vec(), g.to_vec());
    round_messages_of(|| kernel(&mut f, &mut g))
}

fn assert_messages<F: Clone, T: Copy + PartialEq + std::fmt::Debug + 'static>(
    expected: &[T],
    f: &[F],
    g: &[F],
    kernel: impl FnOnce(&mut Vec<F>, &mut Vec<F>),
) {
    assert_eq!(messages::<F, T>(f, g, kernel), expected);
}

fn wrap_branch_free(values: &[BN254Fr]) -> Vec<Bn254BranchFree> {
    values.iter().copied().map(Bn254BranchFree).collect()
}

#[test]
fn bn254_delayed_round_messages_match_eager() {
    let f = make_bn254(1 << N);
    let g = make_bn254(1 << N);
    let ch = make_bn254(N);
    let (f_bf, g_bf, ch_bf) = (
        wrap_branch_free(&f),
        wrap_branch_free(&g),
        wrap_branch_free(&ch),
    );
    let zero = BN254Fr::ZERO;

    let eager: Vec<(BN254Fr, BN254Fr, BN254Fr)> =
        messages(&f, &g, |f, g| sumcheck_deg2_boolean(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_delayed_bn254(f, g, &ch)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_delayed_bn254_generic(f, g, &ch)
    });
    assert_messages(&eager, &f_bf, &g_bf, |f, g| {
        sumcheck_deg2_delayed_bn254_generic(f, g, &ch_bf)
    });

    let eager: Vec<(BN254Fr, BN254Fr, BN254Fr)> =
        messages(&f, &g, |f, g| sumcheck_deg2_projective(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_delayed_bn254(f, g, &ch)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_delayed_bn254_generic(f, g, &ch)
    });
    assert_messages(&eager, &f_bf, &g_bf, |f, g| {
        sumcheck_deg2_projective_delayed_bn254_generic(f, g, &ch_bf)
    });
}

#[test]
fn bn254_eq_delayed_round_messages_match_eager() {
    let f = make_bn254(1 << N);
    let g = make_bn254(1 << N);
    let ch = make_bn254(N);
    let eq_point = make_bn254(N);
    let (f_bf, g_bf, ch_bf, eq_point_bf) = (
        wrap_branch_free(&f),
        wrap_branch_free(&g),
        wrap_branch_free(&ch),
        wrap_branch_free(&eq_point),
    );
    let zero = BN254Fr::ZERO;
    let one = BN254Fr::from(1u64);

    let eq = build_suffix_eq_tables(&eq_point, one);
    let eq_bf = build_suffix_eq_tables(&eq_point_bf, Bn254BranchFree::ONE);
    let eager: Vec<(BN254Fr, BN254Fr)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_delayed_bn254(f, g, &eq, &ch)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_delayed_bn254_generic(f, g, &eq, &ch)
    });
    assert_messages(&eager, &f_bf, &g_bf, |f, g| {
        sumcheck_deg2_eq_delayed_bn254_generic(f, g, &eq_bf, &ch_bf)
    });

    let eq = build_suffix_eq_tables_projective(&eq_point, one);
    let eq_bf = build_suffix_eq_tables_projective(&eq_point_bf, Bn254BranchFree::ONE);
    let eager: Vec<(BN254Fr, BN254Fr)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_delayed_bn254(f, g, &eq, &ch)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_delayed_bn254_generic(f, g, &eq, &ch)
    });
    assert_messages(&eager, &f_bf, &g_bf, |f, g| {
        sumcheck_deg2_eq_projective_delayed_bn254_generic(f, g, &eq_bf, &ch_bf)
    });
}

#[test]
fn bn254_upper_limb_round_messages_match_eager() {
    let f = make_bn254(1 << N);
    let g = make_bn254(1 << N);
    let (ch, limbs) = make_bn254_upper_limb_challenges(N);
    let eq_point = make_bn254(N);
    let zero = BN254Fr::ZERO;
    let one = BN254Fr::from(1u64);

    // The eager upper-limb kernels against the eager kernels with full
    // multiplication, then the delayed upper-limb kernels against both.
    let eager: Vec<(BN254Fr, BN254Fr, BN254Fr)> =
        messages(&f, &g, |f, g| sumcheck_deg2_boolean(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_boolean_bn254_upper(f, g, &limbs)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_delayed_bn254_upper(f, g, &limbs)
    });

    let eager: Vec<(BN254Fr, BN254Fr, BN254Fr)> =
        messages(&f, &g, |f, g| sumcheck_deg2_projective(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_bn254_upper(f, g, &limbs)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_delayed_bn254_upper(f, g, &limbs)
    });

    let eq = build_suffix_eq_tables(&eq_point, one);
    let eager: Vec<(BN254Fr, BN254Fr)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean_bn254_upper(f, g, &eq, &limbs)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_delayed_bn254_upper(f, g, &eq, &limbs)
    });

    let eq = build_suffix_eq_tables_projective(&eq_point, one);
    let eager: Vec<(BN254Fr, BN254Fr)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective_bn254_upper(f, g, &eq, &limbs)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_delayed_bn254_upper(f, g, &eq, &limbs)
    });
}

#[test]
fn fp128_delayed_round_messages_match_eager() {
    let f = make_fp128(1 << N);
    let g = make_fp128(1 << N);
    let ch = make_fp128(N);
    let zero = Fp128::zero();

    let eager: Vec<(Fp128, Fp128, Fp128)> =
        messages(&f, &g, |f, g| sumcheck_deg2_boolean(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_boolean_fp128(f, g, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_delayed_fp128(f, g, &ch)
    });

    let eager: Vec<(Fp128, Fp128, Fp128)> =
        messages(&f, &g, |f, g| sumcheck_deg2_projective(f, g, &ch, zero));
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_fp128(f, g, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_delayed_fp128(f, g, &ch)
    });

    let eager: Vec<(Fp128, Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_projective_1inf_fp128(f, g, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_projective_1inf_delayed_fp128(f, g, &ch)
    });
}

#[test]
fn fp128_eq_delayed_round_messages_match_eager() {
    let f = make_fp128(1 << N);
    let g = make_fp128(1 << N);
    let ch = make_fp128(N);
    let eq_point = make_fp128(N);
    let zero = Fp128::zero();

    let eq = build_suffix_eq_tables(&eq_point, Fp128::one());
    let eager: Vec<(Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean_fp128(f, g, &eq, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_delayed_fp128(f, g, &eq, &ch)
    });

    let eq = build_suffix_eq_tables_projective(&eq_point, Fp128::one());
    let eager: Vec<(Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective_fp128(f, g, &eq, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_delayed_fp128(f, g, &eq, &ch)
    });

    let eager: Vec<(Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective_1inf_fp128(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_1inf_delayed_fp128(f, g, &eq, &ch)
    });
}

/// Evaluate the multilinear polynomial stored on {1,inf}^n at a point, low
/// variable first. `None` is the point at infinity (the leading coefficient).
fn eval_1inf(table: &[Fp128], point: &[Option<Fp128>]) -> Fp128 {
    let mut table = table.to_vec();
    for coordinate in point {
        let half = table.len() / 2;
        for j in 0..half {
            let (at_one, at_inf) = (table[2 * j], table[2 * j + 1]);
            table[j] = match coordinate {
                Some(x) => at_one + (*x - Fp128::one()) * at_inf,
                None => at_inf,
            };
        }
        table.truncate(half);
    }
    assert_eq!(table.len(), 1);
    table[0]
}

/// Sum `weight * f * g` over the remaining variables on {1,inf}, with the
/// earlier variables bound to the challenges and the current one to `x`.
fn direct_1inf_sum(
    f: &[Fp128],
    g: &[Fp128],
    bound: &[Fp128],
    x: Option<Fp128>,
    weights: Option<&[Fp128]>,
) -> Fp128 {
    let n = f.len().trailing_zeros() as usize;
    let rest = n - bound.len() - 1;
    let mut sum = Fp128::zero();
    for index in 0..1usize << rest {
        let mut point: Vec<Option<Fp128>> = bound.iter().copied().map(Some).collect();
        point.push(x);
        point.extend((0..rest).map(|bit| (index >> bit & 1 == 0).then(Fp128::one)));
        let term = eval_1inf(f, &point) * eval_1inf(g, &point);
        sum += match weights {
            Some(weights) => term * weights[index],
            None => term,
        };
    }
    sum
}

#[test]
fn fp128_1inf_round_messages_match_direct_summation() {
    let n = 6usize;
    let f = make_fp128(1 << n);
    let g = make_fp128(1 << n);
    let ch = make_fp128(n);
    let eq_point = make_fp128(n);
    let (zero, one) = (Fp128::zero(), Fp128::one());

    let got: Vec<(Fp128, Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_projective_1inf_fp128(f, g, &ch, zero)
    });
    assert_eq!(got.len(), n);
    for (round, &(h0, h1, h_inf)) in got.iter().enumerate() {
        let bound = &ch[..round];
        assert_eq!(h0, direct_1inf_sum(&f, &g, bound, Some(zero), None));
        assert_eq!(h1, direct_1inf_sum(&f, &g, bound, Some(one), None));
        assert_eq!(h_inf, direct_1inf_sum(&f, &g, bound, None, None));
    }

    let eq = build_suffix_eq_tables_projective(&eq_point, one);
    let got: Vec<(Fp128, Fp128)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective_1inf_fp128(f, g, &eq, &ch, zero)
    });
    assert_eq!(got.len(), n);
    for (round, &(q0, q_inf)) in got.iter().enumerate() {
        let bound = &ch[..round];
        let weights = Some(&eq[round + 1][..]);
        assert_eq!(q0, direct_1inf_sum(&f, &g, bound, Some(zero), weights));
        assert_eq!(q_inf, direct_1inf_sum(&f, &g, bound, None, weights));
    }

    // The kernels bind every variable to its challenge.
    let point: Vec<Option<Fp128>> = ch.iter().copied().map(Some).collect();
    let (mut f_bound, mut g_bound) = (f.clone(), g.clone());
    sumcheck_deg2_eq_gruen_projective_1inf_fp128(&mut f_bound, &mut g_bound, &eq, &ch, zero);
    assert_eq!(f_bound, [eval_1inf(&f, &point)]);
    assert_eq!(g_bound, [eval_1inf(&g, &point)]);
}

macro_rules! small_field_round_message_tests {
    ($name:ident, $field:ty, $make:ident, $suffix:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _delayed_round_messages_match_eager>]() {
                let f = $make(1 << N);
                let g = $make(1 << N);
                let ch = $make(N);
                let eq_point = $make(N);
                let (zero, one) = (<$field>::ZERO, <$field>::ONE);

                let eager: Vec<($field, $field, $field)> =
                    messages(&f, &g, |f, g| sumcheck_deg2_boolean(f, g, &ch, zero));
                assert_eq!(eager.len(), N);
                assert_messages(&eager, &f, &g, |f, g| [<sumcheck_deg2_delayed_ $suffix>](f, g, &ch));

                let eager: Vec<($field, $field, $field)> =
                    messages(&f, &g, |f, g| sumcheck_deg2_projective(f, g, &ch, zero));
                assert_eq!(eager.len(), N);
                assert_messages(&eager, &f, &g, |f, g| {
                        [<sumcheck_deg2_projective_delayed_ $suffix>](f, g, &ch)
                    });

                let eq = build_suffix_eq_tables(&eq_point, one);
                let eager: Vec<($field, $field)> = messages(&f, &g, |f, g| {
                    sumcheck_deg2_eq_gruen_boolean(f, g, &eq, &ch, zero)
                });
                assert_eq!(eager.len(), N);
                assert_messages(&eager, &f, &g, |f, g| {
                        [<sumcheck_deg2_eq_delayed_ $suffix>](f, g, &eq, &ch)
                    });

                let eq = build_suffix_eq_tables_projective(&eq_point, one);
                let eager: Vec<($field, $field)> = messages(&f, &g, |f, g| {
                    sumcheck_deg2_eq_gruen_projective(f, g, &eq, &ch, zero)
                });
                assert_eq!(eager.len(), N);
                assert_messages(&eager, &f, &g, |f, g| {
                        [<sumcheck_deg2_eq_projective_delayed_ $suffix>](f, g, &eq, &ch)
                    });
            }
        }
    };
}

small_field_round_message_tests!(bb4, BB4, make_bb4, bb4);
small_field_round_message_tests!(bb5, BB5, make_bb5, bb5);
small_field_round_message_tests!(kb5, KB5, make_kb5, kb5);
small_field_round_message_tests!(gf128, GF128, make_gf128, gf128);

#[test]
fn bb5_packed_eq_round_messages_match_eager() {
    let f = make_bb5(1 << N);
    let g = make_bb5(1 << N);
    let ch = make_bb5(N);
    let eq_point = make_bb5(N);
    let (zero, one) = (BB5::ZERO, BB5::ONE);

    let eq = build_suffix_eq_tables(&eq_point, one);
    let eager: Vec<(BB5, BB5)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_boolean_bb5(f, g, &eq, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_delayed_bb5_packed(f, g, &eq, &ch)
    });

    let eq = build_suffix_eq_tables_projective(&eq_point, one);
    let eager: Vec<(BB5, BB5)> = messages(&f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective(f, g, &eq, &ch, zero)
    });
    assert_eq!(eager.len(), N);
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_gruen_projective_bb5(f, g, &eq, &ch, zero)
    });
    assert_messages(&eager, &f, &g, |f, g| {
        sumcheck_deg2_eq_projective_delayed_bb5_packed(f, g, &eq, &ch)
    });
}
