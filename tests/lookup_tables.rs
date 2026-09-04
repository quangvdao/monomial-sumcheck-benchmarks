use ark_ff::AdditiveGroup;
use monomial_sumcheck_benchmarks::{
    lookup_tables::{
        eq_evals_boolean, eq_evals_projective, eq_mle_boolean, eq_mle_projective, lt_evals_boolean,
        lt_evals_projective, lt_mle_boolean, lt_mle_projective,
    },
    sumcheck::{build_suffix_eq_tables, build_suffix_eq_tables_projective, make_bn254, BN254Fr},
};

fn bit(index: usize, variable: usize, variables: usize) -> bool {
    ((index >> (variables - 1 - variable)) & 1) == 1
}

fn evaluate_boolean_table(table: &[BN254Fr], point: &[BN254Fr]) -> BN254Fr {
    let one = BN254Fr::from(1u64);
    table
        .iter()
        .enumerate()
        .map(|(index, value)| {
            point
                .iter()
                .enumerate()
                .fold(*value, |term, (variable, coordinate)| {
                    term * if bit(index, variable, point.len()) {
                        *coordinate
                    } else {
                        one - *coordinate
                    }
                })
        })
        .sum()
}

fn evaluate_projective_coefficients(coefficients: &[BN254Fr], point: &[BN254Fr]) -> BN254Fr {
    coefficients
        .iter()
        .enumerate()
        .map(|(index, coefficient)| {
            point
                .iter()
                .enumerate()
                .filter(|(variable, _)| bit(index, *variable, point.len()))
                .fold(*coefficient, |term, (_, coordinate)| term * *coordinate)
        })
        .sum()
}

#[test]
fn equality_table_builders_match_their_closed_forms() {
    let one = BN254Fr::from(1u64);
    let fixed_point = make_bn254(4);
    let evaluation_point = make_bn254(8)[4..8].to_vec();

    let boolean = eq_evals_boolean(&fixed_point, one);
    assert_eq!(
        evaluate_boolean_table(&boolean, &evaluation_point),
        eq_mle_boolean(&evaluation_point, &fixed_point, one),
    );

    let projective = eq_evals_projective(&fixed_point, one);
    assert_eq!(
        evaluate_projective_coefficients(&projective, &evaluation_point),
        eq_mle_projective(&evaluation_point, &fixed_point, one),
    );
}

#[test]
fn less_than_table_builders_match_their_closed_forms() {
    let zero = BN254Fr::ZERO;
    let one = BN254Fr::from(1u64);
    let fixed_point = make_bn254(4);
    let evaluation_point = make_bn254(8)[4..8].to_vec();

    let boolean = lt_evals_boolean(&fixed_point, zero);
    assert_eq!(
        evaluate_boolean_table(&boolean, &evaluation_point),
        lt_mle_boolean(&evaluation_point, &fixed_point, zero, one),
    );

    let projective = lt_evals_projective(&fixed_point, zero, one);
    assert_eq!(
        evaluate_projective_coefficients(&projective, &evaluation_point),
        lt_mle_projective(&evaluation_point, &fixed_point, zero, one),
    );
}

#[test]
fn suffix_equality_tables_use_the_declared_basis() {
    let one = BN254Fr::from(1u64);
    let point = make_bn254(4);
    let boolean = build_suffix_eq_tables(&point, one);
    let projective = build_suffix_eq_tables_projective(&point, one);

    for k in 0..=point.len() {
        for index in 0..(1usize << (point.len() - k)) {
            let mut expected_boolean = one;
            let mut expected_projective = one;
            for variable in k..point.len() {
                let local_variable = variable - k;
                // Suffix tables place the next sum-check variable in the low
                // bit because each round consumes adjacent pairs.
                if ((index >> local_variable) & 1) == 1 {
                    expected_boolean *= point[variable];
                    expected_projective *= point[variable];
                } else {
                    expected_boolean *= one - point[variable];
                }
            }
            assert_eq!(boolean[k][index], expected_boolean);
            assert_eq!(projective[k][index], expected_projective);
        }
    }
}
