use std::iter::zip;
use std::ops::{Add, Mul, Sub};

pub fn eq_evals_boolean<F>(r: &[F], one: F) -> Vec<F>
where
    F: Copy + Mul<Output = F> + Sub<Output = F>,
{
    let mut evals = vec![one; 1 << r.len()];
    let mut size = 1;
    for r_j in r {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = evals[i / 2];
            evals[i] = scalar * *r_j;
            evals[i - 1] = scalar - evals[i];
        }
    }
    evals
}

pub fn eq_evals_projective<F>(r: &[F], one: F) -> Vec<F>
where
    F: Copy + Mul<Output = F>,
{
    let mut coefficients = vec![one; 1 << r.len()];
    let mut size = 1;
    for r_j in r {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let scalar = coefficients[i / 2];
            coefficients[i] = scalar * *r_j;
            coefficients[i - 1] = scalar;
        }
    }
    coefficients
}

pub fn lt_evals_boolean<F>(r: &[F], zero: F) -> Vec<F>
where
    F: Copy + Mul<Output = F> + Add<Output = F> + Sub<Output = F>,
{
    let mut evals = vec![zero; 1 << r.len()];
    for (i, r_i) in r.iter().rev().enumerate() {
        let (left, right) = evals.split_at_mut(1 << i);
        zip(left, right).for_each(|(x, y)| {
            *y = *x * *r_i;
            *x = *x + *r_i - *y;
        });
    }
    evals
}

pub fn lt_evals_projective<F>(r: &[F], zero: F, one: F) -> Vec<F>
where
    F: Copy + Mul<Output = F> + Add<Output = F>,
{
    let mut coefficients = vec![zero; 1 << r.len()];
    let mut suffix_omega = one;
    for (i, r_i) in r.iter().rev().enumerate() {
        let r_times_omega = *r_i * suffix_omega;
        let (left, right) = coefficients.split_at_mut(1 << i);
        zip(left, right).for_each(|(x, y)| {
            *y = *x * *r_i;
            *x = *x + r_times_omega;
        });
        suffix_omega = suffix_omega * (one + *r_i);
    }
    coefficients
}

pub fn eq_mle_boolean<F>(x: &[F], y: &[F], one: F) -> F
where
    F: Copy + Mul<Output = F> + Add<Output = F> + Sub<Output = F>,
{
    assert_eq!(x.len(), y.len());
    let mut result = one;
    for i in 0..x.len() {
        result = result * (x[i] * y[i] + (one - x[i]) * (one - y[i]));
    }
    result
}

pub fn eq_mle_projective<F>(x: &[F], y: &[F], one: F) -> F
where
    F: Copy + Mul<Output = F> + Add<Output = F>,
{
    assert_eq!(x.len(), y.len());
    let mut result = one;
    for i in 0..x.len() {
        result = result * (one + x[i] * y[i]);
    }
    result
}

pub fn lt_mle_boolean<F>(x: &[F], y: &[F], zero: F, one: F) -> F
where
    F: Copy + Mul<Output = F> + Add<Output = F> + Sub<Output = F>,
{
    assert_eq!(x.len(), y.len());
    let mut result = zero;
    let mut eq_term = one;
    for i in 0..x.len() {
        result = result + (one - x[i]) * y[i] * eq_term;
        eq_term = eq_term * (x[i] * y[i] + (one - x[i]) * (one - y[i]));
    }
    result
}

pub fn lt_mle_projective<F>(x: &[F], y: &[F], zero: F, one: F) -> F
where
    F: Copy + Mul<Output = F> + Add<Output = F>,
{
    assert_eq!(x.len(), y.len());
    let mut omega_suffix = vec![one; x.len() + 1];
    for i in (0..x.len()).rev() {
        omega_suffix[i] = omega_suffix[i + 1] * (one + x[i]) * (one + y[i]);
    }

    let mut result = zero;
    let mut eq_prefix = one;
    for i in 0..x.len() {
        result = result + y[i] * eq_prefix * omega_suffix[i + 1];
        eq_prefix = eq_prefix * (one + x[i] * y[i]);
    }
    result
}
