//! A/B profiler for the sequential Fp128 fused delayed-reduction kernel
//! vs. its relaxed (non-canonical intermediates, skip ≥p selector) variant.
//!
//! ```text
//! N=16 N_ITER=2000 cargo run --example relaxed_ab --release
//! ```
//!
//! Reports min/p10/p50/p90/p99/max wall time per call for both kernels,
//! interleaved round-robin to share cache / thermal / OS-noise conditions.

#![cfg(target_arch = "aarch64")]

use std::time::Instant;

use monomial_sumcheck_benchmarks::sumcheck::*;

fn percentiles(samples: &mut Vec<f64>) -> (f64, f64, f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| samples[((samples.len() as f64) * q) as usize];
    (
        samples[0],
        p(0.10),
        p(0.50),
        p(0.90),
        p(0.99),
        samples[samples.len() - 1],
    )
}

fn print_row(label: &str, n: usize, samples: &mut Vec<f64>) {
    let (min, p10, p50, p90, p99, max) = percentiles(samples);
    println!(
        "{label:32} n={n}: min={min:8.2}µs p10={p10:8.2}µs p50={p50:8.2}µs \
         p90={p90:8.2}µs p99={p99:8.2}µs max={max:8.2}µs"
    );
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let f_orig = make_fp128(1 << n);
    let g_orig = make_fp128(1 << n);
    let challenges = make_fp128(n);

    // Warmup both kernels.
    for _ in 0..warmup {
        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        sumcheck_deg2_delayed_fp128_fused(&mut f, &mut g, &challenges);
        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        sumcheck_deg2_delayed_fp128_fused_relaxed(&mut f, &mut g, &challenges);
    }

    // Interleaved samples.
    let mut canonical_us: Vec<f64> = Vec::with_capacity(n_iter);
    let mut relaxed_us: Vec<f64> = Vec::with_capacity(n_iter);

    for i in 0..n_iter {
        if i % 2 == 0 {
            {
                let mut f = f_orig.clone();
                let mut g = g_orig.clone();
                let t = Instant::now();
                sumcheck_deg2_delayed_fp128_fused(&mut f, &mut g, &challenges);
                let elapsed_us = t.elapsed().as_nanos() as f64 / 1_000.0;
                canonical_us.push(elapsed_us);
                std::hint::black_box((f, g));
            }
            {
                let mut f = f_orig.clone();
                let mut g = g_orig.clone();
                let t = Instant::now();
                sumcheck_deg2_delayed_fp128_fused_relaxed(&mut f, &mut g, &challenges);
                let elapsed_us = t.elapsed().as_nanos() as f64 / 1_000.0;
                relaxed_us.push(elapsed_us);
                std::hint::black_box((f, g));
            }
        } else {
            {
                let mut f = f_orig.clone();
                let mut g = g_orig.clone();
                let t = Instant::now();
                sumcheck_deg2_delayed_fp128_fused_relaxed(&mut f, &mut g, &challenges);
                let elapsed_us = t.elapsed().as_nanos() as f64 / 1_000.0;
                relaxed_us.push(elapsed_us);
                std::hint::black_box((f, g));
            }
            {
                let mut f = f_orig.clone();
                let mut g = g_orig.clone();
                let t = Instant::now();
                sumcheck_deg2_delayed_fp128_fused(&mut f, &mut g, &challenges);
                let elapsed_us = t.elapsed().as_nanos() as f64 / 1_000.0;
                canonical_us.push(elapsed_us);
                std::hint::black_box((f, g));
            }
        }
    }

    println!(
        "--- Sequential Fp128 fused: canonical vs relaxed (delayed reduction) ---"
    );
    println!("n = {n}, n_iter = {n_iter}, warmup = {warmup}");
    print_row("FUSED_CANONICAL", n, &mut canonical_us);
    print_row("FUSED_RELAXED", n, &mut relaxed_us);

    let p50_c = {
        canonical_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        canonical_us[canonical_us.len() / 2]
    };
    let p50_r = {
        relaxed_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        relaxed_us[relaxed_us.len() / 2]
    };
    let delta = (p50_r - p50_c) / p50_c * 100.0;
    println!(
        "Δp50 = {:+.2}% (negative = relaxed faster)",
        delta
    );
}
