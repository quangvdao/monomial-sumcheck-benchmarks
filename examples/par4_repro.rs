//! Per-call latency profiler for `par4_pinned`. Useful when Criterion's
//! median is being dragged around by long-tail outliers (preemption,
//! cache cold-starts, etc.) and you want to see the actual distribution.
//!
//! ```text
//! N=16 FIELD=fp128 N_ITER=2000 cargo run --example par4_repro --release \
//!     --features parallel
//! ```
//!
//! Reports p10/p50/p90/p99/min/max wall time per call. Setup (Vec::clone)
//! is excluded from each sample; only the par4_pinned call is timed.

#![cfg(feature = "parallel")]

use std::time::Instant;

use monomial_sumcheck_benchmarks::sumcheck::*;

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let field = std::env::var("FIELD").unwrap_or_else(|_| "fp128".into());
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    eprintln!("n={n} field={field} n_iter={n_iter}");
    eprintln!("warming up pool");
    PinnedPool::global().broadcast_scoped(PinnedPool::global().n_workers(), &|_| {});

    let mut samples: Vec<f64> = Vec::with_capacity(n_iter);

    if field == "fp128" {
        let f_orig = make_fp128(1usize << n);
        let g_orig = make_fp128(1usize << n);
        let challenges = make_fp128(n);
        for _ in 0..n_iter {
            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_fp128_par4_pinned(&mut f, &mut g, &challenges);
            let dt = t.elapsed().as_secs_f64() * 1e6;
            samples.push(dt);
            assert_eq!(f.len(), 1);
        }
    } else {
        let f_orig = make_gf128(1usize << n);
        let g_orig = make_gf128(1usize << n);
        let challenges = make_gf128(n);
        for _ in 0..n_iter {
            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_gf128_par4_pinned(&mut f, &mut g, &challenges);
            let dt = t.elapsed().as_secs_f64() * 1e6;
            samples.push(dt);
            assert_eq!(f.len(), 1);
        }
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| samples[((samples.len() as f64) * q) as usize];
    eprintln!(
        "{} n={n}: p10={:.2}µs p50={:.2}µs p90={:.2}µs p99={:.2}µs min={:.2}µs max={:.2}µs",
        field,
        p(0.10),
        p(0.50),
        p(0.90),
        p(0.99),
        samples[0],
        samples[samples.len() - 1],
    );
}
