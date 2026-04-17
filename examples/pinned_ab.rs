//! A/B profiler over the pinned-pool sumcheck paths. Three
//! configurations, interleaved round-robin to share cache / thermal
//! conditions across samples:
//!
//! 1. `NEW_FUSED`: trait + D2 multi-phase, `use_fused_path = true`.
//!    Rounds 1+ of each phase go through
//!    `SumcheckRound::bind_then_reduce_chunk` (single pass over the
//!    previous-round buffer for bind + reduce).
//! 2. `NEW_UNFUSED`: trait + D2 multi-phase, `use_fused_path = false`.
//!    Classic split `reduce_chunk` + `bind_chunk` protocol.
//! 3. `LEGACY`: single-broadcast, no D2, direct pointer captures.
//!    Kept as a reference for the earlier productionization work.
//!
//! Primary A/B of interest for the fused-bind-eval investigation:
//! `NEW_FUSED` vs `NEW_UNFUSED`. The `LEGACY` row is informational.
//!
//! ```text
//! N=16 FIELD=gf128 N_ITER=2000 cargo run --example pinned_ab --release \
//!     --features parallel
//! ```
//!
//! Reports p10/p50/p90/p99 wall time per call for all three.

#![cfg(feature = "parallel")]

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
        "{label:30} n={n}: min={min:7.2}µs p10={p10:7.2}µs p50={p50:7.2}µs \
         p90={p90:7.2}µs p99={p99:7.2}µs max={max:7.2}µs"
    );
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(14);
    let field = std::env::var("FIELD").unwrap_or_else(|_| "gf128".into());
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    eprintln!("n={n} field={field} n_iter={n_iter}");
    PinnedPool::global().broadcast_scoped(PinnedPool::global().n_workers(), &|_| {});

    let mut samples_fused: Vec<f64> = Vec::with_capacity(n_iter);
    let mut samples_unfused: Vec<f64> = Vec::with_capacity(n_iter);
    let mut samples_legacy: Vec<f64> = Vec::with_capacity(n_iter);

    if field == "fp128" {
        let f_orig = make_fp128(1usize << n);
        let g_orig = make_fp128(1usize << n);
        let challenges = make_fp128(n);

        // Interleave to share thermal / cache conditions across calls.
        for _ in 0..n_iter {
            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_fp128_pinned(&mut f, &mut g, &challenges, true);
            samples_fused.push(t.elapsed().as_secs_f64() * 1e6);

            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_fp128_pinned(&mut f, &mut g, &challenges, false);
            samples_unfused.push(t.elapsed().as_secs_f64() * 1e6);

            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_fp128_pinned_v0(&mut f, &mut g, &challenges);
            samples_legacy.push(t.elapsed().as_secs_f64() * 1e6);
        }
    } else {
        let f_orig = make_gf128(1usize << n);
        let g_orig = make_gf128(1usize << n);
        let challenges = make_gf128(n);

        for _ in 0..n_iter {
            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_gf128_pinned(&mut f, &mut g, &challenges, true);
            samples_fused.push(t.elapsed().as_secs_f64() * 1e6);

            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_gf128_pinned(&mut f, &mut g, &challenges, false);
            samples_unfused.push(t.elapsed().as_secs_f64() * 1e6);

            let mut f = f_orig.clone();
            let mut g = g_orig.clone();
            let t = Instant::now();
            sumcheck_deg2_delayed_gf128_pinned_v0(&mut f, &mut g, &challenges);
            samples_legacy.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }

    print_row(&format!("{field} NEW_FUSED"), n, &mut samples_fused);
    print_row(&format!("{field} NEW_UNFUSED"), n, &mut samples_unfused);
    print_row(&format!("{field} LEGACY"), n, &mut samples_legacy);
}
