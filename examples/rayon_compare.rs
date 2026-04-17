//! Rayon-vs-pinned-pool sweep.
//!
//! For each `n` in `[10, 12, 14, 16, 18, 20]` (or `N_MIN..=N_MAX` if
//! the env vars are set), runs three implementations interleaved:
//!
//! - `pinned`      — our pinned doorbell pool (production)
//! - `rayon_scope` — `rayon::scope` (one scope per round)
//! - `rayon_iter`  — `par_iter().reduce()` (rayon's idiomatic API)
//!
//! Reports p10/p50/p90 wall time and the speed ratio of `pinned`
//! vs each rayon variant. The mandate from the productionization plan
//! is that `pinned` is faster (ratio < 1.00) at *every* `n`. Use
//! this example to validate that mandate after pool / D2 changes.
//!
//! ```text
//! cargo run --example rayon_compare --release --features parallel
//! N_MIN=14 N_MAX=22 N_ITER=200 FIELD=fp128 cargo run \
//!     --example rayon_compare --release --features parallel
//! ```
//!
//! Defaults: N_MIN=10, N_MAX=20, N_ITER auto-shrinks with n (we want
//! similar wall time per n; large n is per-call slower so we measure
//! fewer iterations).

#![cfg(feature = "parallel")]

use std::time::Instant;

use monomial_sumcheck_benchmarks::sumcheck::*;

fn percentiles(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| samples[((samples.len() as f64) * q) as usize];
    (p(0.10), p(0.50), p(0.90))
}

/// Auto-pick iteration count so total wall time per n stays roughly
/// constant. Sequential `delayed` at n=20 GF128 is ~100 ms on aragorn,
/// so 50 iters = 5 s. n=10 is ~50 µs, so 20000 iters = 1 s.
fn default_iters(n: usize) -> usize {
    match n {
        0..=10 => 4000,
        11..=12 => 2000,
        13..=14 => 1000,
        15..=16 => 400,
        17..=18 => 150,
        19..=20 => 60,
        _ => 30,
    }
}

#[derive(Clone, Copy)]
enum Field {
    Gf128,
    Fp128,
}

fn run_sweep(field: Field, n_min: usize, n_max: usize, n_iter_override: Option<usize>) {
    println!();
    let label = match field {
        Field::Gf128 => "GF128",
        Field::Fp128 => "Fp128",
    };
    println!("=== {label} sweep ===");
    println!(
        "{:>3}  {:>5}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "n", "iter", "pin50/µs", "scope50/µs", "pari50/µs", "vs_scope", "vs_pari",
    );
    for n in n_min..=n_max {
        let n_iter = n_iter_override.unwrap_or_else(|| default_iters(n));
        let row = match field {
            Field::Gf128 => sweep_one_gf128(n, n_iter),
            Field::Fp128 => sweep_one_fp128(n, n_iter),
        };
        // Ratios < 1 mean pinned wins.
        let vs_scope = row.scope_p50 / row.pin_p50;
        let vs_pari = row.pari_p50 / row.pin_p50;
        let scope_marker = if row.pin_p50 <= row.scope_p50 { " " } else { "!" };
        let pari_marker = if row.pin_p50 <= row.pari_p50 { " " } else { "!" };
        println!(
            "{:>3}  {:>5}  {:>10.2}  {:>10.2}  {:>10.2}  {:>7.2}x{} {:>7.2}x{}",
            n,
            n_iter,
            row.pin_p50,
            row.scope_p50,
            row.pari_p50,
            vs_scope,
            scope_marker,
            vs_pari,
            pari_marker,
        );
    }
}

struct Row {
    pin_p50: f64,
    scope_p50: f64,
    pari_p50: f64,
}

/// Run `n_iter` calls of `body` in a tight loop, returning per-call
/// wall times in microseconds. Each call rebuilds `f`/`g` from the
/// originals so we always measure the same workload.
///
/// We run each variant as a *burst* (not interleaved) to mirror
/// Criterion's `bench_with_input` measurement model: in production
/// nobody alternates `pinned` and `rayon::scope` per call, and
/// interleaving them deliberately creates QoS / scheduler thrash that
/// is not representative of a real workload. The burst structure also
/// matches `cargo bench --bench sumcheck_parallel`, so numbers here
/// should track Criterion within ~5%.
fn measure_burst<F, T>(n_iter: usize, mut clone: impl FnMut() -> T, mut body: F) -> Vec<f64>
where
    F: FnMut(T),
{
    let mut samples = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let inputs = clone();
        let t = Instant::now();
        body(inputs);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples
}

fn sweep_one_gf128(n: usize, n_iter: usize) -> Row {
    let f_orig = make_gf128(1usize << n);
    let g_orig = make_gf128(1usize << n);
    let challenges = make_gf128(n);
    let clone = || (f_orig.clone(), g_orig.clone());

    let mut pin = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_gf128_pinned(&mut f, &mut g, &challenges, false);
    });
    let mut scope = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_gf128_rayon_scope(&mut f, &mut g, &challenges);
    });
    let mut pari = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_gf128_rayon_iter(&mut f, &mut g, &challenges);
    });

    Row {
        pin_p50: percentiles(&mut pin).1,
        scope_p50: percentiles(&mut scope).1,
        pari_p50: percentiles(&mut pari).1,
    }
}

fn sweep_one_fp128(n: usize, n_iter: usize) -> Row {
    let f_orig = make_fp128(1usize << n);
    let g_orig = make_fp128(1usize << n);
    let challenges = make_fp128(n);
    let clone = || (f_orig.clone(), g_orig.clone());

    let mut pin = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_fp128_pinned(&mut f, &mut g, &challenges, false);
    });
    let mut scope = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_fp128_rayon_scope(&mut f, &mut g, &challenges);
    });
    let mut pari = measure_burst(n_iter, clone, |(mut f, mut g)| {
        sumcheck_deg2_delayed_fp128_rayon_iter(&mut f, &mut g, &challenges);
    });

    Row {
        pin_p50: percentiles(&mut pin).1,
        scope_p50: percentiles(&mut scope).1,
        pari_p50: percentiles(&mut pari).1,
    }
}

fn main() {
    let n_min: usize = std::env::var("N_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_max: usize = std::env::var("N_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_iter_override: Option<usize> =
        std::env::var("N_ITER").ok().and_then(|s| s.parse().ok());
    let field = std::env::var("FIELD").unwrap_or_else(|_| "both".into());

    eprintln!(
        "rayon_compare: n=[{n_min}..={n_max}] field={field} \
         pool_workers={} rayon_threads={}",
        PinnedPool::global().n_workers(),
        rayon::current_num_threads(),
    );
    // Warm both pools.
    PinnedPool::global().broadcast_scoped(PinnedPool::global().n_workers(), &|_| {});
    rayon::scope(|s| {
        for _ in 0..rayon::current_num_threads() {
            s.spawn(|_| {});
        }
    });

    match field.as_str() {
        "gf128" => run_sweep(Field::Gf128, n_min, n_max, n_iter_override),
        "fp128" => run_sweep(Field::Fp128, n_min, n_max, n_iter_override),
        _ => {
            run_sweep(Field::Gf128, n_min, n_max, n_iter_override);
            run_sweep(Field::Fp128, n_min, n_max, n_iter_override);
        }
    }

    println!();
    println!(
        "key: '!' marks rows where pinned is SLOWER than the rayon \
         variant (we want zero '!')"
    );
}
