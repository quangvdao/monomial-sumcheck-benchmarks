//! Parallelism shootout bench.
//!
//! Measures the delayed sumcheck kernel for GF128 and Fp128 across all three
//! approaches in the same process, so Criterion produces a direct comparison
//! table against the sequential baseline.
//!
//! Variants:
//!
//! - `delayed`                 : sequential baseline (same function as
//!                               `benches/sumcheck.rs`).
//! - `delayed_par1_scope`      : manual chunked `rayon::scope` (Approach 1).
//! - `delayed_par1_pariter`    : `par_iter` control to isolate manual-scope
//!                               vs par_iter dispatch overhead.
//! - `delayed_par2_chili_BASE` : chili recursive `scope.join` (Approach 2),
//!                               swept across base-case thresholds.
//!                               (only compiled with `--features parallel_chili`)
//! - `delayed_par3_persistent` : persistent pool + atomic barrier (Approach 3).
//! - `delayed_par4_pinned`     : globally-persistent pinned pool + doorbell
//!                               (Approach 4).
//!
//! Sizes sweep `n ∈ {10, 12, 14, 16, 18, 20}` so the per-round `half` covers
//! both the "parallel obviously loses" and "parallel obviously wins" regimes.
//!
//! The `dispatch_floor` group at the top is a calibration microbench for
//! Rayon's per-dispatch cost. This anchors all crossover interpretation.
//!
//! Usage:
//!   cargo bench --bench sumcheck_parallel --features parallel -- \
//!     'dispatch_floor|GF128/delayed|Fp128/delayed' \
//!     --warm-up-time 2 --measurement-time 5
//!
//!   # To also include chili:
//!   cargo bench --bench sumcheck_parallel --features parallel_chili -- ...

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::seq::SliceRandom;

use monomial_sumcheck_benchmarks::sumcheck::*;

/// Per-field bench sweep of sizes.
const NS: [u32; 6] = [10, 12, 14, 16, 18, 20];

/// Base-case parameters for the Approach-2 chili sweep. Larger base = fewer
/// fork/join levels and thus fewer handoffs but worse load balance.
#[cfg(feature = "parallel_chili")]
const CHILI_BASES: [usize; 4] = [32, 128, 512, 2048];

fn bench_dispatch_floor(c: &mut Criterion) {
    use std::hint::black_box;
    use std::sync::atomic::{AtomicU64, Ordering};

    use rayon::prelude::*;

    let mut group = c.benchmark_group("dispatch_floor");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));

    group.bench_function("par_iter_sum_1", |b| {
        b.iter(|| {
            let s: u64 = (0u64..1).into_par_iter().sum();
            black_box(s);
        });
    });

    group.bench_function("par_iter_sum_num_threads", |b| {
        let n = rayon::current_num_threads() as u64;
        b.iter(|| {
            let s: u64 = (0u64..n).into_par_iter().sum();
            black_box(s);
        });
    });

    group.bench_function("scope_spawn_num_threads_nop", |b| {
        let n = rayon::current_num_threads();
        let ctr = AtomicU64::new(0);
        b.iter(|| {
            rayon::scope(|s| {
                for _ in 0..n {
                    s.spawn(|_| {
                        ctr.fetch_add(1, Ordering::Relaxed);
                    });
                }
            });
            black_box(ctr.load(Ordering::Relaxed));
        });
    });

    #[cfg(feature = "parallel_chili")]
    {
        group.bench_function("chili_scope_join_noop", |b| {
            b.iter(|| {
                let mut scope = chili::Scope::global();
                let (a, bb): (u64, u64) =
                    scope.join(|_| black_box(0u64), |_| black_box(0u64));
                black_box(a.wrapping_add(bb));
            });
        });
    }

    // Dispatch cost for Approach 4's globally-persistent pinned pool.
    // Warm up the pool first so the lazy `OnceLock` init doesn't taint
    // the first sample. Sweep across active-worker counts so we can see
    // how the barrier cost scales with contention.
    let pool = monomial_sumcheck_benchmarks::sumcheck::PinnedPool::global();
    pool.broadcast_scoped(pool.n_workers(), &|_| {});
    for k in [2usize, 4, 8, 12, 16, 24, 32]
        .iter()
        .copied()
        .filter(|k| *k <= pool.n_workers())
    {
        let label = format!("pinned_pool_broadcast_nop_k{k}");
        group.bench_function(label, |b| {
            b.iter(|| {
                pool.broadcast_scoped(k, &|_idx: usize| {
                    black_box(());
                });
            });
        });
    }

    group.finish();
}

// Indices for the randomised run order per `n`.
// Base variants (in order): 0 delayed, 1 par1_scope, 2 par1_pariter,
// 3 par3_persistent, 4 par4_pinned. Chili bases (if enabled) follow.
#[cfg(feature = "parallel_chili")]
const N_GF128_VARIANTS: usize = 5 + CHILI_BASES.len();
#[cfg(not(feature = "parallel_chili"))]
const N_GF128_VARIANTS: usize = 5;

fn bench_gf128(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_deg2/GF128");

    for &n in &NS {
        let n_usize = n as usize;
        let f_orig = make_gf128(1usize << n_usize);
        let g_orig = make_gf128(1usize << n_usize);
        let challenges = make_gf128(n_usize);

        let mut order: Vec<usize> = (0..N_GF128_VARIANTS).collect();
        order.shuffle(&mut rand::thread_rng());
        for idx in order {
            match idx {
                0 => {
                    group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                        b.iter_batched(
                            || (f_orig.clone(), g_orig.clone()),
                            |(mut f, mut g)| {
                                sumcheck_deg2_delayed_gf128(&mut f, &mut g, &challenges);
                            },
                            criterion::BatchSize::LargeInput,
                        )
                    });
                }
                1 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par1_scope", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_gf128_par1_scope(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                2 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par1_pariter", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_gf128_par1_pariter(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                3 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par3_persistent", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_gf128_par3_persistent(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                4 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par4_pinned", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_gf128_par4_pinned(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                #[cfg(feature = "parallel_chili")]
                k if (5..5 + CHILI_BASES.len()).contains(&k) => {
                    let base = CHILI_BASES[k - 5];
                    let label = format!("delayed_par2_chili_b{base}");
                    group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                        b.iter_batched(
                            || (f_orig.clone(), g_orig.clone()),
                            |(mut f, mut g)| {
                                sumcheck_deg2_delayed_gf128_par2_chili(
                                    &mut f,
                                    &mut g,
                                    &challenges,
                                    base,
                                );
                            },
                            criterion::BatchSize::LargeInput,
                        )
                    });
                }
                _ => unreachable!(),
            }
        }
    }
    group.finish();
}

fn bench_fp128(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_deg2/Fp128");

    for &n in &NS {
        let n_usize = n as usize;
        let f_orig = make_fp128(1usize << n_usize);
        let g_orig = make_fp128(1usize << n_usize);
        let challenges = make_fp128(n_usize);

        let mut order: Vec<usize> = (0..N_GF128_VARIANTS).collect();
        order.shuffle(&mut rand::thread_rng());
        for idx in order {
            match idx {
                0 => {
                    group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                        b.iter_batched(
                            || (f_orig.clone(), g_orig.clone()),
                            |(mut f, mut g)| {
                                sumcheck_deg2_delayed_fp128(&mut f, &mut g, &challenges);
                            },
                            criterion::BatchSize::LargeInput,
                        )
                    });
                }
                1 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par1_scope", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_fp128_par1_scope(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                2 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par1_pariter", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_fp128_par1_pariter(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                3 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par3_persistent", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_fp128_par3_persistent(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                4 => {
                    group.bench_with_input(
                        BenchmarkId::new("delayed_par4_pinned", n),
                        &n,
                        |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_fp128_par4_pinned(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        },
                    );
                }
                #[cfg(feature = "parallel_chili")]
                k if (5..5 + CHILI_BASES.len()).contains(&k) => {
                    let base = CHILI_BASES[k - 5];
                    let label = format!("delayed_par2_chili_b{base}");
                    group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
                        b.iter_batched(
                            || (f_orig.clone(), g_orig.clone()),
                            |(mut f, mut g)| {
                                sumcheck_deg2_delayed_fp128_par2_chili(
                                    &mut f,
                                    &mut g,
                                    &challenges,
                                    base,
                                );
                            },
                            criterion::BatchSize::LargeInput,
                        )
                    });
                }
                _ => unreachable!(),
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(6));
    targets = bench_dispatch_floor, bench_gf128, bench_fp128
}
criterion_main!(benches);
