use std::time::Duration;

use ark_ff::AdditiveGroup as _;
use binius_field::Field as _;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::{AdditiveGroup as _, FieldCore as _};
use p3_field::PrimeCharacteristicRing as _;
use rand::seq::SliceRandom;

use monomial_sumcheck_benchmarks::sumcheck::*;

fn bench_bn254(c: &mut Criterion) {
    let ns = [16u32, 20, 24];

    {
        let mut group = c.benchmark_group("sumcheck_deg2/BN254");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let challenges = make_bn254(n_usize);

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_boolean(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        BN254Fr::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_bn254(&mut f, &mut g, &challenges);
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        BN254Fr::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_delayed_bn254(
                                        &mut f,
                                        &mut g,
                                        &challenges,
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

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/BN254");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let challenges = make_bn254(n_usize);
            let eq_point = make_bn254(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, BN254Fr::from(1u64));

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_boolean(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        BN254Fr::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_delayed_bn254(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_projective(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        BN254Fr::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_projective_delayed_bn254(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
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
}

fn bench_bn254_upper(c: &mut Criterion) {
    let ns = [16u32, 20, 24];

    {
        let mut group = c.benchmark_group("sumcheck_deg2/BN254_upper");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let (_, challenge_limbs) = make_bn254_upper_limb_challenges(n_usize);

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_boolean_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_delayed_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_delayed_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &challenge_limbs,
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

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/BN254_upper");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_bn254(1usize << n_usize);
            let g_orig = make_bn254(1usize << n_usize);
            let (eq_point, challenge_limbs) = make_bn254_upper_limb_challenges(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, BN254Fr::from(1u64));

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_boolean_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_delayed_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_projective_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenge_limbs,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_projective_delayed_bn254_upper(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenge_limbs,
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
}

macro_rules! bench_bb_field {
    ($field_label:expr, $field:ty, $make:ident, $zero:expr, $one:expr,
     $delayed_fn:ident, $delayed_eq_fn:ident,
     $proj_delayed_fn:ident, $proj_delayed_eq_fn:ident,
     $eq_boolean_fn:ident, $eq_projective_fn:ident,
     $ns:expr, $c:expr) => {{
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make(1usize << n_usize);
                let g_orig = $make(1usize << n_usize);
                let challenges = $make(n_usize);

                let mut order = [0usize, 1, 2, 3];
                order.shuffle(&mut rand::thread_rng());
                for &idx in &order {
                    match idx {
                        0 => {
                            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_boolean(&mut f, &mut g, &challenges, $zero);
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        1 => {
                            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $delayed_fn(&mut f, &mut g, &challenges);
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        2 => {
                            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_projective(
                                            &mut f,
                                            &mut g,
                                            &challenges,
                                            $zero,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        3 => {
                            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $proj_delayed_fn(&mut f, &mut g, &challenges);
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
        {
            let mut group = $c.benchmark_group(concat!("sumcheck_deg2_eq/", $field_label));
            for &n in &$ns {
                let n_usize = n as usize;
                let f_orig = $make(1usize << n_usize);
                let g_orig = $make(1usize << n_usize);
                let challenges = $make(n_usize);
                let eq_point = $make(n_usize);
                let suffix_eq = build_suffix_eq_tables(&eq_point, $one);

                let mut order = [0usize, 1, 2, 3];
                order.shuffle(&mut rand::thread_rng());
                for &idx in &order {
                    match idx {
                        0 => {
                            group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $eq_boolean_fn(
                                            &mut f,
                                            &mut g,
                                            &suffix_eq,
                                            &challenges,
                                            $zero,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        1 => {
                            group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $delayed_eq_fn(&mut f, &mut g, &suffix_eq, &challenges);
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        2 => {
                            group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $eq_projective_fn(
                                            &mut f,
                                            &mut g,
                                            &suffix_eq,
                                            &challenges,
                                            $zero,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            });
                        }
                        3 => {
                            group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        $proj_delayed_eq_fn(
                                            &mut f,
                                            &mut g,
                                            &suffix_eq,
                                            &challenges,
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
    }};
}

fn bench_bb4(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_bb_field!(
        "BB4",
        BB4,
        make_bb4,
        BB4::ZERO,
        BB4::ONE,
        sumcheck_deg2_delayed_bb4,
        sumcheck_deg2_eq_delayed_bb4,
        sumcheck_deg2_projective_delayed_bb4,
        sumcheck_deg2_eq_projective_delayed_bb4,
        sumcheck_deg2_eq_gruen_boolean,
        sumcheck_deg2_eq_gruen_projective,
        ns,
        c
    );
}

fn bench_bb5(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_bb_field!(
        "BB5",
        BB5,
        make_bb5,
        BB5::ZERO,
        BB5::ONE,
        sumcheck_deg2_delayed_bb5,
        sumcheck_deg2_eq_delayed_bb5_packed,
        sumcheck_deg2_projective_delayed_bb5,
        sumcheck_deg2_eq_projective_delayed_bb5_packed,
        sumcheck_deg2_eq_gruen_boolean_bb5,
        sumcheck_deg2_eq_gruen_projective_bb5,
        ns,
        c
    );
}

fn bench_kb5(c: &mut Criterion) {
    let ns = [16u32, 20];
    bench_bb_field!(
        "KB5",
        KB5,
        make_kb5,
        KB5::ZERO,
        KB5::ONE,
        sumcheck_deg2_delayed_kb5,
        sumcheck_deg2_eq_delayed_kb5,
        sumcheck_deg2_projective_delayed_kb5,
        sumcheck_deg2_eq_projective_delayed_kb5,
        sumcheck_deg2_eq_gruen_boolean,
        sumcheck_deg2_eq_gruen_projective,
        ns,
        c
    );
}

fn bench_fp128(c: &mut Criterion) {
    let ns = [16u32, 20];
    {
        let mut group = c.benchmark_group("sumcheck_deg2/Fp128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_fp128(1usize << n_usize);
            let g_orig = make_fp128(1usize << n_usize);
            let challenges = make_fp128(n_usize);

            let mut order = [0usize, 1, 2, 3, 4, 5];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_boolean(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        Fp128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
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
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_fp128(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        Fp128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(
                            BenchmarkId::new("projective_1inf", n),
                            &n,
                            |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_projective_1inf_fp128(
                                            &mut f,
                                            &mut g,
                                            &challenges,
                                            Fp128::ZERO,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            },
                        );
                    }
                    4 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_delayed_fp128(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    5 => {
                        group.bench_with_input(
                            BenchmarkId::new("projective_1inf_delayed", n),
                            &n,
                            |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_projective_1inf_delayed_fp128(
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
                    _ => unreachable!(),
                }
            }
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/Fp128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_fp128(1usize << n_usize);
            let g_orig = make_fp128(1usize << n_usize);
            let challenges = make_fp128(n_usize);
            let eq_point = make_fp128(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, Fp128::one());
            let initial_claim_1inf = init_sumcheck_deg2_eq_gruen_projective_1inf_fp128_claim(
                &f_orig,
                &g_orig,
                &suffix_eq,
                &eq_point,
                Fp128::ZERO,
            );

            let mut order = [0usize, 1, 2, 3, 4, 5];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_boolean(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        Fp128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_delayed_fp128(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_projective_fp128(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        Fp128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(
                            BenchmarkId::new("projective_1inf", n),
                            &n,
                            |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_eq_gruen_projective_1inf_fp128(
                                            &mut f,
                                            &mut g,
                                            &suffix_eq,
                                            &eq_point,
                                            &challenges,
                                            initial_claim_1inf,
                                            Fp128::ZERO,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            },
                        );
                    }
                    4 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_projective_delayed_fp128(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    5 => {
                        group.bench_with_input(
                            BenchmarkId::new("projective_1inf_delayed", n),
                            &n,
                            |b, _| {
                                b.iter_batched(
                                    || (f_orig.clone(), g_orig.clone()),
                                    |(mut f, mut g)| {
                                        sumcheck_deg2_eq_projective_1inf_delayed_fp128(
                                            &mut f,
                                            &mut g,
                                            &suffix_eq,
                                            &eq_point,
                                            &challenges,
                                            initial_claim_1inf,
                                        );
                                    },
                                    criterion::BatchSize::LargeInput,
                                )
                            },
                        );
                    }
                    _ => unreachable!(),
                }
            }
        }
        group.finish();
    }
}

fn bench_gf128(c: &mut Criterion) {
    let ns = [16u32, 20];
    {
        let mut group = c.benchmark_group("sumcheck_deg2/GF128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_gf128(1usize << n_usize);
            let g_orig = make_gf128(1usize << n_usize);
            let challenges = make_gf128(n_usize);

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_boolean(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        GF128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
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
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective(
                                        &mut f,
                                        &mut g,
                                        &challenges,
                                        GF128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_projective_delayed_gf128(
                                        &mut f,
                                        &mut g,
                                        &challenges,
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

    {
        let mut group = c.benchmark_group("sumcheck_deg2_eq/GF128");
        for &n in &ns {
            let n_usize = n as usize;
            let f_orig = make_gf128(1usize << n_usize);
            let g_orig = make_gf128(1usize << n_usize);
            let challenges = make_gf128(n_usize);
            let eq_point = make_gf128(n_usize);
            let suffix_eq = build_suffix_eq_tables(&eq_point, GF128::ONE);

            let mut order = [0usize, 1, 2, 3];
            order.shuffle(&mut rand::thread_rng());
            for &idx in &order {
                match idx {
                    0 => {
                        group.bench_with_input(BenchmarkId::new("boolean", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_boolean(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        GF128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    1 => {
                        group.bench_with_input(BenchmarkId::new("delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_delayed_gf128(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    2 => {
                        group.bench_with_input(BenchmarkId::new("projective", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_gruen_projective(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
                                        GF128::ZERO,
                                    );
                                },
                                criterion::BatchSize::LargeInput,
                            )
                        });
                    }
                    3 => {
                        group.bench_with_input(BenchmarkId::new("proj_delayed", n), &n, |b, _| {
                            b.iter_batched(
                                || (f_orig.clone(), g_orig.clone()),
                                |(mut f, mut g)| {
                                    sumcheck_deg2_eq_projective_delayed_gf128(
                                        &mut f,
                                        &mut g,
                                        &suffix_eq,
                                        &challenges,
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
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(10));
    targets = bench_bn254, bench_bn254_upper, bench_bb4, bench_bb5, bench_kb5, bench_fp128, bench_gf128
}
criterion_main!(benches);
