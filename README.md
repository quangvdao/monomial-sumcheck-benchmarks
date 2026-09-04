# monomial-sumcheck-benchmarks

Benchmark harness for the paper *The Sum-Check Protocol over the Monomial Basis, and Other Optimizations*.

## Benchmark suites

| Suite | File | What it measures |
|---|---|---|
| `field_ops` | `benches/field_ops.rs` | Field add/sub/mul latency and throughput across BN254, Fp128, BabyBear extensions, GF(2^128) |
| `binding` | `benches/binding.rs` | Per-element binding latency, upper-limb multiplication, and combined in-place binding (Section 6.4) |
| `lookup_tables` | `benches/lookup_tables.rs` | Full-domain EQ and LT table construction |
| `sumcheck` | `benches/sumcheck.rs` | Complete evaluation-and-binding prover kernels for degree-2 and degree-2 x eq sum-check, with precomputed challenges |

## Code layout

- `src/lib.rs`: library entry point for shared benchmark code
- `src/lookup_tables.rs`: EQ/LT table builders and closed-form evaluators shared by benchmarks and tests
- `src/sumcheck/mod.rs`: sumcheck module surface and re-exports used by benches and tests
- `src/sumcheck/data.rs`: synthetic input builders and suffix-EQ table helpers
- `src/sumcheck/generic.rs`: field-agnostic boolean and projective sumcheck kernels
- `src/sumcheck/bn254.rs`: BN254 delayed accumulators and upper-limb challenge variants
- `src/sumcheck/fp128.rs`: Fp128 kernels, delayed accumulators, and `1/inf` experiments
- `src/sumcheck/bb_ext.rs`: BB4, BB5, and KB5 delayed accumulators and bind helpers
- `src/sumcheck/bb5_packed.rs`: packed BB5 EQ evaluators and wrapper paths
- `src/sumcheck/gf128.rs`: GF(2^128) delayed kernels
- `tests/bb5_packed_eq.rs`: integration tests that import the shared library symbols directly
- `tests/lookup_tables.rs`: basis-specific table-builder and closed-form equivalence tests
- `tests/upper_limb_challenge.rs`: transcript-byte mapping and specialized-multiplication tests

## Pinned dependencies

All dependencies are pinned by `Cargo.lock` for reproducibility.
Git dependencies are fixed to exact revisions.

| Crate | Source | Version or commit |
|---|---|---|
| `p3-baby-bear`, `p3-koala-bear`, `p3-field` | [Plonky3/Plonky3](https://github.com/Plonky3/Plonky3) | `b482e1be5f6d2e0917c5ecea3009335bbfd94e42` |
| `hachi-pcs` | [LayerZero-Labs/hachi](https://github.com/LayerZero-Labs/hachi) | `7e81702c87bd7adb9caeb7cb5064d65e16f740ff` |
| `binius-field` | [binius-zk/binius64](https://github.com/binius-zk/binius64) | `6a69077efb40ee3d09e37e1c9f3511e2a9f75c99` |
| `ark-bn254`, `ark-ff` | [crates.io arkworks](https://crates.io/crates/ark-ff) | `0.5` |

The upper-limb BN254 multiplication helper used by these benchmarks is implemented locally in `src/sumcheck/bn254.rs`.

Rust toolchain: `1.94.0` (pinned in `rust-toolchain.toml`).

## Reproducibility

- Benchmark structure is reproducible across machines, but absolute timings depend on CPU, OS, and target architecture.
- The accepted paper rerun is to be collected on an Apple M4 Max (`aarch64-apple-darwin`), single-threaded, with thin LTO enabled. Development smoke runs are not paper data.
- Every suite uses the same five-second warm-up, ten-second measurement window, 100 samples, and 99% Criterion confidence level.
- The combined binding benchmark (`bench_combined` in `binding.rs`) follows Jolt's `bound_poly_var_top` layout: in-place binding on a contiguous 2N-element array. Criterion prepares the working buffer outside the timed routine, so no separately measured copy time is subtracted.
- Headline results require at least 20 independent processes. The repository reports 99% bootstrap intervals and empirical p01/p99 process-level quantiles, with speedups computed from paired observations.
- Some hot kernels in `src/sumcheck/` intentionally keep non-obvious source ordering, helper splitting, or duplicate reloads because those shapes produce measurably better LLVM codegen on the M4/NEON target. `benches/sumcheck.rs` is intentionally kept as thin wiring around those kernels. Re-benchmark those paths before simplifying them for style.

See [docs/benchmark-methodology.md](docs/benchmark-methodology.md) for the measurement boundary, collection protocol, statistical procedure, and artifact-acceptance rules.

The reference environment is an Apple M4 Max running macOS 26.6.2 (build
25G83), with Rust 1.94.0 and LLVM 21.1.8 targeting
`aarch64-apple-darwin`. Every accepted run also stores its exact environment
manifest.

## Usage

```bash
# Sanity check
cargo check --benches
cargo test --test bb5_packed_eq

# Run all suites
cargo bench --bench field_ops
cargo bench --bench binding
cargo bench --bench lookup_tables
cargo bench --bench sumcheck

# Run the complete prover-kernel rows intended for the paper (n = 20)
cargo bench --bench sumcheck -- 'sumcheck_deg2/.*/20|sumcheck_deg2_eq/.*/20'

# Focus on the most codegen-sensitive degree-2 x eq rows while iterating
cargo bench --bench sumcheck -- 'sumcheck_deg2_eq/(BB5|Fp128|GF128)/(boolean|projective|delayed|proj_delayed)/20'

# Reproduce the combined binding table (Section 6.4)
cargo bench --bench binding -- 'combined'

# Collect 20 independent combined-binding processes with raw samples
# (the collector requires a clean committed tree)
python3 scripts/collect.py \
  --suite binding \
  --filter combined \
  --repetitions 20 \
  --label combined

# Summarize a completed run and list its benchmark identifiers
python3 scripts/summarize.py artifacts/runs/RUN_DIRECTORY

# During development only: permit a non-publishable one-process smoke run
python3 scripts/collect.py \
  --suite binding \
  --filter chained_mul_upper_limb \
  --repetitions 1 \
  --label smoke \
  --allow-dirty
```

## Output

Criterion writes reports to `target/criterion/`.
If `gnuplot` is not installed, Criterion falls back to the plotters backend.
That changes the plotting backend, not the benchmark measurements.
The process collector preserves raw data under `artifacts/runs/`; reviewed datasets used by the paper belong under `artifacts/data/`.
