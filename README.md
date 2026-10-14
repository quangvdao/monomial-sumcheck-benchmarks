# monomial-sumcheck-benchmarks

Benchmark harness for the paper *The Sum-Check Protocol over the Monomial Basis, and Other Optimizations*.

## Benchmark suites

| Suite | File | What it measures |
|---|---|---|
| `field_ops` | `benches/field_ops.rs` | Field add/sub/mul latency and throughput across BN254, Fp128, BabyBear extensions, GF(2^128); latency chains run over 2^20 random operands |
| `binding` | `benches/binding.rs` | Upper-limb multiplication and combined in-place binding (paper, Section 6.1) |
| `lookup_tables` | `benches/lookup_tables.rs` | Full-domain EQ and LT table construction |
| `sumcheck` | `benches/sumcheck.rs` | Complete evaluation-and-binding prover kernels for degree-2 and degree-2 x eq sum-check, with precomputed challenges |

The `binding`, `lookup_tables`, and `sumcheck` suites also run BN254 in a branch-free configuration (`Bn254BranchFree`, IDs containing `bf`): branch-free addition and subtraction with arkworks' multiplication, so that mispredicted branches in arkworks' add/sub do not enter the Boolean/projective comparison.

## Code layout

- `src/lib.rs`: library entry point for shared benchmark code
- `src/lookup_tables.rs`: EQ/LT table builders and closed-form evaluators shared by benchmarks and tests
- `src/sumcheck/mod.rs`: sumcheck module surface and re-exports used by benches and tests
- `src/sumcheck/data.rs`: synthetic input builders and suffix-EQ table helpers
- `src/sumcheck/generic.rs`: field-agnostic boolean and projective sumcheck kernels
- `src/sumcheck/bn254.rs`: BN254 delayed accumulators, the branch-free add/sub element type, and upper-limb challenge variants
- `src/sumcheck/fp128.rs`: Fp128 kernels, delayed accumulators, and the `{1,inf}` kernels. With the tables on `{1,inf}`, the value at 1 and the leading coefficient of a round polynomial are tied by the round claim, so the `{1,inf}` equality kernels compute the value at 0 and the leading coefficient
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
| `jolt-field` (features `solinas`, `asm`) | [a16z/jolt](https://github.com/a16z/jolt) | `8b6099f70bb21c8d38abd32dc295d34786c1699a` |
| `binius-field` | [binius-zk/binius64](https://github.com/binius-zk/binius64) | `6a69077efb40ee3d09e37e1c9f3511e2a9f75c99` |
| `ark-bn254`, `ark-ff` | [crates.io arkworks](https://crates.io/crates/ark-ff) | `0.5` |

The upper-limb BN254 multiplication helper used by these benchmarks is implemented locally in `src/sumcheck/bn254.rs`.

Rust toolchain: `1.94.0` (pinned in `rust-toolchain.toml`).

## Reproducibility

- Benchmark structure is reproducible across machines, but absolute timings depend on CPU, OS, and target architecture.
- The paper's BN254 and Fp128 prover kernels were collected on an Apple M4 Max, and everything else on an Apple M4 (`aarch64-apple-darwin`), single-threaded, with thin LTO enabled. Each speedup compares two cases measured on the same machine. Development smoke runs are not paper data.
- Every suite uses the same five-second warm-up, ten-second measurement window, 100 samples, and 99% Criterion confidence level.
- The combined binding benchmark (`bench_combined` in `binding.rs`) follows Jolt's `bound_poly_var_top` layout: in-place binding on a contiguous 2N-element array. Criterion prepares the working buffer outside the timed routine, so no separately measured copy time is subtracted.
- The paper uses ten outer repetitions for the BN254 and Fp128 prover kernels and for the binding and table-construction suites, and five for the field arithmetic and for the BabyBear, KoalaBear, and GF(2^128) prover kernels. Each compared case runs in its own fresh process. The repository reports 99% bootstrap intervals and empirical p01/p99 process-level quantiles, with speedups paired by outer repetition.
- The collector records system-load snapshots and refuses paper-data collection when the one-minute load average per logical CPU exceeds a threshold (`--max-load-per-cpu`, default 0.25; the paper used 0.4 on the M4 Max and 0.6 on the M4). `--allow-busy` is for non-publishable smoke tests only.
- Each projective kernel for BabyBear and KoalaBear extensions in `src/sumcheck/` is a line-for-line mirror of its Boolean baseline, differing only in sums versus differences, so neither variant gets a code shape the other lacks (see [docs/code-shaping.md](docs/code-shaping.md)). `benches/sumcheck.rs` is intentionally kept as thin wiring around those kernels. Keep the two variants mirrored when changing either.

See [docs/benchmark-methodology.md](docs/benchmark-methodology.md) for the measurement boundary, collection protocol, statistical procedure, and artifact-acceptance rules.

The reference environments are an Apple M4 Max and an Apple M4, both running
macOS 27.0.1 (build 26A434), with Rust 1.94.0 and LLVM 21.1.8 targeting
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

# Reproduce the combined binding table (paper, Section 6.1)
cargo bench --bench binding -- 'combined'

# Collect ten independent combined-binding processes with raw samples
# (the collector requires a clean committed tree)
python3 scripts/collect.py \
  --suite binding \
  --case BN254/combined_boolean_full \
  --case BN254/combined_projective_full \
  --case BN254/combined_boolean_upper \
  --case BN254/combined_projective_upper \
  --repetitions 10 \
  --max-load-per-cpu 0.4 \
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

## Paper data

`artifacts/data/` holds every run directory behind the paper's tables and
numbers, unmodified: `m4-runs/` from the Apple M4 and `m4max-runs/` from the
Apple M4 Max. Each directory has the collector's manifest (commit, toolchain,
OS build, CPU, commands, load snapshots) and, per outer repetition, the cargo
logs, the case order, the load records, and Criterion's raw output.
`docs/paper-runs.txt` lists the directories used. To regenerate the paper's
tables and macros:

```bash
cd artifacts/data
python3 ../../scripts/paper_tables.py --out /tmp/paper-tables \
  $(grep -v '^#' ../../docs/paper-runs.txt)
```
