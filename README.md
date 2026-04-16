# monomial-sumcheck-benchmarks

Benchmark harness for the paper *The Sum-Check Protocol over the Monomial Basis, and Other Optimizations*.

## Benchmark suites

| Suite | File | What it measures |
|---|---|---|
| `field_ops` | `benches/field_ops.rs` | Field add/sub/mul latency and throughput across BN254, Fp128, BabyBear extensions, GF(2^128) |
| `binding` | `benches/binding.rs` | Per-element binding latency, upper-limb multiplication, and combined in-place binding (Section 6.4) |
| `lookup_tables` | `benches/lookup_tables.rs` | Full-domain EQ and LT table construction |
| `sumcheck` | `benches/sumcheck.rs` | Thin Criterion harness for the end-to-end sum-check prover (degree-2 and degree-2 x eq), backed by shared kernels in `src/sumcheck/` |

## Code layout

- `src/lib.rs`: library entry point for shared benchmark code
- `src/sumcheck/mod.rs`: sumcheck module surface and re-exports used by benches and tests
- `src/sumcheck/data.rs`: synthetic input builders and suffix-EQ table helpers
- `src/sumcheck/generic.rs`: field-agnostic boolean and projective sumcheck kernels
- `src/sumcheck/bn254.rs`: BN254 delayed accumulators and upper-limb challenge variants
- `src/sumcheck/fp128.rs`: Fp128 kernels, delayed accumulators, and `1/inf` experiments
- `src/sumcheck/bb_ext.rs`: BB4, BB5, and KB5 delayed accumulators and bind helpers
- `src/sumcheck/bb5_packed.rs`: packed BB5 EQ evaluators and wrapper paths
- `src/sumcheck/gf128.rs`: GF(2^128) delayed kernels
- `tests/bb5_packed_eq.rs`: integration tests that import the shared library symbols directly

## Pinned dependencies

All dependencies are pinned to exact git revisions for reproducibility.

| Crate | Repository | Commit |
|---|---|---|
| `p3-baby-bear`, `p3-koala-bear`, `p3-field` | [Plonky3/Plonky3](https://github.com/Plonky3/Plonky3) | `b482e1be5f6d2e0917c5ecea3009335bbfd94e42` |
| `hachi-pcs` | [LayerZero-Labs/hachi](https://github.com/LayerZero-Labs/hachi) | `7e81702c87bd7adb9caeb7cb5064d65e16f740ff` |
| `binius-field` | [binius-zk/binius64](https://github.com/binius-zk/binius64) | `6a69077efb40ee3d09e37e1c9f3511e2a9f75c99` |
| `ark-bn254`, `ark-ff` | [quangvdao/arkworks-algebra](https://github.com/quangvdao/arkworks-algebra) (fork with `mul_by_hi_2limbs`) | `8221f4df9673c59b7f1bde82f483d73a36d5b00f` |

Rust toolchain: `1.94.0` (pinned in `rust-toolchain.toml`).

## Reproducibility

- Benchmark structure is reproducible across machines, but absolute timings depend on CPU, OS, and target architecture.
- The paper numbers were collected on an Apple M4 Max (`aarch64-apple-darwin`), single-threaded, with thin LTO enabled.
- The combined binding benchmark (`bench_combined` in `binding.rs`) follows Jolt's `bound_poly_var_top` layout: in-place binding on a contiguous 2N-element array. Each iteration restores the buffer via `memcpy`; reported times in the paper subtract the measured copy overhead.
- Some hot kernels in `src/sumcheck/` intentionally keep non-obvious source ordering, helper splitting, or duplicate reloads because those shapes produce measurably better LLVM codegen on the M4/NEON target. `benches/sumcheck.rs` is intentionally kept as thin wiring around those kernels. Re-benchmark those paths before simplifying them for style.

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

# Reproduce the paper's end-to-end sumcheck rows (n = 20)
cargo bench --bench sumcheck -- 'sumcheck_deg2/.*/20|sumcheck_deg2_eq/.*/20'

# Focus on the most codegen-sensitive degree-2 x eq rows while iterating
cargo bench --bench sumcheck -- 'sumcheck_deg2_eq/(BB5|Fp128|GF128)/(boolean|projective|delayed|proj_delayed)/20'

# Reproduce the combined binding table (Section 6.4)
cargo bench --bench binding -- 'combined'
```

## Output

Criterion writes reports to `target/criterion/`.
If `gnuplot` is not installed, Criterion falls back to the plotters backend.
That changes the plotting backend, not the benchmark measurements.
