# monomial-sumcheck-benchmarks

This repo contains the benchmark harness used to generate the field operation, binding, lookup table, and end-to-end sumcheck measurements for the monomial sumcheck paper.

## Reproducibility

- The Rust toolchain is pinned in `rust-toolchain.toml`.
- `hachi-pcs` is pinned to `LayerZero-Labs/hachi` commit `7e81702c87bd7adb9caeb7cb5064d65e16f740ff`, which contains the Fp128 arithmetic used by these benchmarks.
- Benchmark structure is reproducible across machines, but absolute timings depend on CPU, OS, and target architecture.
- The paper numbers were collected on Apple Silicon (`aarch64-apple-darwin`).
- Other machines should be able to run the same harness, but should not expect identical wall-clock times.

## Sanity Check

```bash
cargo check --benches
```

## Run All Benchmark Suites

```bash
cargo bench --bench field_ops
cargo bench --bench binding
cargo bench --bench lookup_tables
cargo bench --bench sumcheck
```

## Reproduce The Paper's Sumcheck Rows

The paper reports the end-to-end sumcheck numbers for `n = 20`.
To rerun that subset:

```bash
cargo bench --bench sumcheck -- 'sumcheck_deg2/.*/20|sumcheck_deg2_eq/.*/20'
```

## Output

Criterion writes reports to `target/criterion/`.
If `gnuplot` is not installed, Criterion falls back to the plotters backend.
That changes the plotting backend, not the benchmark measurements.
