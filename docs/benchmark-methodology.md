# Benchmark methodology

This document defines how to reproduce performance claims for the paper
The Sum-Check Protocol over the Monomial Basis, and Other Optimizations.

## Measurement scope

The field-operation suites measure isolated arithmetic kernels. The binding
suite measures in-place updates over a contiguous coefficient table. Criterion
creates a fresh table outside the timed routine, so the reported time includes
the binding loop and excludes restoration.

The sum-check suite measures the complete evaluation-and-binding prover kernel
with precomputed challenges and precomputed suffix equality tables. It includes round-polynomial evaluation and
destructive binding in every round. It excludes transcript hashing, message
serialization, challenge derivation, and polynomial-commitment openings.
Paper text must call this measurement a complete prover kernel, not an
end-to-end prover.

## Inputs

Tables, challenges, and equality points are pseudorandom and deterministic.
Every table comes from its own SplitMix64 stream, so the two tables of an
instance are independent. BN254 elements are full-field: 256 pseudorandom bits
reduced modulo the field order. Fp128 and GF(2^128) elements come from 128
pseudorandom bits. Each BabyBear or KoalaBear coordinate is a 32-bit draw
reduced modulo the prime, which is close to uniform but not exactly uniform.
The upper-limb BN254 challenges are 125-bit values. Compared variants receive
the same inputs.

## Controlled configuration

Every suite uses the shared configuration in src/lib.rs:

- five-second warm-up;
- ten-second measurement window;
- 100 Criterion samples; and
- 99% Criterion confidence level.

The benchmark suites may randomize variant registration, but accepted paired
data do not depend on shared-process ordering: the collector selects one exact
case per Cargo process.

## Independent repetitions

The BN254 and Fp128 prover kernels and the binding and table-construction
suites use ten independent outer repetitions; the per-operation field table
and the prover kernels over the BabyBear and KoalaBear extensions and
GF(2^128) use five. Each compared
case runs as the only selected benchmark in a fresh Cargo process. The
collector pairs cases by outer repetition and cyclically balances their order,
so no variant systematically inherits the first-benchmark penalty. Collect a
four-way binding comparison with:

    python3 scripts/collect.py \
      --suite binding \
      --case BN254/combined_boolean_full \
      --case BN254/combined_projective_full \
      --case BN254/combined_boolean_upper \
      --case BN254/combined_projective_upper \
      --repetitions 10 \
      --max-load-per-cpu 0.4 \
      --label combined

For a degree-2 BN254 comparison at n=20, run:

    python3 scripts/collect.py \
      --suite sumcheck \
      --case sumcheck_deg2/BN254/delayed/20 \
      --case sumcheck_deg2/BN254/proj_delayed/20 \
      --repetitions 10 \
      --max-load-per-cpu 0.4 \
      --label bn254-deg2-n20

The collector writes a machine manifest, per-repetition command order, command
output, system-load snapshots, Criterion estimates, and Criterion samples. It
does not overwrite an existing run. A raw `--filter` remains available for
exploratory measurements, but headline comparisons use repeated exact `--case`
arguments so variants do not share one Criterion process.

The collector refuses a dirty source tree, and it checks before every
repetition that the commit and the working tree have not changed. It also
refuses a busy host: it compares the one-minute load average per logical CPU
with a threshold before and after every case, and a repetition that exceeds
the threshold is marked failed and ends the collection. The reported data use
a threshold of 0.4 on the M4 Max and 0.6 on the M4 (`--max-load-per-cpu`); the
default is 0.25.
`--allow-dirty` and `--allow-busy` are reserved for smoke tests, and such a
run is not eligible for `artifacts/data`.

## Statistical summary

Summaries use each process's Criterion median as one independent observation.
For each benchmark, report:

- the median across processes;
- a 99% percentile-bootstrap confidence interval for that median;
- the observed minimum and maximum;
- the empirical p01 and p99 process-level quantiles (descriptive order
  statistics, not high-confidence tail-latency estimates); and
- the number of independent processes.

The summarizer warns when a benchmark has fewer processes than
`--min-processes` (default 20); pass the repetition count of the run.

Compute speedups within each outer repetition, pairing the fresh-process
observations for the baseline and optimized case. Do not divide
confidence-interval endpoints from two unpaired measurements.

For the combined binding run, first inspect the benchmark identifiers:

    python3 scripts/summarize.py artifacts/runs/RUN_DIRECTORY

Then generate a paired summary:

    python3 scripts/summarize.py artifacts/runs/RUN_DIRECTORY \
      --pair projective=BN254_combined_boolean_full,BN254_combined_projective_full \
      --pair combined=BN254_combined_boolean_full,BN254_combined_projective_upper \
      --min-processes 10 \
      --json artifacts/runs/RUN_DIRECTORY/summary.json \
      --markdown artifacts/runs/RUN_DIRECTORY/summary.md \
      --latex artifacts/runs/RUN_DIRECTORY/summary.tex

Move a completed and reviewed run to artifacts/data before using it in the
paper. Preserve the raw files and generate the LaTeX table from the accepted
summary. Do not transcribe table entries manually.

## Paper tables

`scripts/paper_tables.py` writes every table body and every number quoted in
the paper (`tab_*.tex` and `numbers.tex`) from a list of run directories:

    cd artifacts/data
    python3 ../../scripts/paper_tables.py --out OUT_DIR \
      $(grep -v '^#' ../../docs/paper-runs.txt)

where `artifacts/data` holds the `m4-runs` and `m4max-runs` directories. If two directories contain the same case, the later one is used, and a
speedup is computed only from two cases of the same run directory. The
uncertainty table uses the same paired ratios as `summarize.py`.

## Machine state

The manifest records the collector's full command line, the load threshold,
whether `--allow-busy` or `--allow-dirty` was given, the Git commit, dirty
status, Rust and LLVM versions, target triple, operating-system build, CPU,
initial system load, benchmark commands, and the compiler flags in `RUSTFLAGS`
and `CARGO_ENCODED_RUSTFLAGS`. Each repetition additionally records
load snapshots before and after every case. Record the machine's power mode and
whether it was connected to power in the accepted dataset's README because
those settings are not available through a portable command.

Run one collector at a time. Stop unrelated CPU-intensive work and keep the
machine on AC power. If thermal throttling or an interrupted process is
observed, discard the entire affected process run rather than deleting
individual samples.

The paper's BN254 and Fp128 prover kernels run on an Apple M4 Max; the other
prover kernels and the field-arithmetic, binding, and table-construction suites
run on an Apple M4. Both run macOS 27.0.1 (build 26A434), Rust 1.94.0, LLVM
21.1.8, and the `aarch64-apple-darwin` target. The BN254 prover kernels and the
field arithmetic of every field other than Fp128 come from commit 6d7f6c5; the
equality-factor prover kernels over the BabyBear and KoalaBear extensions come
from 3dc0e5d; everything else comes from 77a25a5. Commit 77a25a5 changes only
the source of the Fp128 implementation, and 3dc0e5d changes only the
accumulation order in those projective kernels (see
[`code-shaping.md`](code-shaping.md)). Treat a toolchain or operating-system change as
a new experimental environment; do not merge its repetitions into an existing
data set.
