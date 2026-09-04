# Benchmark methodology

This document defines how to reproduce performance claims for the paper
The Sum-Check Protocol over the Monomial Basis, and Other Optimizations.

## Measurement scope

The field-operation suites measure isolated arithmetic kernels. The binding
suite measures in-place updates over a contiguous coefficient table. Criterion
creates a fresh table outside the timed routine, so the reported time includes
the binding loop and excludes restoration.

The sum-check suite measures the complete evaluation-and-binding prover kernel
with precomputed challenges. It includes round-polynomial evaluation and
destructive binding in every round. It excludes transcript hashing, message
serialization, challenge derivation, and polynomial-commitment openings.
Paper text must call this measurement a complete prover kernel, not an
end-to-end prover.

## Controlled configuration

Every suite uses the shared configuration in src/lib.rs:

- five-second warm-up;
- ten-second measurement window;
- 100 Criterion samples; and
- 99% Criterion confidence level.

The sum-check and combined-binding suites randomize variant registration within
each process. This limits order and thermal bias when related variants run in
the same process.

## Independent repetitions

Headline claims use at least 20 independent benchmark processes. Collect them
with:

    python3 scripts/collect.py \
      --suite binding \
      --filter combined \
      --repetitions 20 \
      --label combined

For a degree-2 BN254 comparison at n=20, run:

    python3 scripts/collect.py \
      --suite sumcheck \
      --filter 'sumcheck_deg2/BN254/(delayed|proj_delayed)/20' \
      --repetitions 20 \
      --label bn254-deg2-n20

The collector writes a machine manifest, command output, Criterion estimates,
and Criterion samples for each process. It does not overwrite an existing run.
It refuses a dirty source tree by default. `--allow-dirty` is reserved for
smoke tests, and such a run is not eligible for `artifacts/data`.

## Statistical summary

Summaries use each process's Criterion median as one independent observation.
For each benchmark, report:

- the median across processes;
- a 99% percentile-bootstrap confidence interval for that median;
- the empirical p01 and p99 process-level quantiles (descriptive order
  statistics, not high-confidence tail-latency estimates); and
- the number of independent processes.

Compute speedups from paired observations collected in the same process. Do
not divide confidence-interval endpoints from two unpaired measurements.

For the combined binding run, first inspect the benchmark identifiers:

    python3 scripts/summarize.py artifacts/runs/RUN_DIRECTORY

Then generate a paired summary:

    python3 scripts/summarize.py artifacts/runs/RUN_DIRECTORY \
      --pair projective=BN254_combined_boolean_full,BN254_combined_projective_full \
      --pair combined=BN254_combined_boolean_full,BN254_combined_projective_upper \
      --json artifacts/runs/RUN_DIRECTORY/summary.json \
      --markdown artifacts/runs/RUN_DIRECTORY/summary.md \
      --latex artifacts/runs/RUN_DIRECTORY/summary.tex

Move a completed and reviewed run to artifacts/data before using it in the
paper. Preserve the raw files and generate the LaTeX table from the accepted
summary. Do not transcribe table entries manually.

## Machine state

The manifest records the Git commit, dirty status, Rust and LLVM versions,
target triple, operating-system build, CPU, benchmark command, and relevant
compiler flags. Record the machine's power mode and whether it was connected
to power in the accepted dataset's README because those settings are not
available through a portable command.

Run one collector at a time. Stop unrelated CPU-intensive work and keep the
machine on AC power. If thermal throttling or an interrupted process is
observed, discard the entire affected process run rather than deleting
individual samples.

The current reference environment is an Apple M4 Max running macOS 26.6.2
(build 25G83), Rust 1.94.0, LLVM 21.1.8, and the
`aarch64-apple-darwin` target. Treat a toolchain or operating-system change as
a new experimental environment; do not merge its repetitions into an existing
data set.
