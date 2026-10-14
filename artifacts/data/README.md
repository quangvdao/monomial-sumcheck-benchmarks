# Paper benchmark data

Every run directory behind the paper's tables and numbers, unmodified:
`m4-runs/` from the Apple M4 and `m4max-runs/` from the Apple M4 Max.
`docs/paper-runs.txt` lists the directories used, in the order passed to
`scripts/paper_tables.py`.

Each run directory, written by `scripts/collect.py`, contains

- `manifest.json`: benchmark commit, dirty-tree status, toolchain, OS build,
  CPU, collector arguments, commands, and the initial load;
- one `run-NN/` per outer repetition, with the cargo log of every case,
  `command-order.json`, `system-load.json`, `benchmarks.json`, and Criterion's
  raw output under `criterion/`.

To regenerate the paper's tables and macros (`tab_*.tex`, `numbers.tex`):

```bash
cd artifacts/data
python3 ../../scripts/paper_tables.py --out /tmp/paper-tables \
  $(grep -v '^#' ../../docs/paper-runs.txt)
```

To summarize a single run with bootstrap intervals:

```bash
python3 scripts/summarize.py artifacts/data/m4max-runs/RUN_DIRECTORY
```

Do not add a provisional or interrupted run here.
