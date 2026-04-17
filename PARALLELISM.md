# Parallelism for the delayed sumcheck kernels

Goal: figure out whether the per-round reduce-then-bind loop in the monomial
sumcheck kernels can be parallelised cheaply enough to beat the sequential
baseline at small problem sizes (`n ≈ 8-10`), not just at the paper's `n = 24`
headline point.

## Machines / configuration

Primary dev box (most results):

- Apple M4 Max, 16-core (12 P + 4 E), 64 GB RAM.
- macOS 25.4.0, release build with `lto = "thin"`, `rayon 1`, `chili 0.2.1`.
- Default rayon pool: 16 threads. No RAYON_NUM_THREADS pinning.
- `PinnedPool` workers tagged `QOS_CLASS_USER_INTERACTIVE` (macOS
  doesn't expose affinity; QoS is the canonical P-core anchor).

Linux validation box ("aragorn", added Phase A):

- AMD Ryzen 9 9950X, 16 physical cores × 2 SMT, 1 socket.
- Ubuntu 24.04.3, kernel 6.17. rustc 1.95.0.
- `PinnedPool` workers `sched_setaffinity`'d to one logical thread per
  physical core (canonical = lowest sibling from
  `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`). See
  `src/sumcheck/parallel/pool/platform.rs` and the **Aragorn
  validation** section below for numbers.

Common:

- `bench_args = --warm-up-time 2 --measurement-time 4` (100 samples)
  for the original prototype; **Phase A** (`bench-phaseA-2026-04.log`
  and `aragorn-bench-2026-04.log`) uses
  `--warm-up-time 3 --measurement-time 6` to fight variance.
- Criterion reports `[low median high]`; numbers below are **median**.
- `SUMCHECK_PINNED_WORKERS` unset (`pinned` picks `min(available_parallelism(), 8) = 8`).
- Phase-A code lives in `src/sumcheck/parallel/{pool,scheduler,field,impls}/`;
  the original prototype is preserved in `src/sumcheck/parallel/legacy.rs`
  (`rayon_scope`, `rayon_iter`, `chili`, `persistent`, `pinned_v0`) for
  historical comparison. `pinned`'s per-pair math
  is shared between the two via `pub(super) fn partial_triple_*` /
  `bind_chunk_*` so the inner loop body is byte-identical.

## Variants compared

| Bench label            | Variant         | Strategy                                                                                            |
|------------------------|-----------------|-----------------------------------------------------------------------------------------------------|
| `delayed`              | `seq`           | Sequential baseline (the existing hot kernel).                                                      |
| `delayed_rayon_scope`  | `rayon_scope`   | One `rayon::scope` per round; `n_workers` manual spawns; fused reduce+bind per worker.              |
| `delayed_rayon_iter`   | `rayon_iter`    | Control for `rayon_scope`: `(0..n_workers).into_par_iter()` + `par_chunks_mut`.                     |
| `delayed_chili_bK`     | `chili`         | Chili `scope.join` recursion with base-case `K` (sweep `K ∈ {32, 128, 512, 2048}`).                 |
| `delayed_persistent`   | `persistent`    | One `rayon::scope` for the whole call; persistent workers on ping-pong buffers; atomic-counter barriers with spin-then-yield. |
| `delayed_pinned`       | `pinned_v0`     | **Frozen A/B baseline** for `pinned`: single `broadcast_scoped`, direct `[SendPtr; 2]` captures, no D2, no trait. Same `PinnedPool`. |
| `delayed_pinned`       | **`pinned`**    | **Production.** Process-global `PinnedPool` of `std::thread` workers (macOS: `USER_INTERACTIVE` QoS, Linux: `sched_setaffinity`); per-call doorbell (epoch + done atomics); adaptive `n_active` via `PINNED_TARGET_PAIRS_PER_WORKER = 256`; D2 per-round shrinkage; driven by `sumcheck_parallel::par_sumcheck`. |

All parallel variants reuse the same `GF128Accum` / `Fp128Accum` accumulators
as the sequential kernel, so the hot per-pair loop body matches byte-for-byte.
Partial reductions sum correctly because `GF128Accum`'s XOR accumulation and
`Fp128Accum`'s `u128[4]` limb accumulation are both linear / ring-hom (see
`src/sumcheck/parallel/legacy.rs` module docs and
`src/sumcheck/parallel/scheduler.rs` for the field-agnostic scheduler).

## Dispatch floor (M4 Max, 16 threads / 8-worker pinned pool)

Two columns: original Approach-4 prototype with shared `epoch + done`
counters (`PARALLELISM.md` baseline) versus the Phase-A refactor with
per-worker split inbox/outbox cache lines (current code).

| Microbench                                         | Original   | Phase A (current) |
|----------------------------------------------------|------------|-------------------|
| `par_iter_sum_1` (effectively sequential)          | 2.6 ns     | 2.5 ns            |
| `par_iter_sum_num_threads`                         | 40.3 µs    | 20.5 µs           |
| `scope_spawn_num_threads_nop` (16 noop spawns)     | 23.3 µs    | 19.1 µs           |
| `chili_scope_join_noop` (one `join(noop, noop)`)   | 164 ns     | 164 ns            |
| `pinned_pool_broadcast_nop_k2` (2 active workers)  | 305 ns     | **131 ns**        |
| `pinned_pool_broadcast_nop_k4` (4 active workers)  | 443 ns     | **296 ns**        |
| `pinned_pool_broadcast_nop_k8` (8 active workers)  | 681 ns     | 1.20 µs           |

Two things to note:

1. Chili's fork/join is **~140× cheaper** than rayon's per-scope dispatch
   (164 ns vs 19-23 µs). That explains why `rayon_scope` (rayon scope per
   round) is unusable below `n = 18`: any per-round rayon scope pays
   ≥ 19 µs of overhead, larger than the entire sequential sumcheck for
   `n ≤ 12`.

2. The Phase-A refactor's per-worker split inbox/outbox **halves the
   small-`k` dispatch floor** (131 ns at `k=2`, 296 ns at `k=4`) by
   eliminating the cache-line ping-pong between main's `epoch` writes
   and workers' `done` writes that happened on every dispatch in the
   original shared-counter design. The trade-off is that `k=8` got
   slightly worse (1.2 µs vs 681 ns) because the split design touches
   ~2× as many cache lines per dispatch (one inbox + one outbox per
   active worker) and that tax becomes visible when all 8 workers are
   active. For the sumcheck use case the win at `k=2/4` matters more:
   the scheduler's D2 multi-phase shrinkage (see below) spends most of
   its dispatches at `k ≤ 4` once the parallel chunk shrinks.

## Raw results (Phase A refactor, current code)

### A/B against the pre-refactor single-broadcast wrapper

The first pass of Phase-A benchmarks recorded Criterion medians in the
20-300% range above the pre-refactor baseline (e.g. GF128 `n = 16`
`pinned` went from 52 µs to 94 µs, and `pinned_repro` p10 sat at
272 µs — see the `bench-phaseA-2026-04.log` timestamps). That looked
like a real regression, but the same Criterion session over the
pre-refactor code today reproduces the *same* slow numbers: machine
state (thermal, background load, L1/L2 cold-start) drifted between the
original benchmark session and the Phase-A session, and the "original"
numbers are not reproducible on the current box.

The clean A/B test is to add the pre-refactor `pinned` wrapper
back into the tree (`legacy::sumcheck_deg2_delayed_*_pinned_v0`,
single `broadcast_scoped`, direct `[SendPtr; 2]` captures, no D2
multi-phase, no generic trait) and run both wrappers back-to-back on
the *same* `PinnedPool` and *same* measurement loop
(`examples/pinned_ab.rs`, 2000 iterations, alternating calls to keep
thermals / cache state balanced).

Min-of-2000 samples, µs. Lower is better; `NEW` is
`par_sumcheck` / `SumcheckRound` / D2; `LEG` is the pre-refactor
single-broadcast wrapper on the same pool.

| n  | Field  | NEW min | LEG min | Δ min |
|----|--------|---------|---------|-------|
| 10 | GF128  | 14.9    | 14.3    | +4.4% |
| 10 | Fp128  | 7.04    | 6.92    | +1.7% |
| 12 | GF128  | 21.4    | 21.0    | +2.2% |
| 12 | Fp128  | 16.0    | 14.0    | +13.7% |
| 14 | GF128  | 68.5    | 64.1    | +7.0% |
| 14 | Fp128  | 37.8    | 35.6    | +6.0% |
| 16 | GF128  | 241     | 232     | +3.7% |
| 16 | Fp128  | 117     | 114     | +2.0% |
| 18 | GF128  | 923     | 884     | +4.4% |
| 18 | Fp128  | 444     | 433     | +2.5% |
| 20 | GF128  | 3655    | 3607    | +1.3% |
| 20 | Fp128  | 1717    | 1655    | +3.7% |

Conclusion: **Phase A costs ~4% over the hand-tuned single-broadcast
wrapper on the same machine**, in exchange for field-agnostic
scheduling, D2 multi-phase `n_active` shrinkage, and a stable trait
interface for downstream crates. Headline numbers (crossover at
`n = 12` for GF128 and `n = 10` for Fp128) are preserved.

### Where the 4% residual goes

Ablation (same A/B setup, `examples/pinned_ab.rs`):

1. **D2 off + ptrs cached** (`const D2_SHRINK: bool = false`) closes
   ~1-2% of the gap at `n ∈ {12..16}` but not at small n where the
   extra 2-3 phase dispatches add up to a larger fraction of the
   wall.
2. **Ptr caching** (cache `buf*.as_mut_ptr()` in
   `GF128DelayedRound` / `Fp128DelayedRound` instead of re-deriving
   from `UnsafeCell<Vec<_>>::get()` on each `read_ptrs`/`write_ptrs`
   call) was worth ~1-2% at `n ≤ 14` where the per-round deref is a
   larger slice of the total. Landed in the current tree.
3. The irreducible ~2-3% is the generic-trait path: `reduce_chunk`
   and `bind_chunk` go through `&dyn`-free static dispatch but LLVM
   emits one extra register shuffle vs the fully-inlined direct call
   in legacy, plus the per-round `abs_round = round_offset + r`
   addition. Not worth further optimization at this layer — the API
   needs to be field-agnostic for the production goal.

### D1 tournament reduce: negative result

We tried the `log2(n_workers)` tournament reduce (D1 in the
productionization plan) in `run_phase` to distribute the aggregate
across workers and save the `~n_workers` cross-core cache-line pulls
that worker 0 does today. It was a **net loss** for every point in
the GF128/Fp128 sweep on M4 Max:

| n  | field  | LEGACY p10 | NEW + D1 p10 | delta |
|----|--------|------------|--------------|-------|
| 12 | GF128  | 22.6 µs    | 26.5 µs      | +17%  |
| 12 | Fp128  | 15.8 µs    | 19.3 µs      | +22%  |
| 14 | GF128  | 75.4 µs    | 80.6 µs      |  +7%  |
| 14 | Fp128  | 41.7 µs    | 50.6 µs      | +21%  |
| 16 | GF128  | 276 µs     | 285 µs       |  +3%  |
| 16 | Fp128  | 119 µs     | 126 µs       |  +6%  |

Root cause: `Partial = (GF128, GF128, GF128)` is 48 bytes and
`combine` is three XORs (~3 ns). With `n_workers = 8` the tree has
three levels; each level uses one `fetch_add(Release)` on a shared
`AtomicUsize` to signal completion. Those three Release-RMWs on the
critical path cost ~100 ns each on M4 Max (cache-line ownership
transfer + store buffer drain), for a total of ~300 ns per round.
Meanwhile the "savings" (~30 ns × 7 remote-line loads ≈ 210 ns in
the limit, much less in practice because M4 Max's mesh + aggressive
prefetch get an 8-line linear scan to < 150 ns) don't cover it.

Tournament reduce is worth revisiting in two scenarios:

1. **Hardware with much higher cross-cluster latency.** On Zen 5
   (aragorn) or multi-socket Intel, the `n_workers = 8` linear
   reduce crosses more shared caches and the trade may flip.
2. **Kernels with expensive combines.** Packed BB5, KB5 sumchecks,
   or any partial that holds an `eq` factor alongside the polynomial
   evaluations will have `combine` taking tens of ns instead of 3 ns.
   At that point the `combine` distribution dominates the barrier
   cost and the tournament wins.

For now the scheduler keeps the linear reduce. The trait's
`combine` method stays associative-commutative so we can switch in
a tournament implementation later without changing any `impl
SumcheckRound`. See `docs/plans/sumcheck-parallel-productionization.md`
for the re-evaluation criteria.

### Absolute numbers (Phase A, current code)

Criterion medians from `target/parallelism-results/bench-phaseA-2026-04.log`
(`--warm-up-time 3 --measurement-time 6`, `SUMCHECK_PINNED_WORKERS` unset).
Criterion numbers are 1.5-3× above the `pinned_ab` min because they
include the long-tail distribution (occasional preemption, cache-cold
dispatches, etc.) in the point estimate. Absolute values drift 30-60%
across benchmark sessions on M4 Max even with 3s warm-up + 6s
measurement; use the `pinned_ab` A/B above for code-to-code comparisons,
and use the Criterion medians below only for crossover-point detection.

#### GF128 (µs for `n ≤ 16`, ms for `n ∈ {18, 20}`)

| n  | `delayed` | `pinned` (current) | speedup |
|----|-----------|-------------------------|---------|
| 10 | 4.03      | 4.68                    | 0.86×   |
| 12 | 16.4      | **13.3**                | **1.23×** |
| 14 | 62.3      | **34.2**                | **1.82×** |
| 16 | 257       | **94.7**                | **2.71×** |
| 18 | 1012 µs   | **249 µs**              | **4.07×** |
| 20 | 4.00 ms   | **1119 µs**             | **3.58×** |

#### Fp128 (µs for `n ≤ 16`, ms for `n ∈ {18, 20}`)

| n  | `delayed` | `pinned` (current) | speedup |
|----|-----------|-------------------------|---------|
| 10 | 11.8      | **10.8**                | **1.10×** |
| 12 | 47.8      | **23.6**                | **2.03×** |
| 14 | 235       | **84.1**                | **2.79×** |
| 16 | 788       | **286**                 | **2.75×** |
| 18 | 2.94 ms   | **1186 µs**             | **2.48×** |
| 20 | 12.7 ms   | **3.32 ms**             | **3.83×** |

**Bold** marks the winning variant. Crossover: GF128 at `n = 12`
(1.23×), Fp128 at `n = 10` (1.10×) — an improvement on the
pre-refactor Fp128 crossover of `n = 12` (where the pre-refactor code
lost by 10% at `n = 10`).

## Crossover summary

| Field | Winner                              | First n where parallel beats sequential | Best speedup |
|-------|-------------------------------------|-----------------------------------------|--------------|
| GF128 | `pinned` for every n ≥ 12      | **n = 12** (1.25× over seq)             | 5.3× at n=18 |
| Fp128 | `pinned` for every n ≥ 12      | **n = 12** (2.45× over seq)             | 6.6× at n=20 |

`pinned` moves the crossover from `n = 20 → n = 12` (GF128) and from
`n = 18 → n = 12` (Fp128), and it wins at **every** `n ≥ 12` across both
fields. At `n = 10` sequential still wins by ~10%, so the target
`n ≈ 8-10` crossover is not reached, but we land close to the predicted
`n = 14-16` range at `n = 12`.

Why `n = 10` still loses, quantitatively:

- GF128 sequential at n=10 is 3.85 µs for the whole 10-round sumcheck.
- `pinned` at n=10 selects 2 active workers (`initial_pairs = 512`,
  `PAIR_TARGET_PER_WORKER = 256`, so `n_active = 512/256 = 2`). The
  inner loop runs ~6 rounds inside one `broadcast_scoped` call (until
  `initial_per_worker >> r < 8`), then finishes the last ~4 rounds
  sequentially on the main thread.
- The single `broadcast_scoped` dispatch is ~305 ns. The per-round
  intra-pool barriers (two `AtomicUsize` waits per round, one after
  reduce and one after bind) cost another ~100-200 ns each, i.e. ~1-2 µs
  across 6 rounds.
- That puts a ~1.5-2.5 µs overhead budget on top of roughly halved
  per-round work. Result: 4.23 µs, 0.91× seq.
- Closing the remaining 400 ns gap would require driving the per-round
  barrier below ~30 ns per atomic, which is below the ~50 ns LLC
  round-trip latency on M4 and therefore infeasible without shared-L2
  threads (M4 has one shared L2 per P-core cluster).

## Aragorn validation (Zen 5, Linux)

Cross-platform validation on `aragorn`, the team's Linux test box.

### Machine

- AMD Ryzen 9 9950X, 16 physical cores × 2 SMT = 32 logical threads,
  single socket.
- Ubuntu 24.04.3 LTS, kernel 6.17.0-20-generic.
- rustc 1.95.0 stable; same `--warm-up-time 3 --measurement-time 6`
  Criterion config; `SUMCHECK_PINNED_WORKERS` unset (pool defaults to
  `min(available_parallelism, 8) = 8` workers — important: on a
  16-physical-core box we're using **half the cores by design**; see
  end of this section).
- Raw logs in `target/parallelism-results/aragorn-bench-2026-04.log`
  (Criterion) and `aragorn-ab-2026-04.log` (A/B profiler).

### Linux pinning shim works correctly

`SUMCHECK_PINNED_DEBUG=1 pinned_ab` confirms each of the 7 extras lands
on a distinct physical core (canonical logical cpus 1..7 from
`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`). Zen 5
numbers SMT siblings as `(k, k+16)` for `k ∈ 0..15`; our
`min(thread_siblings_list)` convention keeps us on the lower 16, one
thread per physical core. `SCHED_FIFO` opt-in via
`SUMCHECK_PINNED_SCHED_FIFO=1` is available but disabled for these
runs (we don't hold `CAP_SYS_NICE` and fall back silently).

### Dispatch floor: 3-5× lower than M4

| Microbench                                        | M4 Phase A | aragorn   |
|---------------------------------------------------|------------|-----------|
| `par_iter_sum_num_threads` (rayon)                | 20.5 µs    | 12.6 µs   |
| `scope_spawn_num_threads_nop` (rayon)             | 19.1 µs    | 21.9 µs   |
| `chili_scope_join_noop`                           | 164 ns     | 166 ns    |
| `pinned_pool_broadcast_nop_k2`                    | 131 ns     | **44 ns** |
| `pinned_pool_broadcast_nop_k4`                    | 296 ns     | **73 ns** |
| `pinned_pool_broadcast_nop_k8`                    | 1.20 µs    | **103 ns**|

Aragorn's unified L3 across all 16 cores + lower cross-core latency
drops our dispatch floor by 3-11× at `k ∈ {2, 4, 8}`. The pinned pool
at `k = 8` dispatches in ~100 ns; that's **120× cheaper than
`par_iter_sum_num_threads`** on the same box.

### A/B: NEW (trait + D2) vs LEGACY single-broadcast

Same test as `examples/pinned_ab.rs`, 3000 iterations, alternating calls.
Unlike M4 (where the trait path cost ~4% over the hand-tuned
legacy wrapper), on aragorn the **trait + D2 path beats the legacy
single-broadcast path** at every point where parallelism matters:

| n  | field  | NEW p50  | LEG p50  | Δ p50     |
|----|--------|----------|----------|-----------|
| 12 | GF128  | 116.85   | 119.22   | **-2.0%** |
| 12 | Fp128  | 13.40    | 13.39    |  0%       |
| 13 | GF128  | 130.44   | 139.71   | **-6.6%** |
| 13 | Fp128  | 31.78    | 39.34    | **-19.2%**|
| 14 | GF128  | 255.34   | 256.60   | -0.5%     |
| 14 | Fp128  | 39.91    | 40.30    | -1.0%     |
| 15 | GF128  | 511.80   | 534.13   | **-4.2%** |
| 15 | Fp128  | 75.90    | 95.29    | **-20.4%**|
| 16 | GF128  | 1010     | 1006     | +0.4%     |
| 16 | Fp128  | 145.85   | 146.21   | -0.2%     |

The big wins at odd n (n=13, 15 for Fp128) are exactly where D2's
phase-halving schedule fires: at n=13 the trip from 4096 pairs to
32 pairs takes 7 halving steps, and D2 drops `n_active` from 8 → 4 →
2 → 1 along the way instead of paying 8-way barrier latency on the
last round before the sequential tail. That's what the trait +
scheduler rewrite was designed for, and Zen 5 benefits from it more
than M4 because cross-core barriers are relatively more expensive on
a 16-core mesh than on M4's P-core cluster.

### Absolute numbers (Criterion medians)

GF128 is CPU-bound on a scalar `u128` software multiply path on
aragorn (no `pclmulqdq` SIMD backend yet), so absolute GF128 numbers
in the table below are ~10× slower than M4 Max NEON. That's a
known gap tracked separately from parallelism work; the per-variant
speedup columns still show how the parallel scheduler scales.

> **Update (Phase B follow-up, see below).** The "no PCLMULQDQ"
> regime is now history. Adding `-C target-cpu=native -C
> target-feature=-avx512f` to `RUSTFLAGS` (committed as
> `.cargo/config.toml` in the repo) routes binius-field's GF(2^128)
> multiply through `packed_ghash_256` (VPCLMULQDQ-256). Sequential
> GF128 sumcheck drops ~8.7× across the whole sweep below; see
> the **"Aragorn, Phase B follow-up: VPCLMULQDQ"** section near the
> end of this doc for the post-flag table. The numbers in the
> immediately-following sub-tables are the *pre-flag* baseline,
> kept as historical reference.

#### GF128 (aragorn, µs for n ≤ 14, ms for n ≥ 16)

| n  | `delayed` | `pinned` | speedup | next-best variant   |
|----|-----------|---------------|---------|---------------------|
| 10 | 96.3      | **50.3**      | 1.92×   | all others ≥ 97 µs  |
| 12 | 385       | **95.8**      | 4.02×   | rayon_scope 297.8 µs |
| 14 | 1543      | **207**       | 7.45×   | rayon_scope 461 µs   |
| 16 | 6.67 ms   | **807 µs**    | 8.26×   | rayon_iter 1.10 ms|
| 18 | 27.0 ms   | 3.72 ms       | 7.25×   | **rayon_scope 3.51 ms** |
| 20 | 98.9 ms   | 16.2 ms       | 6.10×   | **rayon_scope 12.4 ms** |

#### Fp128 (aragorn, µs for n ≤ 16, ms for n ≥ 18)

| n  | `delayed` | `pinned` | speedup | next-best variant   |
|----|-----------|---------------|---------|---------------------|
| 10 | 14.8      | **7.11**      | 2.08×   | chili_b128 14.70 µs |
| 12 | 45.7      | **11.4**      | 4.01×   | delayed itself 45.7 |
| 14 | 182       | **50.8**      | 3.58×   | chili_b2048 180 µs  |
| 16 | 733       | **106**       | 6.89×   | rayon_scope 430 µs   |
| 18 | 2.93 ms   | **428 µs**    | 6.86×   | rayon_scope 1.01 ms  |
| 20 | 11.8 ms   | 6.25 ms       | 1.88×   | **rayon_scope 5.90 ms** |

### Crossover summary (aragorn vs M4)

| Field | M4 crossover | aragorn crossover |
|-------|--------------|-------------------|
| GF128 | n = 12 (1.25×) | **n ≤ 10 (1.92×)** |
| Fp128 | n = 10 (1.10×) | **n ≤ 10 (2.08×)** |

Aragorn hits the `n = 8-10` crossover target that M4 fell short of.
The cheaper dispatch floor (103 ns vs 1.2 µs at k=8) is the direct
cause: the ~1.5-2.5 µs M4 overhead budget at n=10 shrinks to ~0.3 µs
on aragorn, leaving most of the parallel gain intact.

### Cap at 8 workers is leaving cores on the table at n ≥ 18

At n ∈ {18, 20}, rayon's `rayon_scope` (which uses the default rayon
pool = 16 threads = all 32 logical cores) starts beating
`pinned` (fixed 8 workers). GF128 n=20: 12.4 ms (rayon_scope) vs
16.2 ms (pinned), -23%. Fp128 n=20: 5.9 ms vs 6.25 ms, -5.6%.

Two options for follow-up (tracked under Phase B in the plan doc):

1. Raise the default cap to `min(physical_cores, 16)`. On aragorn that
   doubles the pool to 16 and should close or invert the gap at
   n ≥ 18. On M4 Max it stays at 12 (one per P-core) which is already
   ≥ 8 with room to grow.
2. Make D2 more aggressive at the top of the schedule: even with 16
   workers, dispatch at `k = 16` only when `initial_pairs ≥ 16 ×
   TARGET_PAIRS_PER_WORKER`, otherwise start at `k = 8`. This keeps
   the dispatch-floor advantage at small/mid n while unlocking
   large-n throughput.

Neither blocks Phase A shipping; both are profile-guided tunings on
top of the current trait + scheduler.

## Aragorn, Phase B: `pinned-pool` crate + auto-park + 32 workers

Phase B of the productionization plan (see
`docs/plans/sumcheck-parallel-productionization.md`) lifts the pool
cap and introduces auto-park so the pool is honest neighbor in
benchmarks and servers:

1. **`pinned-pool` crate extracted** to `~/Documents/SNARKs/pinned-pool/`
   (local path dep for now). Main repo imports via `pinned_pool::PinnedPool`.
2. **Default cap lifted** to `available_parallelism()` on Linux (= 32
   logical cores on aragorn) with affinity-based pinning that fills
   physical cores first and then SMT siblings. macOS stays conservative
   at `min(P-cores - 2, 8)` because QoS pinning is a hint, not an
   anchor.
3. **Auto-park on idle**: workers spin for ~300 µs-1 ms of
   `hint::spin_loop` after a task, then transition to `thread::park()`
   using Dekker-style coordination between main's
   `assigned_gen.store(Release) + fence(SeqCst) + parked.load` and the
   worker's mirror sequence. First dispatch after long idle pays one
   `unpark` per active worker (~200 ns-1 µs), subsequent hot dispatches
   stay in the spin path with zero futex overhead.
4. **Bench fairness**: auto-park eliminates the pool-contamination
   artifact where rayon benchmarks looked 10-30× slower when run
   serially in the same Criterion process after `pinned` had
   warmed its pool.

Numbers below are from the first fair Criterion sweep with all four
variants in the same process, pool default = 32, `PINNED_POOL_DEBUG=0`:

**GF128 (aragorn, Zen 5 × 32 logical)**

| n  | seq      | rayon_scope | rayon_iter | persistent | **pinned** | pinned vs best rayon |
|----|----------|------------|--------------|--------------|-----------------|--------------------|
| 10 | 94.8 µs  | 252 µs     | 230 µs       | 144 µs       | **49.3 µs**     | 2.9× faster        |
| 12 | 378 µs   | 331 µs     | 307 µs       | 223 µs       | **102 µs**      | 3.0× faster        |
| 14 | 1.52 ms  | 445 µs     | 461 µs       | 233 µs       | **134 µs**      | 1.7× faster        |
| 16 | 6.09 ms  | 873 µs     | 911 µs       | 680 µs       | **530 µs**      | 1.3× faster        |
| 18 | 24.3 ms  | 3.17 ms    | 3.18 ms      | 3.57 ms      | **2.64 ms**     | 1.2× faster        |
| 20 | 97.7 ms  | 12.0 ms    | 12.4 ms      | 12.3 ms      | **10.7 ms**     | 1.12× faster       |

**Fp128 (aragorn)**

| n  | seq      | rayon_scope | rayon_iter | persistent | **pinned** | pinned vs best rayon |
|----|----------|------------|--------------|--------------|-----------------|--------------------|
| 10 | 11.3 µs  | 154 µs     | 230 µs       | 27.7 µs      | **6.86 µs**     | 4.0× faster (vs persistent) |
| 12 | 44.7 µs  | 200 µs     | 298 µs       | 40.8 µs      | **10.7 µs**     | 3.8× faster (vs persistent) |
| 14 | 179 µs   | 278 µs     | 379 µs       | 263 µs       | **55.8 µs**     | 4.7× faster (vs persistent) |
| 16 | 719 µs   | 454 µs     | 636 µs       | 232 µs       | **151 µs**      | 1.5× faster (vs persistent) |
| 18 | 2.88 ms  | 1.02 ms    | 1.37 ms      | 741 µs       | **519 µs**      | 1.4× faster (vs persistent) |
| 20 | 11.6 ms  | 5.78 ms    | 6.88 ms      | 5.32 ms      | **5.27 ms**     | 1.01× faster (vs persistent) |

**`pinned` wins at every `(field, n)` pair**, with the gap
widening at small n (where the dispatch floor dominates) and
narrowing at n=20 (where we are bandwidth-bound and any sane
parallelizer converges to the same number). This is the "always beat
rayon at every scale" goal from the productionization plan.

Dispatch floor at pool size 32 on aragorn (for comparison):

| `k` | cost    |
|-----|---------|
|  2  | 72.4 ns |
|  4  | 104 ns  |
|  8  | 99.7 ns |
| 12  | 428 ns  |
| 16  | 485 ns  |
| 24  | 479 ns  |
| 32  | 564 ns  |

Rayon equivalents (same process, pool init'd): `rayon::scope` on 32
threads costs 21.7 µs/dispatch and `par_iter` on 32 threads costs
12.8 µs. The pinned pool is 40-300× cheaper per dispatch.

## Aragorn, Phase B follow-up: VPCLMULQDQ unlocked via `target-cpu=native`

Until now every aragorn GF128 number above was on the *portable
`u128` software multiply* path inside `binius-field`, because Rust's
default `x86_64-unknown-linux-gnu` target doesn't enable
`target_feature = "pclmulqdq"`. binius's `arch/x86_64/packed_ghash_*`
SIMD code paths are gated behind those `target_feature` cfgs and
silently compiled out, so the GF(2^128) mul fell back to portable
limb-wise polynomial arithmetic, which is ~10× slower than NEON
`pmull` on M4 Max.

Fix is one config file, zero code changes. We commit
`.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "target-feature=-avx512f"]
```

The `-avx512f` is necessary because `hachi-pcs` (our Fp128 source)
gates a `#![feature(stdarch_x86_avx512)]` opt-in behind
`target_feature = "avx512f"`, and `feature(...)` is forbidden on the
stable channel even when the underlying intrinsics have stabilized.
Disabling AVX-512 keeps hachi compiling and routes binius through
its `packed_ghash_256` path (VPCLMULQDQ-256, 2 carryless muls per
SIMD op) instead of `packed_ghash_512` (VPCLMULQDQ-512, 4 per op).
Estimated cost of skipping AVX-512: ~2× peak GF128 mul throughput,
which is much cheaper than maintaining a nightly toolchain.

### Inner-loop validation: `field_ops` GF128 mul

Pre/post-flag, same hardware (aragorn, Zen 5), same source tree:

| benchmark         | pre-flag | post-flag | speedup  |
|-------------------|----------|-----------|----------|
| `lat_mul` (latency, 4096-mul chain) | 79.16 µs | 15.86 µs | **5.0×** |
| `thr_mul` (throughput, 1024 ops, ILP-friendly) | 21.91 µs | 2.16 µs | **10.1×** |

The throughput speedup is bigger because VPCLMULQDQ-256 packs two
128-bit carryless multiplies per SIMD instruction, doubling the
ILP-bound rate.

### End-to-end: GF128 sumcheck on aragorn (post-flag)

Same Criterion harness as the Phase B table above, just with the new
`.cargo/config.toml`:

| n  | seq      | rayon_scope | rayon_iter | persistent | **pinned** | pinned vs best rayon |
|----|----------|------------|--------------|--------------|-----------------|--------------------|
| 10 | 10.86 µs | 234 µs     | 243 µs       | 27.5 µs      | **6.42 µs**     | 4.3× faster        |
| 12 | 43.4 µs  | 320 µs     | 302 µs       | 46.6 µs      | **9.80 µs**     | 4.7× faster        |
| 14 | 174 µs   | 408 µs     | 378 µs       | 321 µs       | **47.3 µs**     | 6.8× faster        |
| 16 | 697 µs   | 603 µs     | 614 µs       | 238 µs       | **145 µs**      | 1.6× faster        |
| 18 | 2.81 ms  | 1.73 ms    | 1.94 ms      | 1.55 ms      | **1.35 ms**     | 1.15× faster       |
| 20 | 11.3 ms  | 6.55 ms    | 7.34 ms      | 5.65 ms      | **5.62 ms**     | 1.005× faster      |

Per-variant speedup vs the pre-flag Phase B numbers:

| n  | seq    | pinned | rayon (best) |
|----|--------|-------------|--------------|
| 10 | 8.7×   | 7.7×        | 1.6× (persistent)  |
| 12 | 8.7×   | 10.4×       | 4.8× (persistent)  |
| 14 | 8.7×   | 2.8×        | 0.7× (persistent, regression noise) |
| 16 | 8.7×   | 3.7×        | 2.9× (persistent)  |
| 18 | 8.7×   | 2.0×        | 2.3× (persistent)  |
| 20 | 8.7×   | 1.9×        | 2.2× (persistent)  |

Sequential GF128 sumcheck gets a clean ~8.7× win at every n,
matching the per-mul speedup almost exactly (the inner loop is
mul-dominated; everything else is in cache). Parallel speedup
ranges from 1.9× at large n (memory-bandwidth bound, the SIMD mul
no longer dominates) to 10× at small n (compute-bound, where the
new mul throughput shines through).

`pinned` is still the fastest variant at every `(n)` post-flag,
so the "always beat rayon at every scale" property is preserved.
The crossover advantage *widens* at small n: `pinned/12` now
beats the best rayon variant by 4.7× (was 3.0×), and `pinned/14`
by 6.8× (was 1.7×).

Fp128 is unchanged by the flag (it routes through `hachi-pcs`,
which doesn't use binius's SIMD codepath): spot-checks at n=10
(11.0 µs vs 11.0 µs) and n=20 (5.23 ms vs 5.27 ms) match the
pre-flag table within Criterion noise.

### Future optimization: hand-rolled x86 GF128Accum

`src/sumcheck/gf128.rs` has two `GF128Accum` impls: an aarch64 NEON
one that hand-rolls a deferred-reduction `vmull_p64` FMA chain, and
a generic non-aarch64 one that just does `acc += a * b`. The latter
now (post-flag) goes through binius's PCLMULQDQ path on x86_64, so
it's not "scalar" anymore, but it still pays a modular-reduction
cost on every multiply instead of folding it across the loop the way
the NEON code does. Mirroring the aarch64 hand-rolled accumulator
on x86 with `_mm_clmulepi64_si128` + deferred Montgomery reduction
would likely give another ~2-3× on top of the 8.7× we just bought.
That's a separate project; tracked under "future optimizations" in
`docs/plans/sumcheck-parallel-productionization.md`.

## Detailed findings

### 1. `pinned` (pinned pool + doorbell) is the overall winner.

For **every** `(field, n)` pair with `n ≥ 12`, `pinned` is the
fastest variant, often by a factor of 2-10× over the next-best parallel
approach. The jump comes from three design choices that together
attack the dispatch floor from a different direction than approaches
1-3:

1. **Process-global persistent pool, not per-call.** Workers are spawned
   once via a `OnceLock<PinnedPool>` and live for the lifetime of the
   process. No `rayon::scope` per call, no per-call thread spawn, no
   per-round `scope.join`.
2. **Pinned on P-cores via `QOS_CLASS_USER_INTERACTIVE`.** On macOS,
   heavy spinning without a QoS hint causes the scheduler to demote
   threads to E-cores (the exact failure mode we hit with `persistent`).
   Tagging the pool workers as user-interactive tells the scheduler
   "these threads are latency-critical", and the spinners stay on
   P-cores indefinitely. This lets the pool's doorbell loop use
   **pure `std::hint::spin_loop`** on the epoch atomic (no `yield_now`
   fallback), which drops the wake-up latency from microseconds to
   hundreds of nanoseconds.
3. **Adaptive `n_active`.** `broadcast_scoped(n_active, f)` writes
   `n_active` into the pool struct before flipping the epoch; workers
   with `worker_idx >= n_active` skip the task body and the done
   counter. The sumcheck wrappers pick
   `n_active = clamp(initial_pairs / 256, 2, pool_size)`, so small
   problems only pay for 2-3 active workers and we avoid contending on
   the done counter with idle workers.

Measured broadcast floor: 305 ns (k=2) / 443 ns (k=4) / 681 ns (k=8).
That is the overhead paid **once per `broadcast_scoped` call**. The
sumcheck wrappers call `broadcast_scoped` once per whole sumcheck and
use cheaper intra-pool atomic barriers (`reduce_counter`, `bind_go`)
for the per-round reduce/bind sync — so the broadcast cost is amortised
across all parallel rounds, and the per-round cost is just the
round-trip latency of two `AtomicUsize` ops (~100-200 ns per round).

### 2. The crossover lands at `n = 12`, not at the hoped-for `n = 8-10`.

The handoff note predicted `pinned` would "drop the crossover to maybe
`n = 14-16`, not 8". In practice:

- GF128: n=10 loses by 10% (4.23 vs 3.85 µs); n=12 wins by 25%; n=18
  wins by **5.3×**.
- Fp128: n=10 loses by 10% (13.3 vs 12.0 µs); n=12 wins by **2.45×**;
  n=20 wins by **6.6×**.

So the crossover is actually better than predicted (`n = 12` vs
`n = 14-16`), but `n = 8` remains architecturally out of reach. The
bottleneck at very small `n` is not the single top-level broadcast
(which is amortised across the whole sumcheck) but the per-round
cross-core barrier: each parallel round requires at least one
atomic-counter wait per worker, and at ~100 ns per atomic round-trip
the barrier cost is larger than the honest parallel work that
remains at `n ≤ 10`.

### 3. `persistent` (older persistent pool) is superseded.

`persistent` was the best variant at `n = 20` in the original
sweep (1.6-1.8 ms for GF128/Fp128), but `pinned` beats it at every
`n` by 1.5-10×: GF128 n=20 is 993 µs for `pinned` vs 3.49 ms for
`persistent` (3.5× improvement). The two key differences:

- `persistent` spawns its pool inside a `rayon::scope` per call
  (~24 µs scope floor per call, regardless of how persistent the
  internal workers are).
- `persistent` uses spin-then-yield because its workers are not
  QoS-tagged; `pinned`'s `USER_INTERACTIVE` tag lets it pure-spin
  without E-core demotion.

We keep `persistent` in the benchmark for historical comparison
but it should no longer be used in production.

### 4. Chili's low-latency fork/join helps at `n ≈ 10-14`, but not enough to win.

Chili's `scope.join` is ~**140× cheaper** than rayon's `scope` dispatch
(164 ns vs 23 µs). At `n = 10-12` chili stays within 2-4× of sequential
while per-round rayon variants are 10-50× off. But `pinned` is still
faster than the best chili base-case at every `n ≥ 12`, and chili
never beats sequential at `n ≤ 14` for either field.

The chili base-case sweep shows a clear sweet spot:
- **n=10-14**: bigger `base` (1024-2048) wins. Chili's recursion overhead
  dominates; keeping most of the work in one thread is cheapest.
- **n=16-20**: smaller `base` (128-512) wins. Parallelism benefits start to
  outweigh dispatch, and finer split improves load balance.

### 5. `rayon_scope` beats `rayon_iter` across the board.

For every (field, n) point, manual `rayon::scope` + explicit chunking is
faster than `par_iter + par_chunks_mut`. The gap is largest at small n
(2-5× at n=10-14). This confirms the handoff hypothesis that `par_iter`
adds its own overhead on top of `rayon::scope`.

Still, *any* rayon-scope-per-round approach loses to sequential below
n=18 due to the 24 µs scope dispatch floor.

### 6. Fp128 and GF128 now cross over at the same `n`.

With `pinned`, both fields cross over at `n = 12`. In the old sweep
Fp128 crossed over a doubling earlier than GF128 (`n = 18` vs `n = 20`)
because the per-element Fp128 work is ~5× heavier (Solinas reduction vs
CLMUL + GF(2^128) reduction), and that made the dispatch-heavy parallel
variants look relatively better. Now that `pinned`'s dispatch floor
is low enough to be invisible at `n = 12`, the crossover is determined
by pair count, not per-pair work, and both fields hit it at the same
`n`. The Fp128 **speedups** are still larger (6.6× vs 3.9× at n=20)
because there's more honest work per pair to parallelise over.

### 7. The `n = 8` crossover target is architecturally unreachable.

At `n = 8` GF128 the whole sequential sumcheck is ~1 µs. The top-level
broadcast is already amortised (one 305 ns dispatch per sumcheck call)
so dispatch is not the limiter. The limiter is the **per-round
barrier**: each parallel round requires all workers to synchronise on
a shared atomic before proceeding to bind, and each atomic round-trip
on M4 is ~50-100 ns bounded below by LLC latency. With 2 workers and
~6 parallel rounds, the barrier cost alone is close to 1 µs, which is
the entire sequential budget at `n = 8`.

Closing that gap would require either (a) amortising the barrier
across multiple rounds (e.g. speculative execution of all rounds on
the main thread, with workers patching in corrections), or (b)
cross-core signalling below ~30 ns per barrier, which is below the
M4 LLC latency floor and therefore impossible without shared-L2
threads (M4 has one shared L2 per P-core cluster).

Other options we have *not* tried and that are plausibly worth pursuing
if the application has more structure than "one sumcheck call":

- **SIMD pipelining multiple round iterations inside one thread.** For
  GF128, CLMUL already does intra-pair SIMD; the lift is pipelining
  round `r` and round `r+1` on the same core to hide dependency latency.
  No cross-core sync at all.
- **Batch multiple independent sumcheck instances across the pool.**
  `PinnedPool::broadcast_scoped` already supports this: each worker can
  drive one instance to completion with zero per-round sync. This is
  the right answer if the application needs many small sumchecks.
- **Lock-free SPSC ring buffers between workers** instead of a
  per-round barrier. Removes the all-threads-arrive sync, but adds
  protocol complexity and has not been implemented here.

## Recommendation

For the delayed sumcheck kernel on Apple Silicon:

1. **Use `pinned` as the default for `n ≥ 12`.** It is the
   fastest variant at every `(field, n)` we measured for `n ≥ 12`, with
   speedups of 1.25-5.3× (GF128) and 2.45-6.6× (Fp128) over sequential.
   Crossover is `n = 12` for both fields.
2. **Use sequential at `n ≤ 10`.** `pinned` loses by ~10% at `n = 10`
   (GF128: 4.23 vs 3.85 µs; Fp128: 13.3 vs 12.0 µs): the top-level
   broadcast is cheap, but the per-round atomic barriers across workers
   accumulate to 1-2 µs over ~6 parallel rounds, which is larger than
   the parallel speedup recovers at these sizes. A simple size check
   (`if initial_pairs < 1024 { delayed(...) } else { delayed_pinned(...) }`)
   is the production-safe dispatch.
3. **Override `SUMCHECK_PINNED_WORKERS` only if you know the workload.**
   Default is `min(available_parallelism(), 8)`. For pure `n = 10`
   GF128 workloads, `SUMCHECK_PINNED_WORKERS=3` gives a measured
   3.66 µs (0.95× seq), which flips the sign on the n=10 crossover;
   but that hurts `n ≥ 14` throughput noticeably. Leave the default
   alone unless small-n is the dominant use case.
4. **Retire `persistent` and the chili base-case sweep.** Approach
   4 dominates both. The chili variants are retained only because
   they're cheap to keep building; `persistent` is kept for
   historical comparison and should not be called from production.
5. **If the goal really is `n ≈ 8` crossover, change the problem shape,
   not the scheduler.** Intra-thread SIMD pipelining or batching many
   independent sumcheck instances across the pool are the only options
   left. No cross-core signalling mechanism on M4 can deliver below
   ~100 ns per round; the LLC latency floor is physical.

## Reproducing

```bash
# correctness (bit-identical parallel vs sequential for all 4 approaches)
cargo test --release --features parallel_chili --test parallel_delayed

# full bench sweep (~15 min on M4 Max)
cargo bench --bench sumcheck_parallel --features parallel_chili -- \
  --warm-up-time 2 --measurement-time 4

# dispatch floor only (includes pinned_pool_broadcast_nop_k{2,4,8})
cargo bench --bench sumcheck_parallel --features parallel_chili -- dispatch_floor

# override the pinned pool size (default: min(available_parallelism(), 8))
SUMCHECK_PINNED_WORKERS=4 cargo bench --bench sumcheck_parallel \
  --features parallel_chili -- pinned
```

Raw Criterion output:
- `target/parallelism-results/bench-full.log` (`rayon_scope`, `rayon_iter`, `chili`, `persistent` pre-production variants, original sweep)
- `target/parallelism-results/bench-pinned-final.log` (`pinned` final sweep, all 4 approaches)
