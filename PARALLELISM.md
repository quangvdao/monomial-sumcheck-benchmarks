# Parallelism for the delayed sumcheck kernels

Goal: figure out whether the per-round reduce-then-bind loop in the monomial
sumcheck kernels can be parallelised cheaply enough to beat the sequential
baseline at small problem sizes (`n ≈ 8-10`), not just at the paper's `n = 24`
headline point.

## Machine / configuration

- Apple M4 Max, 16-core (12 P + 4 E), 64 GB RAM.
- macOS 25.4.0, release build with `lto = "thin"`, `rayon 1`, `chili 0.2.1`.
- Default rayon pool: 16 threads. No RAYON_NUM_THREADS pinning.
- `bench_args = --warm-up-time 2 --measurement-time 4` (100 samples)
  for the original prototype; **Phase A** (`bench-phaseA-2026-04.log`)
  uses `--warm-up-time 3 --measurement-time 6` to fight variance.
- Criterion reports `[low median high]`; numbers below are **median**.
- `SUMCHECK_PINNED_WORKERS` unset (Approach 4 picks `min(available_parallelism(), 8) = 8`).
- Phase-A code lives in `src/sumcheck/parallel/{pool,scheduler,field,impls}/`;
  the original prototype is preserved in `src/sumcheck/parallel/legacy.rs`
  (Approaches 1-3) for historical comparison. Approach 4's per-pair math
  is shared between the two via `pub(super) fn partial_triple_*` /
  `bind_chunk_*` so the inner loop body is byte-identical.

## Four approaches compared

| Label                        | Strategy                                                                                             |
|------------------------------|------------------------------------------------------------------------------------------------------|
| `delayed`                    | Sequential baseline (the existing hot kernel).                                                       |
| `delayed_par1_scope`         | **Approach 1**: one `rayon::scope` per round; `n_workers` manual spawns; fused reduce+bind per worker. |
| `delayed_par1_pariter`       | Control for Approach 1: `(0..n_workers).into_par_iter()` + `par_chunks_mut`.                         |
| `delayed_par2_chili_bK`      | **Approach 2**: chili `scope.join` recursion with base-case `K` (sweep `K ∈ {32, 128, 512, 2048}`).  |
| `delayed_par3_persistent`    | **Approach 3**: one `rayon::scope` for the whole call; persistent workers on ping-pong buffers; atomic-counter barriers with spin-then-yield. |
| `delayed_par4_pinned`        | **Approach 4**: process-global `PinnedPool` of `std::thread` workers (default 8, tagged `QOS_CLASS_USER_INTERACTIVE` on macOS); per-call doorbell (epoch + done atomics), adaptive `n_active` via `PAR4_TARGET_PAIRS_PER_WORKER = 256`. |

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
   (164 ns vs 19-23 µs). That explains why Approach 1 (rayon scope per
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
`par4_pinned` went from 52 µs to 94 µs, and `par4_repro` p10 sat at
272 µs — see the `bench-phaseA-2026-04.log` timestamps). That looked
like a real regression, but the same Criterion session over the
pre-refactor code today reproduces the *same* slow numbers: machine
state (thermal, background load, L1/L2 cold-start) drifted between the
original benchmark session and the Phase-A session, and the "original"
numbers are not reproducible on the current box.

The clean A/B test is to add the pre-refactor `par4_pinned` wrapper
back into the tree (`legacy::sumcheck_deg2_delayed_*_par4_pinned_legacy`,
single `broadcast_scoped`, direct `[SendPtr; 2]` captures, no D2
multi-phase, no generic trait) and run both wrappers back-to-back on
the *same* `PinnedPool` and *same* measurement loop
(`examples/par4_ab.rs`, 2000 iterations, alternating calls to keep
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

Ablation (same A/B setup, `examples/par4_ab.rs`):

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

### Absolute numbers (Phase A, current code)

Criterion medians from `target/parallelism-results/bench-phaseA-2026-04.log`
(`--warm-up-time 3 --measurement-time 6`, `SUMCHECK_PINNED_WORKERS` unset).
Criterion numbers are 1.5-3× above the `par4_ab` min because they
include the long-tail distribution (occasional preemption, cache-cold
dispatches, etc.) in the point estimate. Absolute values drift 30-60%
across benchmark sessions on M4 Max even with 3s warm-up + 6s
measurement; use the `par4_ab` A/B above for code-to-code comparisons,
and use the Criterion medians below only for crossover-point detection.

#### GF128 (µs for `n ≤ 16`, ms for `n ∈ {18, 20}`)

| n  | `delayed` | `par4_pinned` (current) | speedup |
|----|-----------|-------------------------|---------|
| 10 | 4.03      | 4.68                    | 0.86×   |
| 12 | 16.4      | **13.3**                | **1.23×** |
| 14 | 62.3      | **34.2**                | **1.82×** |
| 16 | 257       | **94.7**                | **2.71×** |
| 18 | 1012 µs   | **249 µs**              | **4.07×** |
| 20 | 4.00 ms   | **1119 µs**             | **3.58×** |

#### Fp128 (µs for `n ≤ 16`, ms for `n ∈ {18, 20}`)

| n  | `delayed` | `par4_pinned` (current) | speedup |
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
| GF128 | `par4_pinned` for every n ≥ 12      | **n = 12** (1.25× over seq)             | 5.3× at n=18 |
| Fp128 | `par4_pinned` for every n ≥ 12      | **n = 12** (2.45× over seq)             | 6.6× at n=20 |

Approach 4 moves the crossover from `n = 20 → n = 12` (GF128) and from
`n = 18 → n = 12` (Fp128), and it wins at **every** `n ≥ 12` across both
fields. At `n = 10` sequential still wins by ~10%, so the target
`n ≈ 8-10` crossover is not reached, but we land close to the predicted
`n = 14-16` range at `n = 12`.

Why `n = 10` still loses, quantitatively:

- GF128 sequential at n=10 is 3.85 µs for the whole 10-round sumcheck.
- Approach 4 at n=10 selects 2 active workers (`initial_pairs = 512`,
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

## Detailed findings

### 1. Approach 4 (pinned pool + doorbell) is the overall winner.

For **every** `(field, n)` pair with `n ≥ 12`, `par4_pinned` is the
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
   threads to E-cores (the exact failure mode we hit with Approach 3).
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

The handoff note predicted Approach 4 would "drop the crossover to maybe
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

### 3. Approach 3 (older persistent pool) is superseded.

`par3_persistent` was the best variant at `n = 20` in the original
sweep (1.6-1.8 ms for GF128/Fp128), but Approach 4 beats it at every
`n` by 1.5-10×: GF128 n=20 is 993 µs for `par4_pinned` vs 3.49 ms for
`par3_persistent` (3.5× improvement). The two key differences:

- `par3_persistent` spawns its pool inside a `rayon::scope` per call
  (~24 µs scope floor per call, regardless of how persistent the
  internal workers are).
- `par3_persistent` uses spin-then-yield because its workers are not
  QoS-tagged; Approach 4's `USER_INTERACTIVE` tag lets it pure-spin
  without E-core demotion.

We keep `par3_persistent` in the benchmark for historical comparison
but it should no longer be used in production.

### 4. Chili's low-latency fork/join helps at `n ≈ 10-14`, but not enough to win.

Chili's `scope.join` is ~**140× cheaper** than rayon's `scope` dispatch
(164 ns vs 23 µs). At `n = 10-12` chili stays within 2-4× of sequential
while per-round rayon variants are 10-50× off. But Approach 4 is still
faster than the best chili base-case at every `n ≥ 12`, and chili
never beats sequential at `n ≤ 14` for either field.

The chili base-case sweep shows a clear sweet spot:
- **n=10-14**: bigger `base` (1024-2048) wins. Chili's recursion overhead
  dominates; keeping most of the work in one thread is cheapest.
- **n=16-20**: smaller `base` (128-512) wins. Parallelism benefits start to
  outweigh dispatch, and finer split improves load balance.

### 5. `par1_scope` beats `par1_pariter` across the board.

For every (field, n) point, manual `rayon::scope` + explicit chunking is
faster than `par_iter + par_chunks_mut`. The gap is largest at small n
(2-5× at n=10-14). This confirms the handoff hypothesis that `par_iter`
adds its own overhead on top of `rayon::scope`.

Still, *any* rayon-scope-per-round approach loses to sequential below
n=18 due to the 24 µs scope dispatch floor.

### 6. Fp128 and GF128 now cross over at the same `n`.

With Approach 4, both fields cross over at `n = 12`. In the old sweep
Fp128 crossed over a doubling earlier than GF128 (`n = 18` vs `n = 20`)
because the per-element Fp128 work is ~5× heavier (Solinas reduction vs
CLMUL + GF(2^128) reduction), and that made the dispatch-heavy parallel
variants look relatively better. Now that Approach 4's dispatch floor
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

1. **Use `par4_pinned` as the default for `n ≥ 12`.** It is the
   fastest variant at every `(field, n)` we measured for `n ≥ 12`, with
   speedups of 1.25-5.3× (GF128) and 2.45-6.6× (Fp128) over sequential.
   Crossover is `n = 12` for both fields.
2. **Use sequential at `n ≤ 10`.** Approach 4 loses by ~10% at `n = 10`
   (GF128: 4.23 vs 3.85 µs; Fp128: 13.3 vs 12.0 µs): the top-level
   broadcast is cheap, but the per-round atomic barriers across workers
   accumulate to 1-2 µs over ~6 parallel rounds, which is larger than
   the parallel speedup recovers at these sizes. A simple size check
   (`if initial_pairs < 1024 { delayed(...) } else { delayed_par4_pinned(...) }`)
   is the production-safe dispatch.
3. **Override `SUMCHECK_PINNED_WORKERS` only if you know the workload.**
   Default is `min(available_parallelism(), 8)`. For pure `n = 10`
   GF128 workloads, `SUMCHECK_PINNED_WORKERS=3` gives a measured
   3.66 µs (0.95× seq), which flips the sign on the n=10 crossover;
   but that hurts `n ≥ 14` throughput noticeably. Leave the default
   alone unless small-n is the dominant use case.
4. **Retire `par3_persistent` and the chili base-case sweep.** Approach
   4 dominates both. The chili variants are retained only because
   they're cheap to keep building; `par3_persistent` is kept for
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
  --features parallel_chili -- par4_pinned
```

Raw Criterion output:
- `target/parallelism-results/bench-full.log` (Approaches 1-3, original sweep)
- `target/parallelism-results/bench-par4-final.log` (Approach 4 final sweep, all 4 approaches)
