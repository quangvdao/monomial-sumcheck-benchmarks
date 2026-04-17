# Parallelism design discussion (2026-04)

> Companion to [`PARALLELISM.md`](../../PARALLELISM.md). Records the
> follow-up conversation about (a) the theoretical optimality of the
> Approach 4 design, (b) what it would take to ship it across SNARK
> codebases, and (c) the parallel-prefix / sequential-tail decomposition.
>
> Source: chat session, 2026-04-16. Pre-implementation handoff is at
> [`../handoffs/parallelism-2026-04.md`](../handoffs/parallelism-2026-04.md).

## 1. In what sense is Approach 4 "optimal"?

It is a strong **local optimum** for one sumcheck call on M4-class
hardware at small `n`. It is **not** provably optimal in any
strong sense. Three frames are useful.

### 1a. Work-span / Brent's theorem

Total work \(W \approx F \cdot 2^n\), span \(S \approx n \cdot (F + B(k))\)
where \(F\) is per-pair arithmetic cost and \(B(k)\) is per-round
barrier cost at `k` workers. Brent's bound:

\[
T_p \geq \frac{W}{p} + S = \frac{F \cdot 2^n}{p} + n \cdot (F + B(k))
\]

Approach 4 gets close:

- **Work term tight.** Even contiguous chunking, `W/p` is sharp.
- **Span term tight to ~2x.** Two `AtomicUsize` waits per round is
  the minimum sync for reduce-then-bind. `B(k)` is ~100-200 ns at
  small `k` on M4, bounded below by cross-cluster L2/LLC latency.

Theory left on the table:

- **Reduce aggregate is `O(k)`, not `O(log k)`.** Worker 0 sums all
  `k` partials sequentially after the reduce barrier
  (`parallel.rs:1429-1433`). At `k = 2` this is identical; at `k = 8`
  it costs ~3 cache-line loads (~300 ns). A tournament/pairwise-tree
  reduce would close this. Worth doing if we push to `k ≥ 16`.
- **Round-to-round pipelining.** Workers that finish round `r`'s bind
  early could start round `r+1`'s reduce while worker 0 is still
  aggregating. In our benches the challenge is pre-generated so this
  is free; in production sumcheck the next challenge depends on the
  current round's output via Fiat-Shamir, blocked by a hash. **Not
  transferable.**
- **Adaptive `n_active` per round.** We pick `n_active` once at call
  entry. As chunks shrink, the optimal `n_active` shrinks too. We
  currently fall off the cliff to sequential (§3 below); a smoother
  taper (8 → 4 → 2 → seq) would help mid-sized `n`.

### 1b. Dispatch floor

Our 305 ns 2-worker broadcast is ~4x the cost of a single chili
`join` (164 ns), which is itself ~2-3x the hardware floor: one
cross-core cache-line round-trip + one L2 store to publish a task
pointer. On M4, cross-cluster coherence is ~40-100 ns (per Dougall
Johnson's M1 reverse-engineering, inherited by M4). So:

| Mechanism | Latency | Multiple of HW floor |
|-----------|---------|----------------------|
| Theoretical floor (one cache-line RT) | ~50 ns | 1x |
| Chili `scope.join` | 164 ns | ~3x |
| **Pinned pool, k=2** | **305 ns** | **~6x** |
| Pinned pool, k=8 | 681 ns | ~13x |
| Rayon `scope` | 23 µs | ~460x |

The gap from chili to us is from (a) the `done` atomic increment +
spin-wait (chili avoids by using callback-on-completion), and (b) all
pool workers spin on the epoch even when inactive, costing extra
cache-line touches.

### 1c. Could a fundamentally different approach do better?

Yes, but only by changing the problem shape:

1. **Batch multiple independent sumchecks.** If the application runs
   `B` sumchecks concurrently, do one broadcast per sumcheck per
   worker, zero per-round sync, each worker drives one whole sumcheck
   end-to-end. Throughput wins by `B x` and crossover moves down to
   `n ≈ 4-6`. The plonky3 / jolt pattern.
2. **GPU.** Wrong regime at small `n` (kernel launch ~µs, H2D copy
   dwarfs the problem). For `n ≥ 20`, one A100 beats 16 M4 P-cores by
   ~10x.

### 1d. Literature

Roughly in order of relevance:

- **Blumofe & Leiserson, "Scheduling Multithreaded Computations by
  Work Stealing"** (JACM 1999). Foundation of Rayon and Cilk. Explains
  why work-stealing has `O(p · S)` synchronization overhead, exactly
  the problem at small `n`.
- **Acar, Charguéraud, Rainey, "Heartbeat Scheduling: Provable
  Efficiency for Nested Parallelism"** (PLDI 2018). Theoretical basis
  for chili-like non-parking schedulers. Proves `O(W/p + S)` with
  bounded per-join overhead.
- **Mellor-Crummey & Scott, "Algorithms for Scalable Synchronization
  on Shared-Memory Multiprocessors"** (TOCS 1991). MCS locks and the
  tournament barrier. Canonical reference for the `O(log k)` barrier
  we're not currently using.
- **Hensgen, Finkel, Manber, "Two Algorithms for Barrier
  Synchronization"** (Int. J. Parallel Programming 1988). Older but
  readable presentation of the tournament barrier.
- **Blelloch, "Programming Parallel Algorithms"** (CACM 1996). Clear
  exposition of the work-span model as a design tool.
- **Chase & Lev, "Dynamic Circular Work-Stealing Deque"** (SPAA 2005).
  If we ever roll our own deque.
- **Dougall Johnson's M1 microarchitecture blog posts**
  (`dougallj.wordpress.com`). Only solid public source on Apple
  Silicon cache-coherence latencies.

There is essentially **no** good academic literature on parallel
sumcheck specifically. Setty's Spartan, Thaler's book, and the Jolt
paper all wave at parallelism but don't analyze the dispatch-floor
issue. Most useful prior art is in code: plonky3's
`eval_eq_batch_common`, `radix_2_dit_parallel`, the binius64 size
gate.

## 2. Productionizing as a rayon replacement for sumchecks

See [`docs/plans/sumcheck-parallel-productionization.md`](../plans/sumcheck-parallel-productionization.md)
for the actionable plan. The considerations identified in discussion
were:

- **Packaging**: extract `PinnedPool` into its own crate (generic
  over closure); a thin `sumcheck-parallel` crate on top. Default
  build pulls neither; opt-in via cargo feature.
- **Field abstraction**: a `DelayedSumcheckField` trait with
  `Accum`, `fmadd`, `reduce` so one generic implementation covers
  GF128, Fp128, BN254, BB ext, etc.
- **Platform support**: macOS uses `QOS_CLASS_USER_INTERACTIVE`;
  Linux needs `sched_setaffinity` to a P-core mask plus optional
  `SCHED_FIFO`; Windows uses `SetThreadAffinityMask` +
  `THREAD_PRIORITY_TIME_CRITICAL`. Without P-core pinning, fall back
  to spin-then-yield (Approach 3's mechanism).
- **Pool lifecycle**: current `OnceLock` global is fine for CLI
  provers but bad for long-running services (workers spin forever).
  Expose `PinnedPool::new(config) -> Arc<PinnedPool>` for explicit
  control alongside `PinnedPool::global()`.
- **Rayon coexistence**: if a consumer has called
  `rayon::ThreadPoolBuilder::build_global()` already, our pool is
  additional. Document this; recommend `rayon::current_num_threads()
  + pool_size <= physical_cores`.
- **Determinism**: GF128 XOR and Fp128 limb addition are
  associative+commutative, so parallel partial sums are bit-identical
  to sequential. Lock this into property-test CI.
- **Observability**: optional `tracing` spans behind a feature flag.
- **Migration order**: binius64 first (origin of the question, has
  the existing `PAR_THRESHOLD_LOG_HALF` gate to replace), Jolt next
  (multiple sumchecks, multiple fields, needs the trait first), then
  hachi.
- **Testing matrix**: Linux x86-64 (Intel + AMD), Linux aarch64
  (Graviton), macOS arm64 (M1-M4), macOS x86-64. Thread counts 1, 2,
  4, 8, 16, 32, 64.

## 3. Parallel-prefix / sequential-tail decomposition

**Yes, we already do this.** The mechanism is `par3_scope_rounds` at
[`src/sumcheck/parallel.rs:103-117`](../../src/sumcheck/parallel.rs).
With `PAR3_MIN_PAIRS_PER_WORKER = 8`:

```text
fn par3_scope_rounds(initial_pairs, n_workers, n_rounds) -> usize {
    if n_workers <= 1 || n_rounds == 0 { return 0; }
    let initial_per_worker = initial_pairs / n_workers;
    if initial_per_worker < PAR3_MIN_PAIRS_PER_WORKER { return 0; }
    let ratio = initial_per_worker / PAR3_MIN_PAIRS_PER_WORKER;
    let max_scope_rounds = ratio.ilog2() as usize + 1;
    max_scope_rounds.min(n_rounds)
}
```

After the parallel prefix runs in `pool.broadcast_scoped`, the tail
runs sequentially:

```text
pool.broadcast_scoped(n_workers, &worker_body);
// ... swap buffers, truncate to live_len ...
if scope_rounds < n_rounds {
    super::gf128::sumcheck_deg2_delayed_gf128(f, g, &challenges[scope_rounds..]);
}
```

Concrete numbers (default pool size 8, adaptive `n_active`):

| n  | `n_workers` | `scope_rounds` | sequential tail rounds |
|----|-------------|----------------|------------------------|
| 10 | 2           | 6              | 4                      |
| 12 | 4           | 7              | 5                      |
| 14 | 8           | 7              | 7                      |
| 16 | 8           | 9              | 7                      |
| 18 | 8           | 11             | 7                      |
| 20 | 8           | 13             | 7                      |

So at `n = 20` we run 13 rounds in the pool and hand off the last 7
(operating on `[128, 64, 32, 16, 8, 4, 2]` pairs) to the sequential
kernel. Those rounds together are a few microseconds; parallelizing
them would cost more in barrier overhead than they save.

### Cleanliness for production

Mostly good, with three known issues:

1. **Single threshold across fields.** `PAR3_MIN_PAIRS_PER_WORKER = 8`
   is shared between GF128 and Fp128. Fp128 per-pair work is ~5x
   heavier so it can justify smaller chunks. Production fix: per-field
   trait constant `DelayedSumcheckField::MIN_PAIRS_PER_WORKER`.
2. **No per-round `n_active` shrinkage.** We pick `n_active` once and
   then fall off the cliff to sequential. A smoother taper
   (8 → 4 → 2 → seq) would help `n = 14-16`. ~30 LOC, ~20% gain at
   those sizes. Defer until benches justify.
3. **No explicit "is parallel worth it" surface.** Currently the
   library decides internally. Production API should accept an
   `Option<&PinnedPool>` and the library decides whether to use it.

### Where the pattern comes from

This (parallel prefix, sequential tail) is the same template used by
plonky3's `eval_eq_batch_common` and `radix_2_dit_parallel`. The
"shrinking chunk" insight from parallel FFT: stop parallelizing once
per-worker work is smaller than barrier cost. Our `ratio.ilog2()`
calculation is the discrete version of "run parallel rounds while
`per_worker_chunk >> r > barrier_floor / per_element_cost`".
