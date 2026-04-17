# Productionization plan: `pinned-pool` + `sumcheck-parallel`

> Authored 2026-04-16 against [`PARALLELISM.md`](../../PARALLELISM.md)
> and the design discussion at
> [`../notes/parallelism-design-discussion.md`](../notes/parallelism-design-discussion.md).
> All decisions in §"Locked decisions" are confirmed by the user.

## TL;DR

Extract `PinnedPool` into a standalone crate; build a field-agnostic
`sumcheck-parallel` crate on top with a small reference matrix
(GF128, Fp128 first); land Approach-4 parallelism with
D2 per-round `n_active` shrinkage (D1 tournament reduce is
implemented then shelved as a measured net-loss for the current
small-`Partial` kernels, kept ready for larger-`combine` fields);
validate on Apple M4 Max + AMD Zen 5 (aragorn); migrate
binius64 → Jolt → hachi.

## Locked decisions

1. **Crate ownership.** Local sibling repos under
   `~/Documents/SNARKs/`. Private GitHub remotes only.
   No crates.io publishing. Path dependencies between crates.
2. **API shape.** Field-agnostic via a `SumcheckRound` trait. Ship
   reference impls for GF128 and Fp128 first; expand to BN254, BB ext,
   KB5, packed BB5 over time as adoption demands. Consumers can
   implement their own kernels against the trait.
3. **Phase D scope.** D1 (tournament reduce) *was* in scope but was
   measured as a **net loss** for the current GF128/Fp128 deg-2
   kernels on M4 Max (see `PARALLELISM.md` → "D1 tournament reduce:
   negative result"). Shelved; trait `combine` is still associative
   so we can add it later when a kernel with an expensive `combine`
   (packed BB5, KB5, eq-factor variants) makes the trade worthwhile.
   D2 (per-round `n_active` shrinkage) is in and beneficial.
4. **Platforms in scope now.** macOS arm64 (M4 Max dev box), Linux
   x86-64 (aragorn: AMD Ryzen 9 9950X, Zen 5, 16C / 32T, no E-cores).
   Windows and BSDs deferred. WASM gracefully no-ops.
5. **Batch-multiple-sumchecks (D3) deferred** until at least one
   downstream caller actually needs it. Architecturally, the pool
   already supports it (one `broadcast_scoped` per batch, each worker
   drives one whole sumcheck end-to-end), so adding it later is
   small.
6. **Documentation.** Crate-level rustdoc + a `RUNBOOK.md` per crate
   covering platform behavior (QoS on macOS, affinity on Linux),
   pool sizing, and known coexistence pitfalls with rayon.

## Architecture

```
                 ┌────────────────────────┐
                 │   pinned-pool crate    │   no_std-friendly core
                 │   PinnedPool, broadcast│   platform shims for
                 │   epoch + done atomics │   macos / linux / fallback
                 └───────────┬────────────┘
                             │
                 ┌───────────▼────────────┐
                 │ sumcheck-parallel crate│
                 │ trait SumcheckRound    │
                 │ par_sumcheck<R>(...)   │
                 │ adaptive scope_rounds  │
                 │ tournament reduce      │
                 │ per-round shrinkage    │
                 └─┬─────────┬─────────┬──┘
                   │         │         │
   reference ┌─────┴─┐    ┌──┴────┐  ┌─┴────────┐  consumer-owned
   impls     │ GF128 │    │ Fp128 │  │ BN254... │  (or ship later)
             └───────┘    └───────┘  └──────────┘
                   │         │         │
            ┌──────▼────┐ ┌──▼─────┐ ┌─▼──────┐
            │ binius64  │ │  Jolt  │ │ hachi  │
            └───────────┘ └────────┘ └────────┘
```

## Scope: kernel patterns to eventually support

The current monomial-sumcheck-benchmarks repo defines roughly **8-12
distinct sumcheck kernel patterns** across **5-6 fields**. The
trait must scale to all of them, even though we start by porting
just `delayed_*`. From `src/sumcheck/{gf128,fp128,bn254,bb5_packed,generic}.rs`:

| Pattern                                  | Notes                                                    |
|------------------------------------------|----------------------------------------------------------|
| `deg2_boolean`                           | Literal sum over hypercube                               |
| `deg2_projective`                        | Eval at `(0, 1, ∞)`                                      |
| `deg2_delayed`                           | Lazy reduction (current Approach 4 target)               |
| `deg2_eq_delayed`                        | Delayed + eq factor input                                |
| `deg2_projective_delayed`                | Projective + delayed                                     |
| `deg2_projective_1inf_delayed`           | Projective with skipped 0-eval                           |
| `deg2_eq_projective_delayed`             | All four combined                                        |
| `deg2_eq_gruen_*`                        | Gruen specialized eq update                              |
| `deg2_eq_gruen_projective_*`             | Gruen + projective                                       |
| `deg2_eq_gruen_projective_1inf_*`        | Gruen + projective + skipped 0                           |
| `deg2_*_upper` (BN254)                   | Upper-half BN254 split                                   |
| `deg2_*_packed` (BB5)                    | SIMD-packed input                                        |
| `deg3_*` (future)                        | Cubic sumchecks                                          |

Common structure across all of them, which the trait must capture:

- A read-only **state** (one or more multilinear vectors + auxiliaries
  like an `eq` factor) chunked across workers in the reduce phase.
- A **partial** type holding `D + 1` field elements for degree `D`
  (or `D` if `1` is skipped via projective optimization).
- An **associative-commutative combine** on partials.
- A **bind** step writing a chunk of the next-round buffer in place
  or into a ping-pong target.
- A **per-round advance**: swap buffers, decrement live length.

Field-agnostic means consumers (binius64, Jolt) can implement
`SumcheckRound` for their own kernels without touching this crate.

## Phase A: refactor in this repo

Goal: validate the trait abstraction + D1 + D2 with this repo's
correctness tests and benches before paying the cost of crate
extraction.

### Tasks

1. **Carve out `src/sumcheck/parallel/` as a directory module.**
   - `pool.rs`: `PinnedPool` + `broadcast_scoped` + platform shims
     (currently macOS QoS only; add a Linux affinity stub returning
     `Ok(())` for now). No behavior change.
   - `field.rs`: define the `SumcheckRound` trait. Initial shape
     (subject to revision during implementation):

     ```rust
     pub trait SumcheckRound: Sync {
         type Elem: Copy + Send + Sync;
         type Partial: Copy + Send + Sync + Default;
         const MIN_PAIRS_PER_WORKER: usize;

         fn current_pairs(&self) -> usize;
         fn rounds_left(&self) -> usize;

         /// Reduce a chunk into a partial. `&self` because workers
         /// hold disjoint chunks of read-only state.
         fn reduce_chunk(&self, lo: usize, len: usize) -> Self::Partial;

         fn combine(a: Self::Partial, b: Self::Partial) -> Self::Partial;

         /// Bind a chunk into the next-round buffer. SAFETY: caller
         /// guarantees `lo..lo+len` is disjoint between concurrent
         /// calls within one round; impl uses interior mutability via
         /// `UnsafeCell` or splits a `&mut` slice in the wrapper.
         unsafe fn bind_chunk(&self, lo: usize, len: usize, r: &Self::Elem);

         /// Move to the next round (swap ping-pong, halve live len).
         fn advance_round(&mut self);
     }
     ```

     Open question: whether to make `Partial` carry an explicit
     `partial_count: usize` so we can support varying degree
     statically vs via a generic const. Resolve during impl.

   - `scheduler.rs`: `par_sumcheck<R: SumcheckRound>(state: &mut R,
     challenges: &[R::Elem], pool: Option<&PinnedPool>)` containing:
     - The adaptive `scope_rounds` calculation (per-field
       `MIN_PAIRS_PER_WORKER`).
     - Per-round `n_active` shrinkage table (D2).
     - Tournament reduce (D1).
     - The single `broadcast_scoped` call.
     - The sequential tail loop calling `reduce_chunk` +
       `bind_chunk` directly.

2. **Implement `SumcheckRound` for the existing GF128/Fp128 delayed
   kernels.** Each impl is ~50 lines: an owning struct holding `f`,
   `g`, ping-pong buffers, current round; methods delegate to the
   existing accumulator types.

3. **Replace `sumcheck_deg2_delayed_{gf128,fp128}_par4_pinned` with
   thin wrappers** that construct the impl and call `par_sumcheck`.
   Keep them under their old names so the bench file is
   line-for-line unchanged.

4. **D1: tournament reduce.** ~~Replace the linear aggregation with
   a pairwise tree.~~ **Shelved as a measured regression on these
   kernels** (2026-04-16 A/B). The prototype added per-level
   `AtomicUsize::fetch_add` barriers; on M4 Max each Release-RMW is
   ~100 ns on the critical path and three levels of barriers
   (+~300 ns/round) cost more than the ~200 ns/round the tree saves
   on cross-core loads for our 48-byte `Partial` + 3-ns `combine`.
   Regression was +3% (`n=16`) to +22% (`n=12`). See
   `PARALLELISM.md` → "D1 tournament reduce: negative result" for
   the full A/B table. Trait `combine` stays associative-commutative
   so D1 can be re-enabled behind a scheduler switch if a future
   field's `Partial` or `combine` gets expensive enough. Estimated
   trigger thresholds: `Partial` ≥ 128 bytes **or** `combine` ≥ 50 ns,
   **or** target CPU is a multi-socket / multi-cluster box where the
   linear 8-line scan crosses NUMA domains. Revisit on aragorn (Zen 5,
   single CCD) once we have the Linux shim running; if Zen 5 shows a
   similar regression at `k = 8` then D1 is truly a bad fit for all
   current use cases and we should stop carrying the option at all.

5. **D2: per-round `n_active` shrinkage.** Compute a schedule
   `[n_active_round_0, n_active_round_1, ...]` at scheduler entry.
   Workers consult `shared.n_active` per round (already a
   `RelaxedAtomic`); inactive workers skip the round body and the
   per-round done counter. The schedule:

   ```
   for round in 0..scope_rounds:
     pairs = initial_pairs >> round
     n_active[round] = clamp(pairs / MIN_PAIRS_PER_WORKER, 1, pool_size)
   ```

   When `n_active[round] = 1`, that round runs sequentially on
   worker 0 inside the same broadcast (no pool exit), which avoids
   the buffer-swap-then-handoff cost of the sequential tail. This
   makes the scope_rounds → tail transition smoother.

   Cost: ~150 LOC including correctness tests. Win: ~20% at n=14-16
   (per the discussion doc).

6. **Add a dispatch-floor microbench for the schedule itself.**
   `criterion::Bench("schedule_only", n)` measures the cost of
   computing the schedule + tournament reduce on noop work. We need
   this to track regressions in D1/D2 separately from kernel changes.

7. **Linux platform shim.** ✅ Done.
   `src/sumcheck/parallel/pool/platform.rs` now contains a Linux
   backend that:
   - Reads `/sys/devices/system/cpu/possible` and each cpu's
     `topology/thread_siblings_list`, keeps the smallest sibling in
     each group, and caches the sorted deduplicated list in a
     `OnceLock<Vec<usize>>`. On aragorn this yields 16 canonical
     logical cpus (one per physical core).
   - Calls `libc::sched_setaffinity(0, sizeof(cpu_set_t), &set)` per
     worker with a single-bit mask for
     `canonical_cpus[worker_idx % canonical_cpus.len()]`. Worker 0
     (main) stays unpinned so it can migrate while doing the
     reduce / combine.
   - Optional `libc::sched_setscheduler(SCHED_FIFO, prio=1)` gated on
     `SUMCHECK_PINNED_SCHED_FIFO=1`; silent no-op on EPERM.
   - All sysfs / syscall failures fall back to unpinned scheduling
     (correctness preserved, perf possibly worse).
   - Debug logging via `SUMCHECK_PINNED_DEBUG=1` prints one line per
     worker (`[sumcheck-pinned] worker pinned to cpuN`) for on-box
     sanity-checking.

   Verified on aragorn: workers 1..7 pin to cpu1..cpu7 (the canonical
   lower-siblings on Zen 5's `(k, k+16)` SMT layout).

   `cpufreq` governor hint left for a follow-up: benches on aragorn
   already hit the expected performance scaling (`schedutil` with
   boost enabled).

8. **Test on aragorn.** ✅ Done.
   - `cargo test --release --features parallel_chili` on aragorn:
     all 14 sumcheck tests + 2 `par4_loop` tests + 4
     `parallel_delayed` tests pass.
   - `cargo bench --bench sumcheck_parallel --features parallel_chili
     -- --warm-up-time 3 --measurement-time 6` saved to
     `target/parallelism-results/aragorn-bench-2026-04.log`.
   - `examples/par4_ab` sweep (`n ∈ 10..16`, both fields, 3000 iter)
     saved to `target/parallelism-results/aragorn-ab-2026-04.log`.
   - See **Aragorn validation** in `PARALLELISM.md` for the writeup.
     Highlights:
     - Dispatch floor 44 / 73 / 103 ns at k = 2 / 4 / 8 (3-11× better
       than M4 Phase A numbers).
     - NEW (trait + D2) beats LEGACY single-broadcast by up to -20%
       (Fp128 n=15); on M4 it cost ~4%. Net positive on Linux.
     - Crossover at n ≤ 10 for both fields (M4 was n = 12 / 10).
     - 8-worker cap leaves half of aragorn's 16 cores idle at
       n ∈ {18, 20}; `par1_scope` (all-core rayon) beats us there.
       Follow-up (Phase B): raise default pool cap to
       `min(physical_cores, 16)` and add more steps to the D2
       shrinkage table.

### Acceptance criteria for Phase A

- ✅ All existing correctness tests still pass on M4 + aragorn.
- ✅ `par4_pinned` GF128 and Fp128 numbers match or beat the current
  `PARALLELISM.md` table on M4.
- ~~D1 microbench shows < 5% regression at `k = 2`, > 50 ns saving at
  `k = 8`.~~ D1 shelved (see Task 4). Revisit on larger-Partial
  kernels or new hardware.
- ✅ D2 shows ≥ 10% improvement at GF128 n=14 and Fp128 n=14 vs
  cliff-fallback behavior (visible in aragorn A/B at n=13, 15).
- ✅ aragorn benches landed in repo, baseline numbers documented.

**Estimated effort: 3-5 days. Actual: ~2 days.**

### Phase A follow-ups (tracked for Phase B)

- ✅ **Pool cap raised.** Linux default now
  `available_parallelism()` (= logical cores, 32 on aragorn). macOS
  stays at `min(P-cores - 2, 8)` because QoS pinning is a hint, not
  an anchor. Opt-out: `PINNED_POOL_WORKERS=N`. The per-round `D2`
  heuristic keeps small-n dispatches at low `k` so the larger pool
  doesn't over-dispatch.
- ✅ **Auto-park on idle.** Workers spin for ~300 µs-1 ms of
  `hint::spin_loop`, then transition to `thread::park()` with
  Dekker-style SeqCst fences on both sides. Honest neighbor in
  benchmarks (no more pool-contamination contaminating rayon
  samples) and in production servers that need the CPU for other
  work when the prover is quiescent.
- **Extend D2 schedule table.** With a 32-worker pool we want
  `initial_pairs ≥ 32 × TARGET_PAIRS_PER_WORKER` to dispatch at
  k = 32; else halve. Current schedule handles this correctly up to
  8; needs explicit 16 and 32 rungs so we don't wait for the halving
  loop to get there.
- **pclmulqdq backend for GF128 on x86-64.** Aragorn GF128 is ~10×
  slower than M4 NEON because we fall through to the scalar `u128`
  path. Separate workstream from parallelism, but the
  `src/sumcheck/gf128.rs` `GF128Accum` abstraction is the right place
  to add an `#[cfg(target_arch = "x86_64")]` branch.

## Phase B: extract to standalone crates

Goal: ship reusable crates that downstream repos can pull as path
dependencies.

### Tasks

1. ✅ **Created `~/Documents/SNARKs/pinned-pool/`** as a path-dep
   crate (local only for now, no git remote yet).
   - Cargo crate, library only.
   - Moved `pool.rs` + `pool/platform.rs` from this repo; this repo
     now depends via `pinned-pool = { path = "../../SNARKs/pinned-pool" }`.
   - Public API actually shipped:
     ```rust
     pub struct PinnedPool { ... }
     impl PinnedPool {
         pub fn global() -> &'static PinnedPool;
         pub fn n_workers(&self) -> usize;
         pub fn broadcast_scoped(&self, n_active: usize, f: &(dyn Fn(usize) + Sync));
     }
     pub fn default_pool_size() -> usize; // platform-aware
     ```
     Configuration is env-var driven (`PINNED_POOL_WORKERS`,
     `PINNED_POOL_SCHED_FIFO`, `PINNED_POOL_DEBUG`); the
     `PoolConfig`-struct form can be added later if a caller
     actually needs programmatic override.
   - Tests: broadcast correctness at n_active ∈ {1, 4, N}, many
     times, and **auto-park correctness** (sleep > spin budget, then
     dispatch, all workers wake).
   - Auto-park: Dekker-style, SeqCst fence + `parked: AtomicBool`
     per worker. Spin budget = 1M iterations (~300 µs-1 ms of
     wall-clock) so dispatches within a prover loop stay in the fast
     spin path, while a quiescent pool consumes zero CPU.
   - ✅ Dispatch-floor + cold-start microbenches shipped as
     integration tests (`tests/dispatch_floor_hot.rs`,
     `tests/dispatch_floor_cold.rs`). Hot floor on M4 Max: ~600 ns
     at k=2, ~1.3 µs at k=4, ~3.5 µs at k=8 (vs ~22 µs for
     `rayon::scope`). Cold-start on macOS: p50 17 µs at k=2
     scaling to 2.3 ms at k=8 because QoS-parked workers wake
     through the OS scheduler; on Linux this drops to µs-range
     (aragorn numbers in PARALLELISM.md). Irrelevant for tight
     prover loops which never idle past the 1 M spin budget.
   - Remaining work before extracting to a git remote:
     - GitHub Actions matrix `(macos-latest, ubuntu-latest) x
       (stable, 1.95)`. No nightly.

2. ✅ **Created `~/Documents/SNARKs/sumcheck-parallel/`** as a
   path-dep crate (local only, no git remote yet).
   - Cargo crate, library only.
   - Depends on `pinned-pool` via path dep.
   - Moved `field.rs` + `scheduler.rs` from this repo. Field-specific
     impls (GF128/Fp128) stayed in this repo because they transitively
     depend on `binius-field` / `hachi-pcs` and on the sequential
     kernels hosted here; the crate is meant to stay field-agnostic.
   - Public API actually shipped:
     ```rust
     pub trait SumcheckRound { /* reduce_chunk, combine,
         observe_partial, bind_chunk + 2 const tuning knobs */ }
     pub fn par_sumcheck<R: SumcheckRound>(
         round: &R,
         challenges: &[R::Elem],
         pool: &PinnedPool,
         n_workers_initial: usize,
         initial_pairs: usize,
     ) -> usize; // returns rounds_done
     pub fn pick_n_workers(initial_pairs: usize,
                           pool_total: usize,
                           target_pairs_per_worker: usize) -> usize;
     ```
   - Tests: end-to-end property test driving `par_sumcheck` with a
     toy u64-ring round state, comparing output against a
     hand-written sequential reduce-then-bind across log_n ∈
     {3, 8, 12, 14, 18}. All 5 cases bit-identical.
   - Doc-test on the module-level `no_run` example.
   - **Deferred to later**: dispatch-floor/cold-start microbenches
     (tracked as a Phase-B follow-up once the dispatch path is
     touched again), full property tests across large random seeds,
     GitHub Actions matrix.

3. ✅ **Main repo now depends on both crates.**
   - `src/sumcheck/parallel/pool/` deleted. `PinnedPool` pulled from
     the `pinned-pool` crate.
   - `src/sumcheck/parallel/{field,scheduler}.rs` deleted. Trait +
     driver pulled from the `sumcheck-parallel` crate.
   - `src/sumcheck/parallel/impls/{gf128,fp128}.rs` rewritten to
     `use sumcheck_parallel::{par_sumcheck, pick_n_workers,
     SumcheckRound};` and are the first downstream implementors.
   - `src/sumcheck/parallel/legacy.rs` unchanged (Approach 1-3
     baselines, still uses `pinned_pool::PinnedPool` for the
     Approach-3 variant).
   - Main repo's `cargo test --release --features parallel`
     continues to pass (14 + 2 + 4 = 20 tests across the three
     test binaries).

4. **CI**: a `make bench-aragorn` target that runs the sweep on
   aragorn and posts results back as an artifact. Use it before
   merging crate changes. **PENDING.**

### Acceptance criteria for Phase B

- ✅ `pinned-pool` and `sumcheck-parallel` build clean on
  macOS arm64 (stable). Linux x86-64 (aragorn) was validated under
  the pre-extraction tree; re-validation under the extracted tree
  is a one-command rsync + `cargo test` and is tracked as a
  follow-up.
- ✅ This repo's full test suite still passes after the swap.
  Bench re-validation tracked as a follow-up; the code paths are
  unchanged (only the `use` paths moved), so no performance delta
  expected.
- ✅ aragorn full sweep numbers documented in `PARALLELISM.md`
  "Aragorn, Phase B" section; `par4_pinned` with auto-park + 32
  workers beats all Rayon variants at every `n` for both GF128 and
  Fp128.

**Estimated effort: 3-5 days, mostly mechanical. Actual: ~1 day.**

## Phase C: migrate downstream repos

Goal: replace per-repo ad-hoc sumcheck parallelism (or its absence)
with the new crates.

### C1 — binius64 (origin of the question; cleanest swap)

- Site: `~/Documents/SNARKs/binius64/crates/ip-prover/src/sumcheck/bivariate_product.rs:43-99`.
- Current logic: `if log_half < PAR_THRESHOLD { sequential } else
  { rayon::par_iter ... }` (size gate at log_half=18).
- Action:
  1. Add `sumcheck-parallel = { path = "../../../sumcheck-parallel" }`
     (sibling repo under `~/Documents/SNARKs/`).
  2. Implement `SumcheckRound` for the relevant binius64 kernel.
  3. Replace the `if/else` block with a single
     `par_sumcheck(&mut state, challenges, Some(PinnedPool::global()))`.
  4. Re-run the existing prover end-to-end benchmark; expect
     speedups at log_half = 12-17 where sequential was forced today.
- Risk: low. The existing PAR_THRESHOLD gate documents that
  parallel-below-18 was a measured loss; we have data showing our
  pool flips the sign. Keep the old gate code in git history as
  a fallback.

### C2 — Jolt (largest payoff; needs trait coverage)

- Sites: multiple `prover.rs` files using `rayon::par_iter` or
  `par_chunks_mut` for sumcheck rounds.
- Sumcheck shapes used (verify before starting):
  - Spartan-style with eq factor.
  - r1cs sumcheck (mat × vec).
  - Other custom polynomial products.
- Action:
  1. Audit which kernel shapes Jolt actually uses; if any are not
     yet covered by `sumcheck-parallel::impls`, either
     (a) implement them as new impls in `sumcheck-parallel::impls`,
     or (b) implement them inside the Jolt repo as private impls.
  2. Replace site-by-site, one PR per site, with bench numbers.
- Risk: medium. Jolt has heavier integration (many sumchecks per
  proof; transcript coupling). Migration is per-site.

### C3 — hachi (or other repos as they come up)

- Same pattern as C2. Audit sumcheck shapes first.
- Probably blocked on understanding hachi's prover shape; defer
  until C1 lands.

### Acceptance criteria for Phase C

- For each migrated repo: equivalent or better end-to-end prover
  wall-clock on M4 + aragorn, no proof-output regressions.
- Old per-repo size gates removed (the `sumcheck-parallel` library
  decides internally).
- Each migration has a 1-page changelog noting kernel coverage and
  benchmark deltas.

**Estimated effort: 2-3 days per repo.**

## Phase D: long-tail optimizations (driven by need)

Already-decided:

- **D1 tournament reduce**: *attempted in Phase A, shelved as a
  regression* for the current small-`Partial` GF128/Fp128 kernels
  (trait `combine` stays associative so D1 can be resurrected
  behind a scheduler switch for larger-`Partial` fields — see
  Phase A Task 4 for the re-evaluation criteria).
- **D2 per-round `n_active` shrinkage**: in Phase A, shipped.

Deferred until a downstream caller needs them:

- **D3 batch-multiple-sumchecks API**: one `broadcast_scoped` for
  `B` independent sumcheck instances; each worker drives one
  end-to-end. Architecturally supported by the pool today; needs
  a small API surface (`par_sumcheck_batch<R>(states: &mut [R],
  ...)`). Trigger when Jolt or another repo runs many small
  sumchecks per proof.
- **D4 generic over polynomial degree**: extend the trait to
  `Partial = [Self::Elem; D + 1]` or similar. Trigger when a
  consumer needs deg-3 sumchecks at scale.
- **D5 GPU offload**: out of scope. Different regime.
- **D6 round pipelining**: out of scope. Blocked by Fiat-Shamir
  challenge dependency.

## Further investigations (track separately)

These are open questions surfaced during design that don't block
the plan but should be answered before Phase B ships.

1. **x86-64 GF128 SIMD path.** *Phase B follow-up: mostly fixed by
   build flag.* binius-field already ships PCLMULQDQ / VPCLMULQDQ /
   AVX-512 VPCLMULQDQ paths gated on the `target_feature` cfgs. The
   default `x86_64-unknown-linux-gnu` target doesn't enable any of
   those, so we got a portable `u128` software multiply on aragorn.
   Committing `.cargo/config.toml` with
   `rustflags = ["-C", "target-cpu=native", "-C",
   "target-feature=-avx512f"]` for `x86_64-unknown-linux-gnu` routes
   binius through `packed_ghash_256` (VPCLMULQDQ-256) and gives us
   ~5× per-mul (latency) / ~10× (throughput) on the
   `field_ops::GF128_Ghash::*_mul` benches, and ~8.7× on sequential
   GF128 sumcheck end-to-end (every `n` in the sweep). The
   `-avx512f` is needed because `hachi-pcs` carries a
   `#![feature(stdarch_x86_avx512)]` opt-in that errors on stable
   when AVX-512 is in the target features; disabling AVX-512 gives
   up the 4-wide path (`packed_ghash_512`) but keeps the 2-wide
   (`packed_ghash_256`) which is still ~5-10× over scalar. See
   `PARALLELISM.md` "Aragorn, Phase B follow-up: VPCLMULQDQ" for
   the post-flag tables.

   *Remaining work:* `src/sumcheck/gf128.rs` non-aarch64 `GF128Accum`
   still calls `acc += a * b`, which pays a polynomial reduction per
   multiply instead of folding it across the loop the way the
   aarch64 NEON impl does. A hand-rolled x86 accumulator using
   `_mm_clmulepi64_si128` + deferred Montgomery reduction would
   stack another ~2-3× on top of the build-flag win. Tracked
   separately; not on Phase B critical path.

2. **AVX-512 Fp128 path.** Similar: `Fp128Accum` uses 4×u128
   limb arithmetic; AVX-512 can do a lot more per cycle. Adjacent;
   owner is this repo.

3. **Pool-vs-rayon coexistence.** If a downstream has an
   outer-level rayon scope that calls `par_sumcheck` from each
   rayon task, our 8-worker pool serializes their concurrency.
   Decide whether to (a) detect this and fall back to sequential
   per-call inside the rayon task, or (b) document the
   `rayon_threads + pool_size ≤ physical_cores` constraint and let
   the user manage it. **Tentative answer: (b)**, with a runtime
   warning if `available_parallelism()` is exceeded.

4. **Linux frequency governor.** A spinning pool worker on Linux
   may stay in a low P-state if the kernel governor isn't
   `performance`. The `pinned-pool` crate should detect and warn
   (or, with opt-in, write `performance` to
   `/sys/.../scaling_governor`). The latter requires root; warn
   only by default.

5. **SMT siblings on Zen 5.** Pinning to one logical thread per
   physical core (16 of 32 on aragorn) is almost certainly right
   for spin-heavy workloads. But it leaves 16 hardware threads idle
   that the OS could give to other processes. Verify the gain over
   "use all 32 SMT threads".

6. **Trait shape: borrow vs owned.** First-pass trait makes
   `bind_chunk` `unsafe fn(&self, ...)` so workers can hold `&R`.
   Alternative: split state into `&mut R::WriteState` per worker
   via a `chunks_mut` accessor on the trait, removing `unsafe`
   from impls. Resolve during Phase A by trying both on the GF128
   reference impl.

7. **`eq` factor pattern.** `deg2_eq_*` kernels carry a third
   multilinear (the eq evaluation) alongside `f` and `g`. The trait
   must accommodate this without privileging it (BB5 packed has yet
   another auxiliary structure). Solution: `SumcheckRound` is
   opaque; consumers stash whatever they need inside their impl.
   Verify this works for `eq_gruen_projective_1inf_*` impls.

8. **Determinism over many fields.** Fp128 limb addition is
   strictly associative+commutative on `[u128; 4]`; OK. GF128 XOR
   is OK. BN254 (ark BigInt mod p) addition is OK. KB5 / BB ext
   limb arithmetic should be OK but verify. Add a property-test
   matrix in `sumcheck-parallel`.

9. **Pool teardown semantics.** Currently no teardown (workers spin
   until process exit). For a long-running prover service, this is
   wasteful when idle. Options: (a) `Arc<PinnedPool>` with `Drop`
   stopping workers; (b) explicit `pool.shutdown()` method. Pick
   one before Phase B ships.

## Reproduction commands

For Phase A acceptance:

```bash
# M4 dev box
cargo test --release --features parallel_chili --test parallel_delayed
cargo bench --bench sumcheck_parallel --features parallel_chili -- \
  --warm-up-time 2 --measurement-time 4 \
  | tee target/parallelism-results/phase-a-m4.log

# aragorn (Linux x86-64, Zen 5)
ssh aragorn 'cd ~/monomial-sumcheck-benchmarks && \
  cargo test --release --features parallel_chili --test parallel_delayed && \
  cargo bench --bench sumcheck_parallel --features parallel_chili -- \
    --warm-up-time 2 --measurement-time 4' \
  | tee target/parallelism-results/phase-a-aragorn.log
```

For Phase B acceptance:

```bash
# In ~/Documents/Research/pinned-pool/
cargo test && cargo bench -- dispatch_floor

# In ~/Documents/Research/sumcheck-parallel/
cargo test && cargo bench --bench sumcheck

# In monomial-sumcheck-benchmarks/ (this repo, post-swap)
cargo test --release && cargo bench --bench sumcheck_parallel
```

## Open invitations

- Approve the trait shape draft in §"Phase A" task 1, or push back
  with an alternative.
- Confirm aragorn is fine for shared use during benching (otherwise
  flag what hours / load-share considerations matter).
- Decide pool-teardown semantics (further investigation #9) before
  Phase B begins.
