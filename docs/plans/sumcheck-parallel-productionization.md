# Productionization plan: `pinned-pool` + `sumcheck-parallel`

> Authored 2026-04-16 against [`PARALLELISM.md`](../../PARALLELISM.md)
> and the design discussion at
> [`../notes/parallelism-design-discussion.md`](../notes/parallelism-design-discussion.md).
> All decisions in §"Locked decisions" are confirmed by the user.

## TL;DR

Extract `PinnedPool` into a standalone crate; build a field-agnostic
`sumcheck-parallel` crate on top with a small reference matrix
(GF128, Fp128 first); land Approach-4 parallelism with
two further optimizations (D1 tournament reduce, D2 per-round
`n_active` shrinkage); validate on Apple M4 Max + AMD Zen 5
(aragorn); migrate binius64 → Jolt → hachi.

## Locked decisions

1. **Crate ownership.** Local sibling repos under
   `~/Documents/SNARKs/`. Private GitHub remotes only.
   No crates.io publishing. Path dependencies between crates.
2. **API shape.** Field-agnostic via a `SumcheckRound` trait. Ship
   reference impls for GF128 and Fp128 first; expand to BN254, BB ext,
   KB5, packed BB5 over time as adoption demands. Consumers can
   implement their own kernels against the trait.
3. **Phase D scope.** D1 (tournament reduce) **and** D2 (per-round
   `n_active` shrinkage). Both are required because consumers will
   call sumcheck across MANY shapes and sizes; the static
   8-or-sequential cliff is too coarse.
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

4. **D1: tournament reduce.** In `scheduler.rs`, replace the
   linear `for slot in partials` aggregation with a pairwise tree.
   Cost: ~50 LOC. Win: ~100 ns at `k = 8`. Zero-cost at `k = 2` (the
   tree degenerates to a single combine).

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

7. **Linux platform shim**.
   - `pool/platform/macos.rs`: existing `pthread_set_qos_class_self_np`
     code.
   - `pool/platform/linux.rs`: `sched_setaffinity` to a
     one-thread-per-physical-core mask (parse `/sys/devices/system/cpu`
     for SMT topology), optional `SCHED_FIFO` if process has
     `CAP_SYS_NICE`, `cpufreq` performance governor hint via
     `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` (no
     hard requirement; document if missing).
   - `pool/platform/fallback.rs`: no-op; `spin_until_ge` falls back to
     spin-then-yield (already implemented).

8. **Test on aragorn.** SSH config already set up
   (`Hostname 100.100.234.84`, `User omid`). Run:
   - `cargo test --release --features parallel_chili --test parallel_delayed`
   - `cargo bench --bench sumcheck_parallel --features parallel_chili -- dispatch_floor`
   - Full sweep, results checked into
     `target/parallelism-results/aragorn-bench-2026-04.log` for
     comparison with M4 Max.

   Expected: AMD Zen 5 has different cache-coherence behavior
   (single-CCD on 9950X) and ~4x lower per-cycle pclmul throughput
   than M4's NEON, so GF128 sequential will be slower; the relative
   parallel speedups should be similar or better.

### Acceptance criteria for Phase A

- All existing correctness tests still pass on M4 + aragorn.
- `par4_pinned` GF128 and Fp128 numbers match or beat the current
  `PARALLELISM.md` table on M4.
- D1 microbench shows < 5% regression at `k = 2`, > 50 ns saving at
  `k = 8`.
- D2 shows ≥ 10% improvement at GF128 n=14 and Fp128 n=14 vs current
  cliff-fallback behavior.
- aragorn benches landed in repo, baseline numbers documented.

**Estimated effort: 3-5 days.**

## Phase B: extract to standalone crates

Goal: ship reusable crates that downstream repos can pull as path
dependencies.

### Tasks

1. **Create `~/Documents/SNARKs/pinned-pool/`** as a new git repo.
   - Cargo crate, library only.
   - Move `pool.rs` + `pool/platform/*` from this repo.
   - Public API:
     ```rust
     pub struct PinnedPool { ... }
     pub struct PoolConfig {
         pub size: Option<usize>,         // None = min(available_parallelism, 8)
         pub pin_p_cores: bool,           // default true
         pub spin_budget: Option<u32>,    // None = pure spin (requires pinning)
         pub qos_user_interactive: bool,  // macOS only; default true
     }
     impl PinnedPool {
         pub fn global() -> &'static PinnedPool;
         pub fn new(config: PoolConfig) -> Arc<Self>;
         pub fn n_workers(&self) -> usize;
         pub fn broadcast_scoped(&self, n_active: usize, f: &(dyn Fn(usize) + Sync));
     }
     ```
   - Tests: ABA detection, shutdown on drop, broadcast correctness
     across 1, 2, 4, 8, 16 workers.
   - Benches: dispatch floor (k=1..16), shutdown latency.
   - CI: GitHub Actions matrix `(macos-latest, ubuntu-latest) x
     (stable, 1.94)`. No nightly.

2. **Create `~/Documents/SNARKs/sumcheck-parallel/`.**
   - Cargo crate, library only.
   - Depends on `pinned-pool` (path dep).
   - Move `field.rs` + `scheduler.rs` from this repo.
   - Move GF128/Fp128 reference impls into `src/impls/`.
   - Public API:
     ```rust
     pub trait SumcheckRound { ... }
     pub fn par_sumcheck<R: SumcheckRound>(
         state: &mut R,
         challenges: &[R::Elem],
         pool: Option<&PinnedPool>,
     );
     pub mod impls {
         pub mod gf128;
         pub mod fp128;
     }
     ```
   - Tests: bit-identical parallel-vs-sequential property tests over
     random inputs, all reference impls.
   - Benches: per-(field, n) sweep, mirror this repo's
     `benches/sumcheck_parallel.rs`.
   - Doc-tests on the public API.

3. **Update this repo to depend on the new crates.**
   - Replace `src/sumcheck/parallel/` with `[dev-dependencies]
     sumcheck-parallel = { path = "../../SNARKs/sumcheck-parallel" }`.
     (This repo lives in `~/Documents/Research/`; the new crates live
     in `~/Documents/SNARKs/` alongside binius64, hachi, etc.)
   - The repo continues to host the GF128/Fp128/BN254/BB ext kernels
     (those stay sequential and codegen-tuned), plus the bench harness
     calling `sumcheck-parallel`.
   - PARALLELISM.md updated to reflect the crate split.

4. **CI**: a `make bench-aragorn` target that runs the sweep on
   aragorn and posts results back as an artifact. Use it before
   merging crate changes.

### Acceptance criteria for Phase B

- `pinned-pool` and `sumcheck-parallel` build clean on
  `(macOS arm64, Linux x86-64) × (stable, 1.94)`.
- This repo's full test + bench suite still passes after the
  swap, with M4 numbers within 2% of Phase A.
- aragorn full sweep numbers documented and within expected range.

**Estimated effort: 3-5 days, mostly mechanical.**

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

- **D1 tournament reduce**: in Phase A.
- **D2 per-round `n_active` shrinkage**: in Phase A.

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

1. **x86-64 GF128 SIMD path.** Current `GF128Accum` has NEON
   intrinsics for aarch64 and a scalar fallback for everything else.
   On aragorn (Zen 5) the scalar path will be the bottleneck. Adding
   a `pclmulqdq` / `vpclmulqdq` fallback under
   `#[cfg(target_arch = "x86_64")]` would close most of the gap. Not
   strictly part of the parallelism work but adjacent and worth
   tracking. Owner: this repo, not the new crates.

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
