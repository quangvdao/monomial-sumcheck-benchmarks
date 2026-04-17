# Handoff: parallelism-library shootout on monomial sumcheck kernels

> **Status: superseded by [`PARALLELISM.md`](../../PARALLELISM.md) (2026-04).**
> This doc is the original scoping note from before implementation.
> It lists three approaches; the actual work tested four (`pinned`,
> the pinned-pool + doorbell design, was added during implementation
> and is the production recommendation). Predictions in this doc about
> chili performance and the `n = 8-10` crossover were not borne out.
> Read `PARALLELISM.md` for conclusions; read this only for historical
> context on how the question was originally framed.

- **Handoff reason:** end of session; scoping complete, implementation not started
- **Summary:** Investigate how to parallelize the per-round reduce-then-bind loop in the existing sumcheck kernels with minimum per-round dispatch overhead. Compare three approaches (manual chunked `rayon::scope`, `chili`, persistent-pool with atomic-barrier per-round sync) against the current single-threaded baseline, across field types and problem sizes. The benchmark harness in this repo is nearly ideal for this (self-contained, no external prover, `black_box` instead of a protocol driver, pre-generated challenges).
- **Goal and scope:** Decide which parallelization library/pattern minimizes per-round dispatch overhead enough that the parallel path beats sequential at small sizes (n = 16 → log_half = 15, even lower if possible), not just at large sizes (n = 24). Produce quantitative data per field and per size. Upstreamable output would be a sibling `benches/sumcheck_parallel.rs` and a `PARALLELISM.md` results doc.
- **Current state:** Investigation only. No code changes in this repo. Working tree clean (`git status` shows `## main...origin/main`, nothing to commit). No stashes.

## Parent-context origin (do not reopen)

This work spun out of an investigation in the sibling repo `~/Documents/SNARKs/binius64`. That investigation produced findings at `~/Documents/SNARKs/binius64/scripts/tracing-ab/findings.md` (lines 304-441 specifically). The relevant facts:

- Rayon's `par_iter().map().reduce()` call has a **~19 µs fixed dispatch floor** on Apple Silicon (Apple M4 Max, `aarch64-apple-darwin`, `OptimalPackedB128`). See the sweep table at `findings.md:333-357` for raw numbers across `log_half = 0..=20`.
- Crossover where parallel first beats sequential was found at `log_half = 17-18` for a trivial bivariate round. For smaller sizes (almost all rounds of a sumcheck), the parallel path is **many times slower** than sequential because of Rayon's worker wake-up cost.
- The production fix there was just a size gate at `log_half < 18 → sequential` (`binius64/crates/ip-prover/src/sumcheck/bivariate_product.rs:43-99`).

The open question is: can we push the crossover down to `log_half ≈ 8-10` with a better dispatch strategy? Three candidate approaches were identified, but never benchmarked. This repo is a cleaner place to test them.

## Why this repo is the right sandbox

- **Zero external prover plumbing.** Kernels are plain `fn(&mut Vec<F>, &mut Vec<F>, &[F], ...)` at `src/sumcheck/generic.rs:6-189` (and field-specific variants). No `SumcheckProver` trait, no transcript, no `rayon::ThreadPoolBuilder` lurking anywhere.
- **Pre-generated challenges + `black_box` verifier.** The round message computation uses `black_box((h0, h1, h_inf))` (e.g. `src/sumcheck/generic.rs:31`). There is no Fiat-Shamir between rounds, so the "cross-round sync" in `persistent` below is purely synthetic: workers can do all rounds back-to-back with just a barrier for each `black_box`. That's *exactly* what we want to benchmark — the per-round barrier cost in isolation.
- **Many field types, one harness.** BN254 (256-bit prime), Fp128 (128-bit prime), BB4/BB5 (BabyBear deg-4/5 extensions), KB5 (KoalaBear deg-5), GF128 (binary-char 128-bit GHASH). Different per-element costs → different sequential-to-parallel ratios → different crossover points.
- **Sizes sweep the interesting range.** `ns = [16, 20, 24]` for the large fields; `ns = [16, 20]` for BB/KB and Fp128/GF128. At n = 16, first-round `half = 2^15 = 32768` packed elements, last-round `half = 1`. At n = 24 BN254, first-round `half = 2^23`. This spans both "parallel is obviously winning" and "parallel is obviously losing" regimes in the *same* prover run.
- **Single-threaded baseline is already tuned and published.** The paper numbers (M4 Max, single-threaded, thin LTO) are the oracle against which parallel must win. `src/sumcheck/*` contains codegen-sensitive tuning (see `README.md:44-45`) — **do not touch those kernels**. Only add new parallel wrappers that call the existing sequential code per chunk.

## Machine

- Apple M4 Max, `aarch64-apple-darwin`. 14 cores (10 P + 4 E) reported by typical `sysctl -n hw.ncpu`; Rayon will pick up the full count. For chili, the default pool uses all cores too.
- Rust toolchain pinned at `1.94.0` in `rust-toolchain.toml`.
- `RUSTFLAGS="-C target-cpu=native"` is **not** set in `Cargo.toml`; confirm whether the paper's numbers were measured with or without it before comparing parallel results. Current `[profile.bench]` just sets `lto = "thin"`.

## The three approaches to benchmark (in order of implementation effort)

Full rationale and drop-in code snippets are in the originating conversation. Short version:

### 1. Manual chunked `rayon::scope` (no new top-level deps)

Replace the sequential inner loop of one kernel with `scope.spawn` calls on exactly `num_threads` contiguous slices. Kills Rayon's recursive-split-tree overhead and the `MultiZip` / `reduce` plumbing. Keeps the parked-worker wake-up cost (~15 µs on Apple Silicon).

Add `rayon = "1"` to `[dev-dependencies]` of `Cargo.toml` (already in `Cargo.lock` transitively).

For the `sumcheck_deg2_delayed_gf128` kernel (`src/sumcheck/gf128.rs:91-126`), the parallel version looks like:

```rust
fn par_reduce_and_bind(f: &mut Vec<GF128>, g: &mut Vec<GF128>, r: GF128) -> (GF128, GF128, GF128) {
    let half = f.len() / 2;
    let n_threads = rayon::current_num_threads().min(half.max(1));
    let chunk = half.div_ceil(n_threads);

    // Reduce phase: each worker owns a chunk of the 2N-element input and produces a (h0, h1, h_inf) triple.
    let mut partials = vec![(GF128::default(), GF128::default(), GF128::default()); n_threads];
    rayon::scope(|s| {
        for (i, slot) in partials.iter_mut().enumerate() {
            let lo = i * chunk;
            let hi = ((lo + chunk).min(half)) * 2;
            let slice_f = &f[lo * 2..hi];
            let slice_g = &g[lo * 2..hi];
            s.spawn(move |_| {
                let mut h0 = GF128Accum::zero();
                let mut h1 = GF128Accum::zero();
                let mut h_inf = GF128Accum::zero();
                for j in 0..(hi - lo * 2) / 2 {
                    let f0 = slice_f[2 * j]; let f1 = slice_f[2 * j + 1];
                    let g0 = slice_g[2 * j]; let g1 = slice_g[2 * j + 1];
                    let df = f1 - f0; let dg = g1 - g0;
                    h0.fmadd(f0, g0); h1.fmadd(f1, g1); h_inf.fmadd(df, dg);
                }
                *slot = (h0.reduce(), h1.reduce(), h_inf.reduce());
            });
        }
    });
    let (h0, h1, h_inf) = partials.iter().fold(
        (GF128::ZERO, GF128::ZERO, GF128::ZERO),
        |(a0, a1, ai), (b0, b1, bi)| (a0 + *b0, a1 + *b1, ai + *bi),
    );

    // Bind phase: independent; parallelize same way.
    // (Pull into a second scope or use par_chunks_mut; either is fine.)
    // Truncate on the main thread.
    f.truncate(half); g.truncate(half);
    (h0, h1, h_inf)
}
```

The *point* of this approach is that **every round of the outer `for round in 0..n` loop pays one fresh `rayon::scope` dispatch**. We expect this to be slightly faster than `par_iter` at the same thread count but still dominated by worker wake-up.

### 2. Chili (non-parking heartbeat scheduler)

Add to `Cargo.toml`:

```toml
chili = "0.2"
```

API is `chili::Scope::global()` (or a per-pool `ThreadPool::new(Config { ... })`) with `scope.join(|s| ..., |s| ...)`. Pattern is recursive divide-and-conquer. Write a `par_reduce_bind_chili` that recurses on the slice halves down to a base case (tune `base` ∈ [32, 512] packed elements).

Chili claims ~100-300 ns dispatch (vs Rayon's ~19 µs) because workers don't park. The cost: all worker threads spin (with light backoff) when idle, so ambient CPU when idle is non-zero. For a benchmark this is invisible; for a deployed prover service it's a consideration.

Known chili caveats (from chili README / rayon issue #1235): no scoped/spawn API, no thread-index queries. Neither matters here because the kernels don't use per-thread scratch.

**Expected:** crossover at `log_half ≈ 8-10` instead of 17-18, if the theory is right.

### 3. Persistent-pool + atomic-barrier across all rounds

Spend one pool dispatch for the entire `sumcheck_deg2_delayed_*` call. Workers own contiguous slices across all rounds; main thread coordinates the (synthetic) `black_box` handoff per round via atomic counters.

This is the structural win: you pay N round dispatches in Approaches 1 and 2, but 1 dispatch total in `persistent`. For sumcheck specifically, N ≈ 16-24. If Rayon dispatch is the floor, `persistent` should crush 1 at small sizes.

Shape (pseudocode, concrete version lives in the parent-context conversation):

```rust
struct Shared {
    partials: Vec<UnsafeCell<(F, F, F)>>,
    round_done: Vec<AtomicUsize>,
    challenge: Vec<UnsafeCell<MaybeUninit<F>>>,
    challenge_ready: Vec<AtomicUsize>,
    n_workers: usize, n_rounds: usize,
}

rayon::scope(|s| {
    for worker in 0..n_workers - 1 {
        s.spawn(move |_| worker_loop(shared, worker, ...));
    }
    // main thread acts as worker 0 plus the "driver" that calls black_box
    // and publishes the next round's challenge.
    for round in 0..n_rounds {
        // compute own partial; publish; spin-wait for all workers;
        // black_box the reduced triple; publish challenge for next round;
        // fold own chunk.
    }
});
```

Shrinking-chunk wrinkle: once `half / n_workers < some_threshold`, the per-worker barrier costs more than the serial work. Fall back to sequential on the main thread for the tail rounds (cheap: maybe the last 8 rounds out of 24).

**Evidence that the pattern works for analogous algorithms (even though the *original* claim that plonky3/halo2 use this for sumcheck was wrong — they don't have multilinear sumcheck):** `~/Documents/SNARKs/plonky3/multilinear-util/src/eq_batch.rs:561-635` (`eval_eq_batch_common`) does the moral equivalent for a *tree recursion*: a sequential prefix of recursion levels on the main thread, then one `par_chunks_exact_mut(...).for_each(...)` where each worker does all the remaining recursion levels in-thread with no cross-thread sync. Same for `~/Documents/SNARKs/plonky3/dft/src/radix_2_dit_parallel.rs:158-163` and `:296-315` — the FFT does all `mid` butterfly layers in-thread between two global bit-reversal barriers. These are the template.

## Concrete plan for the next agent

### Phase 0: baseline sanity (30 min)

1. `cd ~/Documents/Research/monomial-sumcheck-benchmarks`
2. `cargo check --benches` — confirm the tree still builds cleanly on Rust 1.94.
3. `cargo test --test bb5_packed_eq` — confirm kernels are unchanged.
4. Run a **short** baseline on a single target to anchor numbers before any changes:
   ```bash
   cargo bench --bench sumcheck -- 'sumcheck_deg2/GF128/delayed/20' --warm-up-time 2 --measurement-time 5
   ```
   Record the median from Criterion's output. Do the same for `Fp128/delayed/20` and `BN254/delayed/20`.

### Phase 1: `rayon_scope` (manual chunked `rayon::scope` (half day)

1. Add `rayon = "1"` to `[dev-dependencies]` in `Cargo.toml`.
2. Create a **new** file `benches/sumcheck_parallel.rs` (register it in `Cargo.toml` as `[[bench]]` with `harness = false`). Mirror the structure of `benches/sumcheck.rs` but:
   - Call parallel variants of the kernels.
   - Add a suffix `_rayon_scope` to the Criterion benchmark IDs to avoid clashing with existing IDs.
   - Include the same sizes and fields: start with GF128 and Fp128 for speed, add BN254 if time permits.
3. Add a sibling module `src/sumcheck/parallel.rs` (declared from `src/sumcheck/mod.rs`). Put the parallel wrappers there so existing kernels stay untouched. One wrapper per sequential kernel we want to test.
4. Run `cargo bench --bench sumcheck_parallel -- 'GF128/delayed_rayon_scope/20'` and compare against the Phase-0 median.
5. Also compare against a `par_iter` version (the "control") by implementing a `par_iter_control` variant that uses the standard `into_par_iter().fold().reduce()` pattern from rayon. This isolates the "manual scope vs par_iter" overhead.

**Success criterion for Phase 1:** parallel beats sequential on at least one (field, size) pair at n = 20, and you know empirically how much of the 19 µs Rayon floor went away. Report result with a short note.

### Phase 2: `chili` (half day)

1. Add `chili = "0.2"` to `[dev-dependencies]`.
2. Add chili variants to `src/sumcheck/parallel.rs`. Use recursive `scope.join` with a base case of ~64-256 elements. Parameterize `base` so it can be swept.
3. Add `_chili` variants to `benches/sumcheck_parallel.rs`.
4. Sweep `base` in `[32, 64, 128, 256, 512]` for one field at n = 20 to find the sweet spot, then use that for all other runs.
5. Compare against Phase 1.

**Success criterion for Phase 2:** crossover point (smallest n at which parallel beats sequential) drops meaningfully vs Phase 1. If chili doesn't help, that tells us the dominant cost *isn't* worker wake-up and we should reconsider Phase 3.

### Phase 3: `persistent` (rayon scope + persistent workers + atomic barrier (1-2 days)

Only do this if Phase 2 shows chili clearly helps. The structural refactor is only worth it if parking is really the floor.

1. Add a `persistent` variant per field that wraps *the entire outer loop* (all `n_rounds` rounds) inside one `rayon::scope`.
2. Design the shared state carefully (see shape above). Use `crossbeam_utils::CachePadded<AtomicUsize>` for the barriers to avoid false sharing.
3. Think carefully about the shrinking-chunk tail: at some round, the per-worker chunk is too small to justify the barrier, and you should fall back to sequential for the tail.
4. Benchmark against Phases 1 and 2.

**Success criterion for Phase 3:** n = 16 parallel wins by more than Phases 1 and 2, and you know (from the sweep) at which round the shrinking-chunk tail should cut over.

### Phase 4: writeup

Produce `~/Documents/Research/monomial-sumcheck-benchmarks/PARALLELISM.md` in the same style as the findings doc in binius64. Required content:

- Raw per-(field, size, approach) table.
- Crossover-n per field for each approach.
- Per-approach ambient CPU observation (chili spins; the other two don't).
- Recommendation: which approach to port back into production prover codebases (binius64, hachi, jolt).

## Context files

Do read (they are the oracle and the templates):

- `src/sumcheck/generic.rs` (89 lines, canonical kernel shapes).
- `src/sumcheck/gf128.rs:91-253` (accumulator-based delayed kernels for the field most likely to benefit).
- `src/sumcheck/fp128.rs:278-486` (prime-field 128-bit delayed kernels).
- `src/sumcheck/bn254.rs:130-360` (largest field; sequential is slowest → parallel wins earliest).
- `benches/sumcheck.rs` (the existing bench shape to mirror).
- `README.md` (runs matrix, including the warning at lines 44-45: don't touch the hot kernels).

Do **not** read unless you specifically need the pattern for barrier/atomic code:

- The conversation in `~/Documents/SNARKs/binius64` about `PAR_THRESHOLD_LOG_HALF`. The decision was already made there. This repo is about testing a *different* question.

Reference templates:

- `~/Documents/SNARKs/plonky3/multilinear-util/src/eq_batch.rs:561-635` — single-dispatch + sequential prefix pattern.
- `~/Documents/SNARKs/plonky3/dft/src/radix_2_dit_parallel.rs:158-315` — in-worker multi-level work with global barriers.

## Key decisions and rationale

- **Separate `benches/sumcheck_parallel.rs` file** instead of adding parallel IDs to `benches/sumcheck.rs`. Keeps the paper-reproducing bench runnable independently and prevents accidental interference (e.g. Criterion shuffling changes medians).
- **Separate `src/sumcheck/parallel.rs` module** instead of adding `#[cfg(parallel)]` gates on the existing kernels. The existing kernels are codegen-tuned (`README.md:44-45`); touching them is a risk.
- **Start with GF128 and Fp128, not BB4/BB5.** Per-element cost is higher (128-bit multiplication vs ~160-bit extension arithmetic with base-field packings) → sequential baseline is slower per element → parallel wins at smaller sizes. Easier signal.
- **Do not benchmark packed BB5** (the `bb5_packed.rs` kernels) for parallelism until the simple cases work. The packing already provides 4x SIMD speedup; mixing SIMD packing with thread-level parallelism needs extra care to not double-count chunks.
- **`dev-dependencies` not `dependencies`** for rayon and chili. This repo's public library (`src/lib.rs`) should stay minimal; only the benches need them.

## Blockers / Errors

None. Nothing has been attempted yet.

## Open questions / Risks

- **Is the M4 Max figure of ~19 µs Rayon floor accurate on this machine?** It was measured in binius64 with `OptimalPackedB128`. Re-measure it here by running a trivial `(0..1).into_par_iter().sum::<u64>()` benchmark as the first experiment.
- **Is `RUSTFLAGS="-C target-cpu=native"` on or off for the paper numbers?** Verify with the repo author (`git blame` points to recent commits `64c1f6e`, `daeb0ba`, `e3b201f` — review those commit messages for clues). Set it consistently across baseline and parallel runs.
- **Criterion's `iter_batched` with `LargeInput` does a `clone()` per iteration.** For BN254 at n = 24 that's a 2^24 × 32-byte = 512 MiB clone, which may dominate wall time. If that's the case, restructure to use `iter_batched_ref` with a pre-allocated scratch buffer (std-allocated, not in `MaybeUninit`).
- **Thread count floor.** On the M4 Max, Rayon uses 14 threads. For `n_workers > half`, the chunking logic must clamp. Already handled in the snippet above (`n_threads.min(half.max(1))`) but easy to get wrong for the tail.
- **Chili's `Scope::global()` pool allocation is lazy.** First call may show a one-time warm-up cost larger than subsequent calls. Use Criterion's warmup to hide this, but verify by running the first criterion group twice.

## Cleanup needed

None yet.

## Tests and commands run during scoping

None in this repo. Scoping was read-only inspection of:

- `README.md`, `Cargo.toml`, `rust-toolchain.toml` — confirmed layout, pinned toolchain 1.94.0, pinned field crates.
- `git log --oneline -20` — repo is clean; last commit `64c1f6e` adds the README of the current layout.
- `git status` — clean working tree on `main` tracking `origin/main`.
- Source files listed in the "Context files" section.
- `rg -n 'rayon|par_iter|parallel'` across the tree — only transitive rayon via `p3-maybe-rayon` in `Cargo.lock`; no first-party parallelism.

## Next steps (numbered)

1. Run Phase 0 sanity + baseline for GF128/Fp128/BN254 at n = 20.
2. Implement Phase 1 (manual chunked `rayon::scope`). Add dep, new bench file, new kernel module, one GF128 variant. Bench; compare.
3. If Phase 1 works for at least one (field, n), extend to Fp128 and BN254.
4. Phase 2: `chili`.
5. Decide based on (1-4) whether Phase 3 is worth it. If yes, implement. If no, close out with a `PARALLELISM.md` that documents "Rayon-dispatch cost is not the bottleneck on this workload" and what the actual bottleneck is.
6. Write `PARALLELISM.md` with the table.

## How to resume

```bash
cd ~/Documents/Research/monomial-sumcheck-benchmarks
cargo check --benches          # 1. confirm clean build
cargo test --test bb5_packed_eq # 2. confirm existing kernels agree with their tests
cargo bench --bench sumcheck -- 'sumcheck_deg2/GF128/delayed/20' \
    --warm-up-time 2 --measurement-time 5  # 3. record baseline median
```

Then read `src/sumcheck/gf128.rs:91-126` (the sequential kernel you'll wrap first) and `benches/sumcheck.rs:819-984` (the existing GF128 bench layout you'll mirror). Start Phase 1.
