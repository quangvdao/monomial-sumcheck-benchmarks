# CPU (and later CPU + GPU) sum-check platform plan

> Authored 2026-04-17.
> Companion to [`sumcheck-parallel-productionization.md`](./sumcheck-parallel-productionization.md),
> which covers the parallel dispatch layer (shipped).

## Mission

Build the **best** sum-check proving platform one can cook up on CPU,
and later a hybrid CPU + GPU one (with Apple Metal as the first GPU
target on M-class silicon).

"Best" here means, in order:

1. **Fastest** wall-clock per round across the kernel shapes used by
   real provers (binius64, Jolt, hachi, and the long-tail internal
   experiments).
2. **Field-agnostic** trait-based API that downstream provers can
   adopt without rewriting their polynomial state, so optimizations
   land once and help everyone.
3. **Portable**: macOS arm64 (M-class) and Linux x86-64 (Zen 5 at
   minimum) ship together; WASM gracefully no-ops; Windows / BSD
   deferred.
4. **GPU-ready trait shape**: the CPU abstraction should be close
   enough to the GPU kernel pattern used in
   `~/Documents/SNARKs/jolt-cpp` that porting to Metal (and later CUDA
   via the same crate's trait) is *adding backends*, not *rewriting
   the protocol*.

The existing [`pinned-pool`](../../../../SNARKs/pinned-pool) and
[`sumcheck-parallel`](../../../../SNARKs/sumcheck-parallel) crates
established the parallel dispatch pillar. This plan covers the other
pillars and starts with the one that has the biggest immediate payoff:
**fused bind + eval kernels**, borrowed directly from the GPU shape in
jolt-cpp.

### Non-goals for this doc

- Re-litigating parallel-dispatch design. See
  [`sumcheck-parallel-productionization.md`](./sumcheck-parallel-productionization.md)
  for `PinnedPool`, D2 shrinkage, and the shipped trait surface.
- GPU work. GPU backends are a future pillar; this doc only asks that
  Phase 1's trait shape **not close that door**.
- Migrating binius64 / Jolt / hachi. That's tracked in Phase C of the
  companion doc and is downstream of the kernel work here.

## Pillars and ordering

| # | Pillar                                            | Status                                        |
|---|---------------------------------------------------|-----------------------------------------------|
| 0 | Parallel dispatch (`pinned-pool`, scheduler)      | Shipped, see companion doc                    |
| 1 | **Fused bind + eval CPU kernels**                 | **This doc, Phase 1**                         |
| 2 | SIMD backends (GF128 `pclmulqdq`, AVX-512 Fp128)  | Outlined, Phase 2                             |
| 3 | Coverage for the non-delayed kernel shapes        | Outlined, Phase 2                             |
| 4 | deg-3 (and higher) generic-`D` trait              | Outlined, Phase 3                             |
| 5 | Batched multi-sumcheck API                        | Deferred until a caller needs it (D3 in companion) |
| 6 | Metal backend for apple silicon                   | Outlined, Phase 4                             |
| 7 | CUDA backend                                      | Outlined, Phase 5                             |

Phases 2-7 are scoped later. This doc only commits to Phase 1.

---

## Phase 1: Fused bind + eval on CPU

### The pattern (as seen in jolt-cpp GPU kernels)

Reference kernels:

- `~/Documents/SNARKs/jolt-cpp/src/jolt-core/zkvm/instruction_lookups/gpu/ra_virtual_kernels/bind_eval_roundN.cuh`
  (`ra_bind_eval_roundN_kernel`, lines 38-199). For each output pair
  `i` of round k, reads 4 round-(k-1) elements, binds with r_{k-1}
  to produce `a0 = round_k[2i]` and `a1 = round_k[2i+1]`, writes
  them to the round-k buffer, then in the same pass accumulates
  round-k's eval partial from `(a0, a1)`. Round-k state is never
  re-read.
- `~/Documents/SNARKs/jolt-cpp/src/jolt-core/zkvm/ram/gpu/kernels/sumcheck.cuh`
  (`raf_bind_eval_kernel`, lines 73-119). Same idea, deg-2 product.

The fusion constraint: round r's reduce must fully complete before
r_r is known (Fiat-Shamir), so **bind-r + reduce-r can't be fused**;
the next best thing, and what jolt-cpp does, is **bind-(r-1) +
reduce-r**, which *is* possible because r_{r-1} was already observed
by the time we start round r.

### What the CPU code does today (unfused)

Sequential kernels (`src/sumcheck/fp128.rs:278-317`,
`src/sumcheck/gf128.rs:97-132`) run two full passes per round:

1. Reduce pass reads each pair, accumulates `(h0, h1, h_inf)`.
2. After `black_box`, bind pass reads the same pairs **again** and
   writes the next-round half-size buffer.

The parallel kernels mirror this:
`partial_triple_{fp128,gf128}` at `src/sumcheck/parallel/legacy.rs:728-799`
and `bind_chunk_{fp128,gf128}` at `legacy.rs:754-820` are called in
sequence by `par_sumcheck`
(`~/Documents/SNARKs/sumcheck-parallel/src/scheduler.rs:177-220`),
so the same pair gets loaded at least twice (once in reduce, once in
bind) every round.

Binius64 and Jolt (the Rust side) both share this unfused structure.

### Per-pair op accounting for the deg-2 delayed kernel

Counting loads/stores/subs/mul_adds/fmadd per round-r pair (pair = 2
round-r elements = 4 round-(r-1) elements for the fused path); reads
count f and g together:

| path                      | reads | writes | subs | mul_add | fmadd |
|---------------------------|------:|-------:|-----:|--------:|------:|
| unfused (bind-r + reduce-r in the flow) | 12    | 4      | 6    | 4       | 3     |
| **fused (bind-(r-1) + reduce-r)**       | **8** | 4      | 6    | 4       | 3     |

Savings: 4 reads per pair (~33% of loads) by avoiding the re-read of
the freshly-bound round-r state. Expected wall-clock delta is smaller
than the read-count ratio (the bottleneck on 128-bit fields is the
`mul_add` throughput, not L1 loads), but the fused form is **strictly
fewer instructions** issued and opens further ILP (shared
`(f01-f00)` / `(g01-g00)` subexpressions across the bind and the
finite-difference pass). We should expect ~5-15% wall-clock on M4 in
the deg-2 GF128/Fp128 kernels; more on the GF128 NEON path (cheaper
`fmadd` relative to loads), less on the Fp128 Solinas `mul_add` path
(more compute per pair).

Across a full sum-check of L rounds and N_0 initial pairs, the fused
path saves ~N_0 reads total (out of ~4 N_0 in the unfused path), on
top of the per-pair savings above.

### Where the savings don't show up

Round 0 cannot be fused (there's no r_{-1}). The bind for the last
round cannot be fused with a reduce either (there's no reduce-L
after). So the savings apply only to rounds 1..L-1; for very short
sum-checks (L ≤ 2) the fused path is at best neutral. This is fine
because small sum-checks are sequential-tail territory anyway.

### Trait change (in the `sumcheck-parallel` crate)

Add one method to `SumcheckRound`:

```rust
/// Bind round `round - 1` state using the previous challenge
/// `r_prev`, writing the result to the round-`round` buffer, and
/// simultaneously reduce pairs `[lo, lo + len)` of the freshly-bound
/// round-`round` state.
///
/// `lo` and `len` are in **round-`round` pair units**. Implementors
/// read 4*len elements from the round-(round-1) buffer at offset
/// `4*lo`, write 2*len elements to the round-`round` buffer at
/// offset `2*lo`, and return the reduce partial for those `len`
/// round-`round` pairs.
///
/// The default impl simply calls `bind_chunk` then `reduce_chunk`,
/// which is correct but un-fused. Override to fuse the two passes.
///
/// # Safety
///
/// Same as [`bind_chunk`](SumcheckRound::bind_chunk): caller
/// guarantees disjoint windows across concurrent invocations in
/// this round.
unsafe fn bind_then_reduce_chunk(
    &self,
    round: usize,
    lo: usize,
    len: usize,
    r_prev: &Self::Elem,
) -> Self::Partial {
    // SAFETY: forwarded from the caller's disjoint-windows contract.
    unsafe { self.bind_chunk(round - 1, lo * 2, len * 2, r_prev); }
    self.reduce_chunk(round, lo, len)
}
```

Rationale: backward-compatible default, so existing impls (and
downstream crates) keep working. Impls that can fuse override.

### Scheduler change

The per-phase inner loop in
`~/Documents/SNARKs/sumcheck-parallel/src/scheduler.rs:177-220`
currently issues `reduce → observe → bind` once per round. For fused
mode, the pattern becomes:

```text
// Only at the very start of a phase:
round 0 of phase: reduce_chunk(abs_round)
                  observe + release bind_go

// All subsequent rounds in the phase:
for r in 1..phase_rounds:
    bind_then_reduce_chunk(abs_round + r, ..., r_prev = challenges[r - 1])
    observe + release bind_go

// After the inner loop: one more bind to finish the phase's last
// challenge into the next phase's read buffer.
bind_chunk(last_abs_round, ..., r_last)
```

Count-preservation: the fused scheduler issues `phase_rounds` barrier
waits (one per reduce) and `phase_rounds` binds, identical to the
unfused scheduler; only the pairing shifts.

Concurrency invariants carry over. Worker `i`'s chunk in round-r
pair units is `[lo, lo + len)`; in the fused call it reads round-(r-1)
positions `[4 lo, 4 lo + 4 len)` and writes round-r positions
`[2 lo, 2 lo + 2 len)`. Ping-pong parity swaps every round, just as
today. Self-read next-round invariant ("worker i reads exactly what
it wrote last round") still holds because `[2 lo, 2 lo + 2 len)` is
the same range the *previous* round's fused call wrote.

**Toggle for A/B.** A new argument `use_fused_path: bool` on
`par_sumcheck` selects between the existing split protocol and the
fused protocol. Default: true. Bench harnesses exercise both; we cut
the toggle (always-fused) once we've confirmed the sign across
fields and hardware.

### Impl overrides (in this repo)

Field-specific fused kernels live in
`src/sumcheck/parallel/legacy.rs` alongside the existing
`partial_triple_*` / `bind_chunk_*`:

```rust
pub(super) fn bind_then_reduce_chunk_fp128(
    rf_prev: *const Fp128,
    rg_prev: *const Fp128,
    wf: *mut Fp128,
    wg: *mut Fp128,
    lo: usize,       // in round-r pair units
    n_pairs: usize,  // in round-r pair units
    r_prev: Fp128,
) -> (Fp128, Fp128, Fp128);
```

Inner loop (one iteration per round-r pair):

```text
let base_prev = (lo + j) * 4;
let f00 = *rf_prev.add(base_prev);
let f01 = *rf_prev.add(base_prev + 1);
let f10 = *rf_prev.add(base_prev + 2);
let f11 = *rf_prev.add(base_prev + 3);
// analog for g

let f0 = (f01 - f00).mul_add(r_prev, f00);
let f1 = (f11 - f10).mul_add(r_prev, f10);
let g0 = (g01 - g00).mul_add(r_prev, g00);
let g1 = (g11 - g10).mul_add(r_prev, g10);

*wf.add(2 * (lo + j))     = f0;
*wf.add(2 * (lo + j) + 1) = f1;
*wg.add(2 * (lo + j))     = g0;
*wg.add(2 * (lo + j) + 1) = g1;

h0.fmadd(f0, g0);
h1.fmadd(f1, g1);
h_inf.fmadd(f1 - f0, g1 - g0);
```

`Fp128DelayedRound::bind_then_reduce_chunk` and
`GF128DelayedRound::bind_then_reduce_chunk` in
`src/sumcheck/parallel/impls/{fp128,gf128}.rs` delegate to these
helpers, using `read_ptrs(round - 1)` for the prev buffer and
`write_ptrs(round - 1)` (= `read_ptrs(round)`) for the current
buffer. Parity is unchanged.

### Sequential fused kernel

For a clean A/B baseline without any parallel overhead, add
`sumcheck_deg2_delayed_fp128_fused` and `sumcheck_deg2_delayed_gf128_fused`
to `src/sumcheck/{fp128,gf128}.rs`. Shape:

```text
Round 0 (reduce-only, in-place state):
  for j in 0..half:
    load pair j, accumulate h0/h1/h_inf
  black_box partial
  read r_0

Rounds 1..L-1 (fused in-place, shrinks state):
  for j in 0..new_half:
    load 4 old-round elements, bind with r_{r-1}, write 2 new-round
    elements in place at positions 2j, 2j+1
    accumulate partial using the freshly bound (f0, f1) pair
  black_box partial
  read r_r

After loop:
  final bind with r_{L-1} to produce the 1-element state
  truncate
```

In-place safety: iteration j reads positions 4j..4j+3 and writes
positions 2j, 2j+1. For j >= 1, 4j > 2j + 1, so the read range is
strictly above the write range of the current iteration. Later
iterations j' > j read `[4 j', 4 j' + 4)`, which is above any prior
iteration's write range (`[0, 2j + 2)` for j' = j + 1 gives reads at
4j + 4, writes were at 2j + 1 < 4j + 4 even for j = 0). No aliasing.

### Correctness tests

- Extend `tests/parallel_delayed.rs` so the `check_{gf128,fp128}`
  helpers also run the fused path (both sequential and parallel) and
  assert bit-identical `(f, g)` against the unfused sequential
  reference, at the existing `n ∈ {4, 6, 8, 10, 12, 14}` sweep plus
  `n = 16`.
- Add a smaller `tests/fused_delayed_seq.rs` that checks the
  sequential fused kernel on its own (no parallel feature required)
  at `n ∈ {1, 2, 3, 4, 8, 12}`. `n ∈ {1, 2}` exercises the
  "round 0 + final bind only, no fused iterations" edge cases.

### Benchmarks

- `benches/sumcheck.rs`: add `delayed_fused` variants in the Fp128
  and GF128 groups, alongside the existing `delayed`. Uses the same
  `criterion::BatchSize::LargeInput` / shuffled-order structure as
  the existing entries so the two are directly comparable.
- `benches/sumcheck_parallel.rs`: add a pair of criterion groups
  (`fp128_fused` vs `fp128_unfused`, same for GF128) that drive
  `par_sumcheck` with `use_fused_path` set each way. Same `n` sweep
  as the existing bench.
- `examples/par4_ab`: extend with a `--fused` / `--unfused` / `--both`
  switch and log the fused numbers into
  `target/parallelism-results/` alongside the existing
  `aragorn-ab-*.log`.
- Final A/B write-up lands in a new
  `docs/notes/fused-bind-eval-ab.md`, linked from this plan and
  summarised (one paragraph, one table) in the companion doc's
  follow-ups list.

### Acceptance criteria for Phase 1

- All existing correctness tests continue to pass on M4 + aragorn.
- Sequential fused kernel (both fields) is bit-identical to
  sequential unfused across the full test sweep.
- Parallel fused kernel (both fields) is bit-identical to parallel
  unfused (= already bit-identical to sequential).
- Sequential fused bench is ≥ 5% faster than sequential unfused at
  `n ∈ {14, 16}` on at least one of M4 or aragorn, for both fields.
  If neither field hits this bar, treat as a negative result,
  document it like D1 in the companion doc, and shelve the CPU
  fused path for the delayed kernel (trait change stays because
  other kernel shapes and GPU backends will still want it).
- Parallel fused bench ≥ 3% faster than parallel unfused at
  `n ∈ {14, 16}` on M4, allowing for barrier overhead dominating at
  small chunk sizes.

Effort estimate: 2-4 days.

### Phase 1 task list

1. Add sequential fused kernels
   (`sumcheck_deg2_delayed_{fp128,gf128}_fused`) with in-place
   buffer shrink; plus `tests/fused_delayed_seq.rs`.
2. Add `delayed_fused` entries to `benches/sumcheck.rs` for both
   fields.
3. Extend `SumcheckRound` with `bind_then_reduce_chunk` and default
   impl in `~/Documents/SNARKs/sumcheck-parallel/src/field.rs`.
4. Update the scheduler to the round-0 + fused-rest + final-bind
   protocol, with a `use_fused_path: bool` arg on `par_sumcheck`.
5. Override `bind_then_reduce_chunk` on `Fp128DelayedRound` and
   `GF128DelayedRound` in `src/sumcheck/parallel/impls/*`.
6. Extend `tests/parallel_delayed.rs` to cover fused variants.
7. Extend `benches/sumcheck_parallel.rs` + `examples/par4_ab` with
   fused A/B.
8. Run the bench sweep on M4 + aragorn; write up
   `docs/notes/fused-bind-eval-ab.md`; cross-link from the
   companion doc's follow-up list.

---

## Phase 2+ (sketch only, no commitments yet)

### Phase 2: SIMD backends for the 128-bit field paths

- x86-64 GF128 via `pclmulqdq` / `vpclmulqdq` (currently a scalar
  fallback; adjacency noted in the companion doc's follow-up list).
  Closes most of the aragorn-vs-M4 GF128 gap.
- AVX-512 Fp128 path to replace the current 4×u128 scalar limb
  arithmetic. Larger delta than GF128 but more work.
- Both are kernel-local and don't need trait changes.

### Phase 3: Cover the non-delayed kernel shapes

From the companion doc's kernel table:
`deg2_projective`, `deg2_projective_1inf`, `deg2_eq_*`,
`deg2_eq_gruen_*`, `deg2_*_upper` (BN254), `deg2_*_packed` (BB5),
eventually deg-3. For each, the fused pattern from Phase 1 carries
over mechanically; the only new bit is that eq-factored and projective
variants emit a different `Partial` shape.

### Phase 4: deg-D generic trait

Replace the bespoke `(h0, h1, h_inf)` tuple with a generic
`Partial = [Self::Elem; D + 1]` (or similar), so deg-3 sum-checks can
share the same scheduler. Trigger: first consumer that needs deg-3
at scale.

### Phase 5: Metal backend for Apple silicon

The trait shape is the entry point. The Metal backend would implement
`SumcheckRound` by pushing `reduce_chunk` / `bind_then_reduce_chunk`
onto a `MTLComputeCommandEncoder`, with the host driving the
Fiat-Shamir transcript between dispatches. The fused method is
load-bearing here: Apple GPUs pay through the nose for bandwidth, so
the per-round re-read we save on CPU is a much bigger win on Metal.

### Phase 6: CUDA backend

Same shape as Phase 5, plugged into the CUDA kernels already in
jolt-cpp. The `sumcheck-parallel` trait becomes the unified
host-side abstraction; CPU / Metal / CUDA each provide a backend.

### Phase 7: Batched multi-sum-check (D3 in the companion doc)

Trigger when Jolt / hachi actually use it. Architecturally already
supported by `PinnedPool::broadcast_scoped`; needs
`par_sumcheck_batch<R>(states: &mut [R], ...)` on the crate side.

---

## Cross-references

- Companion plan for parallel dispatch:
  [`sumcheck-parallel-productionization.md`](./sumcheck-parallel-productionization.md).
- Parallel results log: [`../../PARALLELISM.md`](../../PARALLELISM.md).
- Jolt-cpp reference kernels:
  - `~/Documents/SNARKs/jolt-cpp/src/jolt-core/zkvm/instruction_lookups/gpu/ra_virtual_kernels/bind_eval_roundN.cuh`
  - `~/Documents/SNARKs/jolt-cpp/src/jolt-core/zkvm/ram/gpu/kernels/sumcheck.cuh`
- First downstream migration target (binius64):
  `~/Documents/SNARKs/binius64/crates/ip-prover/src/sumcheck/bivariate_product.rs:43-99`
  (see companion doc Phase C1).
