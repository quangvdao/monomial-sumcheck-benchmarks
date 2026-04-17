# Fused bind-eval A/B findings (2026-04-17)

> Phase 1 result of [`../plans/sumcheck-cpu-platform.md`](../plans/sumcheck-cpu-platform.md).
> Benchmark plumbing lives in
> [`examples/pinned_ab.rs`](../../examples/pinned_ab.rs) and
> [`benches/sumcheck_parallel.rs`](../../benches/sumcheck_parallel.rs).

## TL;DR

- **GF128 fused bind+reduce wins 3.6x - 7x across sequential and parallel
  paths** on Apple M4 Max once the inlining pathology in
  `binius_field::M128`'s trait impls is worked around.
- **Fp128 fused is ~5 - 10 % slower** than unfused across
  `n in {14, 18, 20}` and does not recover at large `n`. The Fp128
  path is compute-bound on
  [`hachi::Fp128::mul_add_raw_aarch64`](/Users/quang.dao/.cargo/git/checkouts/hachi-7638e1df81e5672d/7e81702/src/algebra/fields/fp128.rs) (a 35-instruction
  inline-asm block), so the extra cache traffic that fusion saves is
  immaterial and the longer serial dependency chain (bind -> reduce
  per iteration) hurts ILP.
- Conclusion: default to fused for GF128, keep unfused for Fp128. The
  `use_fused_path: bool` toggle on `sumcheck_deg2_delayed_*_pinned`
  exposes this choice to callers; both correctness paths remain tested
  in `tests/parallel_delayed.rs` and `tests/pinned_loop.rs`.
- **Delayed reduction does not help Fp128 Solinas** on CPU: the
  `[p, 2^128)` non-canonical sliver is only ~`c` wide (~2^8), so
  the 3-instruction savings on a relaxed `mul_add` are cancelled by
  the 2-instruction surcharge on every `sub` that has to tolerate
  non-canonical inputs. Instruction-count change per fused iteration
  is exactly zero, and empirically the relaxed kernel runs ~3 %
  *slower* across `n = 12 - 20` (likely codegen artifacts around
  the pointer-reinterpretation at the accumulator boundary). The
  technique remains the right default for Montgomery fields (BN254,
  BLS12-381) where the relaxed range is ~`p`-wide.

## GF128: codegen bug masking a 5x win

### Symptom

Initial measurements showed fused parallel GF128 5 - 20 % **slower**
than unfused, which contradicted both the theoretical memory-traffic
argument and the jolt-cpp GPU experience.

### Diagnosis (assembly read)

Disassembling the fused hot loop (function
`sumcheck_deg2_delayed_gf128_fused`, 521 lines AArch64 before fix):

| metric | fused (before) | unfused reduce | unfused bind |
|---|---|---|---|
| hot loop lines | 521 | 57 | 230 |
| `pmull` (ops) | 36 | 36 | 24 |
| `bl` (function calls) | **60** | 0 | 0 |
| stack spill stores | 7 | 0 | 1 |
| stack spill loads | 9 | 1 | 2 |
| stack frame | 624 B | 96 B | 144 B |

Every `bl` was a jump to a trivial `binius_field::M128` trait
implementation: `From<u128>`, `BitXorAssign`, `Default::default`,
`ops::Sub`. Each impl is individually tiny and tagged
`#[inline]`, but the fused loop's overall code size blew past LLVM's
inlining budget. The compiler fell back to out-of-line calls and
had to spill 16 q-registers per iteration to preserve state across
those calls. The unfused kernels used the same trait impls, but
their inner loops were small enough that the inlining budget was
never exhausted.

This is *not* a binius bug. It is an LLVM heuristic that happens to
fire on one particular large hot loop.

### Fix

Introduce a local helper that performs the same `a + r * (b - a)`
operation entirely with NEON intrinsics + the existing
`gf2_128_reduce_u128` reducer, bypassing `binius_field::M128`'s
trait impls. The helper is small and tagged `#[inline(always)]`, so
LLVM always inlines it, and the fused loop becomes a flat sequence
of `pmull` / `eor` / `ldr` / `str` with no branches into external
symbols. See
[`src/sumcheck/gf128.rs`](../../src/sumcheck/gf128.rs) - search for
`gf128_bind`.

After fix:

| metric | fused (after) | change |
|---|---|---|
| hot loop lines | 206 | -60 % |
| `pmull` | 36 | same |
| `bl` | 0 | -100 % |
| spill stores | 0 | -100 % |
| spill loads | 0 | -100 % |

### GF128 wall-clock results

Sequential (`cargo run --release --example bind_only` with best-of-30
and 3-iter warmup on M4 Max P-cluster):

| n | unfused (us) | fused (us) | ratio |
|---|---|---|---|
| 14 | 479 | 68 | **7.04x** |
| 16 | 1914 | 272 | **7.04x** |
| 18 | 7073 | 1000 | **7.07x** |
| 20 | 30483 | 4359 | **6.99x** |

Parallel pinned-pool (`examples/pinned_ab.rs`, p50 over 100 - 2000 iters):

| n | NEW_UNFUSED | NEW_FUSED | LEGACY | fused / unfused |
|---|---|---|---|---|
| 14 | 78.3 us | 21.6 us | 78.3 us | **0.28x** (3.6x faster) |
| 18 | 994.8 us | 176.5 us | 991.0 us | **0.18x** (5.6x faster) |
| 20 | 3882.1 us | 693.3 us | 3823.3 us | **0.18x** (5.6x faster) |

Why the sequential-parallel divergence narrows: at 4 workers each
thread processes a smaller chunk that fits in L1, so the unfused
path's "extra" re-reads are L1 hits. In the sequential case the
re-reads spill out of L1 and pay main-memory latency. GPU kernels in
jolt-cpp see the bigger version of this same effect.

## Fp128: compute-bound, fusion does not help

### Codegen is already clean

Same assembly read on `sumcheck_deg2_delayed_fp128_fused`:

| metric | fused | unfused reduce |
|---|---|---|
| hot loop lines | 389 | 135 |
| `mul` | 40 | 12 |
| `umulh` | 36 | 12 |
| `bl` | 0 | 0 |
| spill stores | 13 | 8 |
| spill loads | 16 | 9 |

No inlining pathology. `hachi::Fp128::mul_add_raw_aarch64` is a 35-
instruction inline-asm block that LLVM treats as opaque; it inlines
fine. Every `mul_add` inside the fused loop is materialized in place.

### Why fusion does not help

Three reinforcing reasons:

1. **Compute dominates over memory.** Each Fp128 mul_add is ~30
   cycles of dependent arithmetic (4 widening muls + carry chain +
   Solinas reduce + canonicalize via `ccmp`). A pair of round-k
   elements is ~17 bytes; even a single mul_add is enough to hide
   an L2 or main-memory read. The extra re-reads the unfused path
   performs do not show up on the profile.

2. **Fused loop has a longer per-iteration critical path.** Fused
   must complete `bind(f00, f01, r_prev)` before `h0.fmadd(f0, g0)`
   can start; the dependency chain is 2 mul_adds long per iteration
   (~60 cycles). The unfused reduce is 1 mul_add per iteration with
   three *independent* accumulators (`h0`, `h1`, `h_inf`), so LLVM
   and the M4 back-end can software-pipeline three parallel chains.

3. **`mul_add` vs `mul_wide` scheduling.** The unfused reduce's
   `Fp128Accum::fmadd` internally uses `mul_wide` (pure Rust: 4
   widening `mul64_wide` calls + carry adds), which the compiler
   freely interleaves across iterations. The fused path's bind uses
   `mul_add_raw_aarch64` which is **atomic to the scheduler**; LLVM
   cannot reorder instructions inside it or across adjacent asm
   blocks, so four sequential bind mul_adds per iteration serialize
   more than the compiler would like.

### Fp128 wall-clock results (after keeping mul_add as-is)

Sequential:

| n | unfused (us) | fused (us) | ratio |
|---|---|---|---|
| 14 | 171 | 186 | 1.08x |
| 16 | 671 | 734 | 1.09x |
| 18 | 2681 | 2915 | 1.09x |
| 20 | 11001 | 11896 | 1.08x |

Parallel (p50):

| n | NEW_UNFUSED | NEW_FUSED | LEGACY | fused / unfused |
|---|---|---|---|---|
| 14 | 47.3 us | 47.4 us | 43.3 us | 1.00x |
| 18 | 438.5 us | 453.9 us | 444.1 us | 1.03x |
| 20 | 1703.5 us | 1808.0 us | 1705.6 us | 1.06x |

### Tried and rejected

Two experiments aimed at the scheduling-opacity angle:

1. **`(diff) * r + base`** (split into two asm blocks the compiler
   can reorder): ~9 % *slower*. `mul_add` saves 5 carry-chain
   instructions over `mul + add`, and `mul_raw_aarch64`'s own
   canonicalize step re-does work that `mul_add_raw_aarch64` merges
   with the addend.

2. **`mul_wide + solinas_reduce`** (the pure-Rust path
   [`Fp128Accum::fmadd`] uses for the reduce side, hoisted into the
   bind site). Gives LLVM complete scheduling freedom over the four
   widening muls and the Solinas fold. Result: **5 - 15 % slower**
   in both seq and par.

   | n | fused (mul_add) | fused (mul_wide) | unfused |
   |---|---|---|---|
   | 14 (seq) | 186 us | 179 us (!) | 159 us |
   | 18 (seq) | 2915 us | 3134 us | 2721 us |
   | 18 (par p50) | 454 us | 479 us | 438 us |
   | 20 (par p50) | 1808 us | 1890 us | 1673 us |

   At n=14 seq the pure-Rust path wins marginally (cache-resident,
   benefits most from ILP), but at n >= 16 the 17-instruction
   overhead per bind (4 per iteration = 68 extra instructions) beats
   the scheduler freedom.

   Kept as a commented-out alternative in
   [`src/sumcheck/fp128.rs::fp128_bind`](../../src/sumcheck/fp128.rs);
   the helper shape is also forward-compatible with a future SIMD
   Fp128 lane that would have a different optimum.

3. **Instruction reordering inside the fused body** (load f then
   bind f then store f; same for g): neutral to slightly negative
   across `n = 16, 20`. Fp128 has zero stack spills and the four
   sibling `fp128_bind` operations are independent, so both orderings
   expose the same instruction-level parallelism to the M4's OOO
   scheduler. Reverted.

4. **Delayed reduction (relaxed mul_add + relaxed sub)**: the
   canonical shape of fused bind+reduce for Montgomery fields. We
   implemented a non-canonicalizing `mul_add_raw_aarch64_relaxed`
   (saves 3 inst: drops the `ccmp`-based ≥p selector, keeps a direct
   `csel`+`adds`+`adc` to fold the fold-2 overflow) and a matching
   `sub_raw_aarch64_relaxed` (+2 inst: iterates the `-c if borrow`
   correction a second time because hachi's standard `Sub` silently
   mis-reduces when an input sits in the non-canonical sliver
   `[p, 2^128)`). Intermediate f/g values stay in `[0, 2^128)`; the
   last round canonicalizes before the final bind.

   See [`src/sumcheck/fp128.rs`](../../src/sumcheck/fp128.rs):
   `sub_raw_aarch64_relaxed`, `mul_add_raw_aarch64_relaxed`,
   `fp128_bind_relaxed`, `sumcheck_deg2_delayed_fp128_fused_relaxed`.
   Correctness pinned to the canonical kernel byte-for-byte in
   `tests/fused_delayed_seq.rs::fp128_relaxed_*`.

   Algebra (per fused iteration, shared loads ignored):

   | Op | Count | Δ inst per op | Net Δ |
   |---|---|---|---|
   | `mul_add` | 4 | `-3` | `-12` |
   | `sub` (bind) | 4 | `+2` | `+8` |
   | `sub` (h_inf) | 2 | `+2` | `+4` |
   | **total** | | | **0** |

   Empirical A/B on M4 Max, sequential kernel
   ([`examples/relaxed_ab.rs`](../../examples/relaxed_ab.rs),
   `n_iter = 500`, interleaved round-robin, p10):

   | n | canonical (us) | relaxed (us) | Δ |
   |---|---|---|---|
   | 12 | 47.62 | 49.21 | **+3.3 %** |
   | 14 | 189.33 | 194.92 | +3.0 % |
   | 16 | 753.92 | 779.04 | +3.3 % |
   | 18 | 3065.00 | 3168.58 | +3.4 % |
   | 20 | 12365.29 | 12689.17 | +2.6 % |

   Consistent ~3 % regression across sizes. The algebra predicts
   zero; the empirical ~3 % likely comes from codegen artifacts: the
   relaxed kernel reinterprets the `Vec<Fp128>` backing buffer as
   `*mut [u64; 2]` via raw pointers, and the `Fp128Accum::fmadd`
   boundary goes through a `transmute` from `[u64; 2]` to `Fp128`.
   Neither should cost instructions in theory, but LLVM does emit
   a slightly different register allocation / memory op ordering at
   those boundaries vs. the canonical path which stays fully in
   `Fp128` ops.

   **Why delayed reduction is structurally neutral for Fp128 here**
   (a correction to the initial intuition that it should buy 10 -
   20 % like on Montgomery fields):

   - For a Solinas prime `p = 2^128 - c` with `c < 2^32`, the gap
     between `p` and the storage ceiling `2^128` is only `c`
     (≈ `2^8` for the `c = 275` prime we use). The "relaxed"
     storage range is `[p, 2^128)`, a sliver of size `c`. This is
     unlike Montgomery where `p` is commonly ~half the storage
     ceiling, giving a real `[0, 2p)` relaxed range with factor-of-2
     slack.
   - Every op that subtracts operands reduced mod `p` must handle
     inputs anywhere in `[0, 2^128)`. Hachi's `Sub` is correct on
     `[0, p)` but off by `c` (or `2c`) when an operand is in the
     non-canonical sliver AND the subtrahend is larger. Fix costs
     2 extra instructions per `sub`.
   - Per-bind savings on `mul_add` (3 inst) are exactly cancelled
     by added `sub` cost (2 inst * 3 subs per pair-bind, since the
     `h_inf` kernel needs `f1 - f0` and `g1 - g0` too). See the
     inst-count table above.

   **Where delayed reduction *does* pay off** (for future per-field
   specializations; see § "Cross-field generalization" below):

   - Montgomery fields (BN254, BLS12-381, curve25519 base, secp256k1
     base): storage is 4+ limbs, `p` is ~half the ceiling, relaxed
     `[0, 2p)` form is the standard output of CIOS / SOS Montgomery
     mul. Adds become a single add with at-most-one conditional `-p`
     fold-back, subs the mirror. Skipping the final `>= p` selector
     across N rounds of sumcheck saves a meaningful fraction of
     every round's compute, typically 10 - 20 %. Arkworks, fiat-
     crypto, and plonky3's Montgomery fields all ship values in
     relaxed form; our fused kernels plugging into those field crates
     inherit the optimization for free.
   - Small primes with 1-2 limb storage and cheap reductions
     (BabyBear, KoalaBear, Goldilocks, Mersenne31): the reduction is
     already only 3 - 6 instructions; saving the final canonicalize
     gives ~1 - 2 instruction improvement per mul, well below noise.
     The real win for these fields is *SIMD batching* of 4 - 16 lanes
     per mul, which is orthogonal to delayed reduction.
   - Binary fields (GF128, GF2^n): no modular reduction in the
     `mul mod p` sense; the "reduction" is a handful of XORs against
     a fixed polynomial. Nothing to delay.

## Cross-field generalization

| Field | Storage | Reduction style | Non-canonical headroom | Relaxed mul gain | Fusion bandwidth gain |
|---|---|---|---|---|---|
| GF128 (binius) | 1 x u128 | XOR-only reduce by fixed poly | N/A (no modulus) | 0 | 5 - 7x (this repo) |
| Fp128 (hachi, Solinas `2^128 - c`) | 2 x u64 | 2-fold Solinas + ccmp canonicalize | ~c (≈ 2^8) | **~0** (measured -3 %) | ~0 (compute-bound) |
| Goldilocks (`2^64 - 2^32 + 1`) | 1 x u64 | 2-fold + conditional correction | small | ~2 % | moderate |
| BabyBear (`2^31 - 2^27 + 1`) | 1 x u32 (or SIMD-16) | Monty 32 or direct | small | ~2 % | **large via SIMD** |
| BN254 scalar (Montgomery) | 4 x u64 | CIOS + `>= p` selector | ~p (full factor of 2) | **10 - 20 %** | moderate |
| BLS12-381 scalar (Montgomery) | 4 x u64 | CIOS + `>= p` selector | ~p | **10 - 15 %** | moderate |
| curve25519 base (`2^255 - 19`) | 5 x u52 (radix 2^51) | lazy-carry Solinas | factor of 4 | 10 - 15 % | moderate |

Practical implication for the platform plan: the fused `bind +
reduce` kernel shape is field-independent; what varies is whether
intermediate storage lives in canonical or relaxed form. The
cleanest factoring is a per-field trait with `mul_add`, optional
`mul_add_relaxed`, and `canonicalize` (identity when storage is
already canonical). GF128 and Fp128 Solinas opt out of the
relaxed path; Montgomery fields and curve25519-style Solinas-with-
headroom opt in.

## Kernel-shape generalization (deg-2 * eq and beyond)

The deg-2 kernel we benchmarked here is:

```text
// per pair (f0, f1, g0, g1):
h0    += f0 * g0
h1    += f1 * g1
h_inf += (f1 - f0) * (g1 - g0)
```

Three `fmadd`s and two `sub`s per pair, on top of the 4 `bind`s
(inherited from the previous round). The extensions of interest:

- **deg-2 * eq(z) sumcheck**: each term weighted by a per-pair `eq`
  factor `ew`. Per pair: `q1 += f1*g1 * ew`, `q_inf += (f1-f0)(g1-g0) * ew`.
  Two extra `mul`s per pair; the rest is identical. The extra muls
  have the same shape as the `bind` muls (one operand is a per-pair
  value, the other is the shared `ew`), so every optimization above
  -- fusion, per-field relaxed mul, SIMD batching -- applies
  symmetrically. The additional compute shifts the bound slightly
  toward memory for Fp128 (extra 2 muls per pair / ~70 cycles) but
  not enough to flip the fused vs unfused choice; GF128 still wins
  big on fusion, Fp128 still breaks even.
- **Higher-degree sumchecks** (deg-3, deg-4 for Jolt-style
  instruction lookups): more evaluation points per pair, each is an
  extra `fmadd`. The per-pair memory footprint is unchanged (still
  2 reads + 1 write per dim), but per-pair compute grows linearly
  in degree. For Fp128 this makes the compute/memory balance
  *more* compute-bound, so fusion's bandwidth argument gets weaker,
  not stronger. For GF128, SIMD-like throughput on `pmull` means
  fusion continues to win.
- **Zerocheck / boolean / logup-GKR**: same bind+reduce skeleton,
  different per-pair reduce expression. All of these reduce to
  "N muls + M adds per pair"; the rules of thumb carry over.

The bind step (applied to round `r-1`'s buffer to produce round
`r`'s) is *shape-independent*: it always reads 4, writes 2 per
output pair and does one `sub` + one `mul_add` per write. So the
bind-level optimizations (per-field relaxed mul_add, inlining
hygiene for GF128, SIMD batching for small primes) are shared
across every sumcheck variant in the platform.

## Directional take-aways

- Fusion on CPU is a **memory-traffic** optimization, not a
  compute-count optimization. It wins exactly when the bound is
  cache bandwidth (GF128, small prime fields, large Mersenne fields
  without deep asm), and is neutral-to-bad when the bound is a hand-
  tuned multiply (Fp128, BN254, 256-bit fields).
- GPU fusion wins almost unconditionally because the memory hierarchy
  is much flatter (no L1/L2 reuse window) and the multiplier is
  cheaper relative to a DRAM read. That is why jolt-cpp's
  `bind_eval_roundN_kernel` is a clear win on CUDA even for large-
  prime fields.
- Default selection for this repo: GF128 defaults to fused, Fp128
  defaults to unfused. Both are exercised by the
  `use_fused_path: bool` toggle and covered by
  `tests/parallel_delayed.rs` + `tests/pinned_loop.rs`.
- Future work, tracked in
  [`../plans/sumcheck-cpu-platform.md`](../plans/sumcheck-cpu-platform.md):
  - Phase 2: the legacy single-broadcast implementation matches
    fused GF128 for `n >= 14`. Consolidate `legacy.rs` down to the
    pinned+fused kernels and retire the rayon variants.
  - Phase 3+: Metal port of GF128 fused kernel. The CPU-side kernel
    already matches the jolt-cpp shape, so the port is mostly a
    storage-class and index-unrolling rewrite.
