# Source-level code-shaping notes

These notes preserve implementation observations that helped tune the benchmark
kernels. They are not part of the benchmark methodology and are not independent
evidence for a paper claim.

On Apple M4 Max, semantically equivalent Rust source sometimes produced
different register allocation and stack traffic. Three patterns were useful:

- Naming intermediate sums shortened live ranges in several projective kernels.
- Asymmetric inlining kept the packed BabyBear Ext5 loop compact.
- Reloading packed equality weights was sometimes cheaper than retaining them
  across the full accumulation.

In one packed BabyBear Ext5 kernel, reordering the loop body reduced the compiled
function from 5,996 to 5,436 bytes, the stack frame from `0x530` to `0x410`
bytes, and the counted stack references from 385 to 248 under the compiler used
at that time. These figures are compiler-specific diagnostics, not portable
performance results. Recheck them whenever the pinned toolchain changes.

The algebraic operation counts do not depend on these choices. The accepted
benchmark data must come from the independent-process procedure in
[`benchmark-methodology.md`](benchmark-methodology.md).
