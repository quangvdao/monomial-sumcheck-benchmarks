# Source-level code-shaping notes

These notes preserve implementation observations that helped tune the benchmark
kernels. They are not part of the benchmark methodology and are not independent
evidence for a paper claim.

On Apple M4 Max, semantically equivalent Rust source sometimes produced
different register allocation and stack traffic. Earlier revisions used three
such patterns in the projective kernels only: accumulating the two terms in a
different order from the Boolean kernel, routing one accumulator of the packed
BabyBear Ext5 loop through a non-inlined helper, and packing the equality
weights a second time instead of retaining them.

That asymmetric shaping has been removed. It gave the projective kernel a code
shape that its Boolean baseline did not have, so the comparison measured the
shaping together with the basis. Each projective kernel in `bb5_packed.rs` and
`bb_ext.rs` is now a line-for-line mirror of its Boolean counterpart: the same
loads, the same single packing of the equality weights, the same inlined
accumulator helper, and the same accumulation order, differing only in sums
versus differences. Both variants therefore share one code shape. Keep them
mirrored when changing either.

The sizes of compiled functions and stack frames remain compiler-specific
diagnostics, not portable performance results. Recheck them whenever the pinned
toolchain changes.

The algebraic operation counts do not depend on these choices. The accepted
benchmark data must come from the independent-process procedure in
[`benchmark-methodology.md`](benchmark-methodology.md).
