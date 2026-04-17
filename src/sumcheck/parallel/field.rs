//! Field-agnostic [`SumcheckRound`] trait.
//!
//! Implementors describe one running sumcheck instance over their own
//! field, owning the multilinears, ping-pong buffers, and any auxiliary
//! state. The [`super::scheduler::par_sumcheck`] driver then runs
//! `scope_rounds` of parallel reduce-then-bind on the trait, using the
//! [`super::pool::PinnedPool`].
//!
//! ### Why these methods
//!
//! - [`reduce_chunk`](SumcheckRound::reduce_chunk): each worker
//!   independently reduces its chunk of the current read buffer to a
//!   small `Partial` (e.g. a deg-2 projective triple). Read-only over
//!   `&self`.
//! - [`combine`](SumcheckRound::combine): associative+commutative
//!   merge, currently used by the scheduler's linear reduce on
//!   worker 0. The trait constraint is associative+commutative so
//!   we can swap in a tournament reduce later without touching any
//!   `impl` block (D1 in the productionization plan — measured as a
//!   regression for the current 48-byte deg-2 `Partial` on M4 Max,
//!   kept as an option for future large-`Partial` kernels).
//! - [`bind_chunk`](SumcheckRound::bind_chunk): each worker writes its
//!   chunk of the next-round buffer in-place. The "in-place" is via
//!   ping-pong: we read from one buffer and write to the other, so
//!   reading worker A's range cannot collide with writing worker B's
//!   range. The unsafety is in the disjoint-windows guarantee, which
//!   the scheduler enforces by partitioning `[0, current_pairs)`.
//! - [`observe_partial`](SumcheckRound::observe_partial): the per-round
//!   transcript hook. Default `black_box`; production impls would feed
//!   the partial into Fiat-Shamir before extracting the next challenge.
//!
//! ### Lifetime model
//!
//! All four methods take `&self` so workers can hold an `&R` across the
//! whole broadcast. Internal mutation (buffer swapping, live-length
//! tracking) is the impl's responsibility, typically via [`UnsafeCell`]
//! (the round index passed in by the scheduler tells the impl which
//! buffer is read vs. write).
//!
//! [`UnsafeCell`]: std::cell::UnsafeCell

/// One running sumcheck instance, parameterised by its field and
/// per-round partial type.
///
/// Implementors describe a *parallel-friendly* slice of one sumcheck:
/// reduce a chunk, combine partials, bind a chunk. The scheduler in
/// [`super::scheduler::par_sumcheck`] is field-agnostic and only sees
/// these four operations.
///
/// The trait is intentionally minimal. State that doesn't affect the
/// scheduler (multilinears, scratch buffers, transcript handles) lives
/// inside the implementor as `UnsafeCell` or similar interior
/// mutability, so the `&self` worker borrow is enough.
pub trait SumcheckRound: Sync {
    /// Field element used for challenges.
    type Elem: Copy + Send + Sync;

    /// Per-chunk partial sum produced by [`reduce_chunk`]. For a deg-2
    /// projective-1∞ kernel this is a triple `(h0, h1, h_inf)`; for a
    /// deg-3 kernel it is one element larger. Must be cheap to copy.
    ///
    /// [`reduce_chunk`]: SumcheckRound::reduce_chunk
    type Partial: Copy + Send + Sync + Default;

    /// Minimum live pairs per worker before parallelism is a net loss
    /// for this kernel. The scheduler uses this to pick the number of
    /// rounds it keeps in the parallel scope (`scope_rounds`): rounds
    /// where the per-worker pair count would drop below this run on
    /// the sequential tail instead.
    ///
    /// Tuned empirically per (field, hardware). 8 is a reasonable
    /// default for 128-bit fields on M-class silicon.
    const MIN_PAIRS_PER_WORKER: usize = 8;

    /// First-round target pairs-per-worker. Used by the scheduler to
    /// pick the number of active workers from the initial problem size.
    /// Larger ⇒ fewer workers but better amortisation of the per-round
    /// barrier; smaller ⇒ more workers, more barrier contention.
    const TARGET_PAIRS_PER_WORKER: usize = 256;

    /// Compute the partial sum over pair indices `[lo, lo + len)` in
    /// the round-`round` *read* buffer. Read-only over `&self`; workers
    /// call this concurrently with disjoint windows.
    fn reduce_chunk(&self, round: usize, lo: usize, len: usize) -> Self::Partial;

    /// Combine two partials. Must be associative and commutative.
    fn combine(a: Self::Partial, b: Self::Partial) -> Self::Partial;

    /// Per-round transcript hook on the main thread, called once after
    /// all workers' partials have been combined and before workers
    /// start binding. The default `black_box` models the Fiat-Shamir
    /// dependency the verifier transcript would impose; a production
    /// caller would feed `partial` into the transcript here and derive
    /// `r` from the result.
    #[inline(always)]
    fn observe_partial(_round: usize, partial: Self::Partial, _r: &Self::Elem) {
        std::hint::black_box(partial);
    }

    /// Bind one chunk of pairs `[lo, lo + len)` from the round-`round`
    /// read buffer into the round-`round` write buffer using challenge
    /// `r`. Workers call this concurrently with disjoint windows.
    ///
    /// # Safety
    ///
    /// The caller (scheduler) must ensure the `[lo, lo + len)` ranges
    /// across concurrent invocations within one round are disjoint
    /// (which the chunk partition guarantees) and that the read and
    /// write buffers do not alias for any worker's range (which the
    /// ping-pong layout guarantees).
    unsafe fn bind_chunk(&self, round: usize, lo: usize, len: usize, r: &Self::Elem);
}
