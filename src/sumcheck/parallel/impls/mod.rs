//! Reference [`sumcheck_parallel::SumcheckRound`] implementations and
//! their public wrapper functions.
//!
//! Each submodule pairs a struct (e.g. [`gf128::GF128DelayedRound`])
//! that implements [`sumcheck_parallel::SumcheckRound`] with a public
//! function (e.g. [`gf128::sumcheck_deg2_delayed_gf128_par4_pinned`])
//! that orchestrates one whole sumcheck call: pick `n_workers`,
//! pick `scope_rounds`, run the parallel scheduler, then hand off
//! to the sequential kernel for the tail.

pub(crate) mod fp128;
pub(crate) mod gf128;

pub use fp128::sumcheck_deg2_delayed_fp128_par4_pinned;
pub use gf128::sumcheck_deg2_delayed_gf128_par4_pinned;
