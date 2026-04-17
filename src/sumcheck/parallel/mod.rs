//! Parallel sumcheck infrastructure.
//!
//! The scheduler + trait have been extracted to the external
//! [`sumcheck-parallel`](sumcheck_parallel) crate at
//! `~/Documents/SNARKs/sumcheck-parallel/`. The pool lives at
//! `~/Documents/SNARKs/pinned-pool/` and is re-exported as
//! [`PinnedPool`]. See
//! [`docs/plans/sumcheck-parallel-productionization.md`](../../../docs/plans/sumcheck-parallel-productionization.md)
//! for the extraction plan.
//!
//! This module is now thin: it only hosts the field-specific
//! [`SumcheckRound`](sumcheck_parallel::SumcheckRound)
//! implementations for this repo's GF128 and Fp128 delayed kernels,
//! plus the Approach 1-3 legacy baselines used by the
//! `PARALLELISM.md` comparison table.
//!
//! - [`impls`]: reference
//!   [`SumcheckRound`](sumcheck_parallel::SumcheckRound)
//!   implementations for the GF128 and Fp128 delayed kernels in this
//!   repo, plus the public wrapper functions
//!   (`sumcheck_deg2_delayed_{gf128,fp128}_par4_pinned`) used by the
//!   benches and tests.
//! - [`legacy`]: Approaches 1-3 (`*_par1_*`, `*_par2_chili`,
//!   `*_par3_persistent`) preserved as benchmark baselines for the
//!   `PARALLELISM.md` comparison table. Frozen; do not extend.

pub mod impls;
pub mod legacy;

pub use impls::{sumcheck_deg2_delayed_fp128_par4_pinned, sumcheck_deg2_delayed_gf128_par4_pinned};
pub use legacy::*;
pub use pinned_pool::PinnedPool;
pub use sumcheck_parallel::{par_sumcheck, SumcheckRound};
