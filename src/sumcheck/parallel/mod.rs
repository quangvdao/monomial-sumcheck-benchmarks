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
//! plus the pre-production baselines (`rayon_*`, `chili`,
//! `persistent`, `pinned_v0`) used by the `PARALLELISM.md`
//! comparison table.
//!
//! - [`impls`]: reference
//!   [`SumcheckRound`](sumcheck_parallel::SumcheckRound)
//!   implementations for the GF128 and Fp128 delayed kernels in this
//!   repo, plus the public wrapper functions
//!   (`sumcheck_deg2_delayed_{gf128,fp128}_pinned`) used by the
//!   benches and tests.
//! - [`legacy`]: `rayon_scope`, `rayon_iter`, `chili`, `persistent`,
//!   `pinned_v0` variants preserved as benchmark baselines for the
//!   `PARALLELISM.md` comparison table. Frozen; do not extend.

pub mod impls;
pub mod legacy;

pub use impls::{sumcheck_deg2_delayed_fp128_pinned, sumcheck_deg2_delayed_gf128_pinned};
pub use legacy::*;
pub use pinned_pool::PinnedPool;
pub use sumcheck_parallel::{par_sumcheck, Schedule, SumcheckRound};
