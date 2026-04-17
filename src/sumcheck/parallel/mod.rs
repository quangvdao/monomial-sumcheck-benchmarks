//! Parallel sumcheck infrastructure.
//!
//! This module is the in-repo prototype for the future `pinned-pool`
//! and `sumcheck-parallel` crates (see
//! `docs/plans/sumcheck-parallel-productionization.md`). It is split
//! into:
//!
//! - [`pool`]: process-global persistent worker pool with QoS pinning
//!   and a Release/Acquire epoch broadcast.
//! - [`field`]: the [`SumcheckRound`](field::SumcheckRound) trait that
//!   abstracts a single sumcheck instance over its field, partials,
//!   and bind step.
//! - [`scheduler`]: [`par_sumcheck`](scheduler::par_sumcheck), the
//!   field-agnostic driver that calls
//!   [`PinnedPool::broadcast_scoped`](pool::PinnedPool::broadcast_scoped)
//!   once and contains the D1 (tournament reduce) and D2 (per-round
//!   `n_active` shrinkage) optimizations.
//! - [`impls`]: reference [`SumcheckRound`](field::SumcheckRound)
//!   implementations for the GF128 and Fp128 delayed kernels in this
//!   repo, plus the public wrapper functions
//!   (`sumcheck_deg2_delayed_{gf128,fp128}_par4_pinned`) used by the
//!   benches and tests.
//! - [`legacy`]: Approaches 1-3 (`*_par1_*`, `*_par2_chili`,
//!   `*_par3_persistent`) preserved as benchmark baselines for the
//!   `PARALLELISM.md` comparison table. Frozen; do not extend.

pub mod field;
pub mod impls;
pub mod legacy;
pub mod pool;
pub mod scheduler;

pub use field::SumcheckRound;
pub use impls::{sumcheck_deg2_delayed_fp128_par4_pinned, sumcheck_deg2_delayed_gf128_par4_pinned};
pub use legacy::*;
pub use pool::PinnedPool;
pub use scheduler::par_sumcheck;
