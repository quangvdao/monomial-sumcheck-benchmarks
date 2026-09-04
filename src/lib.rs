pub mod lookup_tables;
pub mod sumcheck;

use std::time::Duration;

use criterion::Criterion;

/// Shared Criterion configuration for every paper benchmark.
///
/// Process-level repetition and paired confidence intervals are handled by
/// the collection and summarization scripts.
pub fn benchmark_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(10))
        .confidence_level(0.99)
        .sample_size(100)
}
