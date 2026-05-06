/// Robust min/max aggregation mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregation {
    /// Exact min/max.
    Exact,
    /// Exact min/max with gradients distributed across tied extrema.
    DistributedExact,
    /// Smooth min/max using log-sum-exp with the given positive scale.
    Smooth { scale: f64 },
}

impl Aggregation {
    pub const fn smooth(scale: f64) -> Self {
        Self::Smooth { scale }
    }
}

/// Evaluation options shared by all formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalOptions {
    pub predicate_scale: f64,
    pub aggregation: Aggregation,
}

impl EvalOptions {
    pub const fn exact() -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::Exact,
        }
    }

    pub const fn distributed_exact() -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::DistributedExact,
        }
    }

    pub const fn smooth(scale: f64) -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::Smooth { scale },
        }
    }

    pub const fn with_predicate_scale(mut self, predicate_scale: f64) -> Self {
        self.predicate_scale = predicate_scale;
        self
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self::exact()
    }
}
