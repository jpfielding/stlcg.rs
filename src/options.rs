use crate::{Result, StlcgError};

/// Robust min/max aggregation mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregation {
    /// Exact min/max.
    Exact,
    /// Exact min/max with gradients distributed across tied extrema.
    DistributedExact,
    /// Smooth min/max using log-sum-exp with the given positive scale.
    Smooth {
        /// Positive log-sum-exp scale.
        scale: f64,
    },
}

impl Aggregation {
    /// Create a smooth aggregation mode with the given log-sum-exp scale.
    pub const fn smooth(scale: f64) -> Self {
        Self::Smooth { scale }
    }
}

/// Evaluation options shared by all formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalOptions {
    /// Positive multiplier applied to predicate robustness values.
    pub predicate_scale: f64,
    /// Min/max aggregation strategy used by Boolean and temporal operators.
    pub aggregation: Aggregation,
}

impl EvalOptions {
    /// Use exact min/max aggregation and a predicate scale of `1.0`.
    pub const fn exact() -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::Exact,
        }
    }

    /// Use exact min/max values with gradients distributed across tied extrema.
    pub const fn distributed_exact() -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::DistributedExact,
        }
    }

    /// Use smooth log-sum-exp aggregation with the given positive scale.
    pub const fn smooth(scale: f64) -> Self {
        Self {
            predicate_scale: 1.0,
            aggregation: Aggregation::Smooth { scale },
        }
    }

    /// Set the positive predicate robustness scale.
    pub const fn with_predicate_scale(mut self, predicate_scale: f64) -> Self {
        self.predicate_scale = predicate_scale;
        self
    }

    /// Validate that all numeric options are finite and semantically valid.
    pub fn validate(self) -> Result<()> {
        if !self.predicate_scale.is_finite() || self.predicate_scale <= 0.0 {
            return Err(StlcgError::InvalidOption(
                "predicate_scale must be finite and positive".to_string(),
            ));
        }

        match self.aggregation {
            Aggregation::Exact | Aggregation::DistributedExact => Ok(()),
            Aggregation::Smooth { scale } if scale.is_finite() && scale > 0.0 => Ok(()),
            Aggregation::Smooth { .. } => Err(StlcgError::InvalidOption(
                "smooth aggregation scale must be finite and positive".to_string(),
            )),
        }
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self::exact()
    }
}
