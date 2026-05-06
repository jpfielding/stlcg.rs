use thiserror::Error;

/// Result type used by this crate.
pub type Result<T> = core::result::Result<T, StlcgError>;

/// Errors returned while building or evaluating STL formulas.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StlcgError {
    /// A named signal was referenced by a formula but was not present in the environment.
    #[error("signal `{0}` was not found")]
    MissingSignal(String),

    /// Evaluation requires at least one signal so scalar expressions have a tensor template.
    #[error("signal environment is empty")]
    EmptySignalEnv,

    /// Signal traces must contain at least one time step.
    #[error("time dimension must be non-empty")]
    EmptyTimeDimension,

    /// A temporal interval is malformed.
    #[error("invalid interval: {0}")]
    InvalidInterval(String),

    /// Evaluation options are malformed.
    #[error("invalid evaluation option: {0}")]
    InvalidOption(String),

    /// Signal traces in the same environment do not share the same shape.
    #[error("shape mismatch for `{name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Name of the signal with the mismatched shape.
        name: String,
        /// Expected trace shape `[batch, time, dim]`.
        expected: [usize; 3],
        /// Actual trace shape `[batch, time, dim]`.
        actual: [usize; 3],
    },

    /// A requested time index is outside the evaluated trace.
    #[error("robustness requested at time {time}, but trace length is {len}")]
    TimeOutOfBounds {
        /// Requested non-reversed time index.
        time: usize,
        /// Number of available time steps.
        len: usize,
    },

    /// A variadic formula such as conjunction or disjunction has no children.
    #[error("formula requires at least one child")]
    EmptyFormulaSet,
}
