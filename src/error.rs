use thiserror::Error;

pub type Result<T> = core::result::Result<T, StlcgError>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StlcgError {
    #[error("signal `{0}` was not found")]
    MissingSignal(String),

    #[error("signal environment is empty")]
    EmptySignalEnv,

    #[error("time dimension must be non-empty")]
    EmptyTimeDimension,

    #[error("invalid interval: {0}")]
    InvalidInterval(String),

    #[error("shape mismatch for `{name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: [usize; 3],
        actual: [usize; 3],
    },

    #[error("robustness requested at time {time}, but trace length is {len}")]
    TimeOutOfBounds { time: usize, len: usize },

    #[error("formula requires at least one child")]
    EmptyFormulaSet,
}
