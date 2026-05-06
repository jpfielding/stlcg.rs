use crate::{Result, StlcgError};

/// Temporal interval over integer trace indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interval {
    /// `[0, infinity)`.
    #[default]
    Unbounded,
    /// `[start, end]`, inclusive.
    Closed {
        /// Inclusive lower offset bound.
        start: usize,
        /// Inclusive upper offset bound.
        end: usize,
    },
    /// `[start, infinity)`.
    From {
        /// Inclusive lower offset bound.
        start: usize,
    },
}

impl Interval {
    /// Create an unbounded interval `[0, infinity)`.
    pub const fn unbounded() -> Self {
        Self::Unbounded
    }

    /// Create a closed interval `[start, end]`.
    pub const fn closed(start: usize, end: usize) -> Self {
        Self::Closed { start, end }
    }

    /// Create a lower-bounded interval `[start, infinity)`.
    pub const fn from(start: usize) -> Self {
        Self::From { start }
    }

    pub(crate) fn validate(self) -> Result<()> {
        match self {
            Self::Closed { start, end } if start > end => Err(StlcgError::InvalidInterval(
                "closed interval start must be <= end".to_string(),
            )),
            _ => Ok(()),
        }
    }
}
