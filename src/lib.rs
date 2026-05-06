//! Core Signal Temporal Logic robustness evaluation on Burn tensors.
//!
//! This crate intentionally uses a Rust-native formula AST rather than mirroring
//! upstream Python classes. Traces are rank-3 Burn tensors with shape
//! `[batch, time, dim]`, and time is expected to be reversed as in the original
//! STLCG implementation.

mod env;
mod error;
mod expr;
mod formula;
mod interval;
mod ops;
mod options;

pub use env::SignalEnv;
pub use error::{Result, StlcgError};
pub use expr::{Expr, scalar, var};
pub use formula::{Formula, IntegralOptions, IntegrationScheme, PaddingMode, PredicateOp};
pub use interval::Interval;
pub use options::{Aggregation, EvalOptions};

use burn::tensor::{Bool, Tensor, backend::Backend};

/// Rank-3 signal trace: `[batch, time, dim]`.
pub type Trace<B> = Tensor<B, 3>;

/// Rank-3 Boolean trace: `[batch, time, dim]`.
pub type BoolTrace<B> = Tensor<B, 3, Bool>;

/// Reverse the time axis of a trace.
pub fn reverse_time<B: Backend>(trace: Trace<B>) -> Trace<B> {
    trace.flip([1])
}
