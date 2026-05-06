use std::fmt;
use std::ops::Not;

use burn::tensor::{Tensor, backend::Backend};

use crate::ops::{maxish3, minish3, reduce_last_max, reduce_last_min, relu, select_time, temporal};
use crate::{BoolTrace, EvalOptions, Expr, Interval, Result, SignalEnv, StlcgError, Trace};

const LARGE_NEGATIVE: f64 = -1.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateOp {
    LessEqual,
    GreaterEqual,
    Equal,
}

/// Integration scheme for [`Formula::Integral`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationScheme {
    Riemann,
    Trapezoid,
}

/// Padding used for finite-window integrals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    Zero,
    Same,
    Custom(f64),
}

/// Options used by the integral formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegralOptions {
    pub interval: Option<Interval>,
    pub use_relu: bool,
    pub padding: PaddingMode,
    pub scheme: IntegrationScheme,
}

impl IntegralOptions {
    pub const fn cumulative() -> Self {
        Self {
            interval: None,
            use_relu: false,
            padding: PaddingMode::Same,
            scheme: IntegrationScheme::Riemann,
        }
    }

    pub const fn window(interval: Interval) -> Self {
        Self {
            interval: Some(interval),
            use_relu: false,
            padding: PaddingMode::Same,
            scheme: IntegrationScheme::Riemann,
        }
    }
}

impl Default for IntegralOptions {
    fn default() -> Self {
        Self::cumulative()
    }
}

/// STL formula AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Formula {
    Expression(Expr),
    Predicate {
        lhs: Expr,
        op: PredicateOp,
        rhs: Expr,
    },
    Not(Box<Formula>),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Always {
        subformula: Box<Formula>,
        interval: Interval,
    },
    Eventually {
        subformula: Box<Formula>,
        interval: Interval,
    },
    Until {
        lhs: Box<Formula>,
        rhs: Box<Formula>,
        interval: Interval,
        overlap: bool,
    },
    Then {
        lhs: Box<Formula>,
        rhs: Box<Formula>,
        interval: Interval,
        overlap: bool,
    },
    Integral {
        subformula: Box<Formula>,
        options: IntegralOptions,
    },
}

impl Formula {
    pub fn expression(expr: Expr) -> Self {
        Self::Expression(expr)
    }

    pub(crate) fn less_equal(lhs: Expr, rhs: Expr) -> Self {
        Self::Predicate {
            lhs,
            op: PredicateOp::LessEqual,
            rhs,
        }
    }

    pub(crate) fn greater_equal(lhs: Expr, rhs: Expr) -> Self {
        Self::Predicate {
            lhs,
            op: PredicateOp::GreaterEqual,
            rhs,
        }
    }

    pub(crate) fn equal(lhs: Expr, rhs: Expr) -> Self {
        Self::Predicate {
            lhs,
            op: PredicateOp::Equal,
            rhs,
        }
    }

    pub fn negated(self) -> Self {
        Self::Not(Box::new(self))
    }

    pub fn and(self, rhs: Self) -> Self {
        let mut children = match self {
            Self::And(children) => children,
            other => vec![other],
        };
        match rhs {
            Self::And(rhs_children) => children.extend(rhs_children),
            other => children.push(other),
        }
        Self::And(children)
    }

    pub fn or(self, rhs: Self) -> Self {
        let mut children = match self {
            Self::Or(children) => children,
            other => vec![other],
        };
        match rhs {
            Self::Or(rhs_children) => children.extend(rhs_children),
            other => children.push(other),
        }
        Self::Or(children)
    }

    pub fn implies(self, rhs: Self) -> Self {
        Self::Implies(Box::new(self), Box::new(rhs))
    }

    pub fn always(self, interval: Interval) -> Self {
        Self::Always {
            subformula: Box::new(self),
            interval,
        }
    }

    pub fn eventually(self, interval: Interval) -> Self {
        Self::Eventually {
            subformula: Box::new(self),
            interval,
        }
    }

    pub fn until(self, rhs: Self, interval: Interval, overlap: bool) -> Self {
        Self::Until {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            interval,
            overlap,
        }
    }

    pub fn then(self, rhs: Self, interval: Interval, overlap: bool) -> Self {
        Self::Then {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            interval,
            overlap,
        }
    }

    pub fn integral(self, options: IntegralOptions) -> Self {
        Self::Integral {
            subformula: Box::new(self),
            options,
        }
    }

    pub fn robustness_trace<B: Backend>(
        &self,
        env: &SignalEnv<B>,
        options: EvalOptions,
    ) -> Result<Trace<B>> {
        env.validate_compatible_shapes()?;
        self.validate()?;
        self.robustness_trace_inner(env, options)
    }

    pub fn robustness_at<B: Backend>(
        &self,
        env: &SignalEnv<B>,
        time: usize,
        options: EvalOptions,
    ) -> Result<Trace<B>> {
        let trace = self.robustness_trace(env, options)?;
        let len = trace.dims()[1];
        if time >= len {
            return Err(StlcgError::TimeOutOfBounds { time, len });
        }
        Ok(trace.narrow(1, len - time - 1, 1))
    }

    pub fn eval_trace<B: Backend>(
        &self,
        env: &SignalEnv<B>,
        options: EvalOptions,
    ) -> Result<BoolTrace<B>> {
        Ok(self.robustness_trace(env, options)?.greater_elem(0.0))
    }

    pub fn eval_at<B: Backend>(
        &self,
        env: &SignalEnv<B>,
        time: usize,
        options: EvalOptions,
    ) -> Result<BoolTrace<B>> {
        Ok(self.robustness_at(env, time, options)?.greater_elem(0.0))
    }

    fn robustness_trace_inner<B: Backend>(
        &self,
        env: &SignalEnv<B>,
        options: EvalOptions,
    ) -> Result<Trace<B>> {
        match self {
            Self::Expression(expr) => Ok(expr.eval(env)? * options.predicate_scale),
            Self::Predicate { lhs, op, rhs } => {
                let lhs = lhs.eval(env)?;
                let rhs = rhs.eval(env)?;
                let trace = match op {
                    PredicateOp::LessEqual => rhs - lhs,
                    PredicateOp::GreaterEqual => lhs - rhs,
                    PredicateOp::Equal => (lhs - rhs).abs().neg(),
                };
                Ok(trace * options.predicate_scale)
            }
            Self::Not(subformula) => Ok(subformula.robustness_trace_inner(env, options)?.neg()),
            Self::And(children) => {
                let traces = eval_children(children, env, options)?;
                Ok(reduce_last_min(traces, options.aggregation))
            }
            Self::Or(children) => {
                let traces = eval_children(children, env, options)?;
                Ok(reduce_last_max(traces, options.aggregation))
            }
            Self::Implies(lhs, rhs) => {
                let lhs = lhs.robustness_trace_inner(env, options)?.neg();
                let rhs = rhs.robustness_trace_inner(env, options)?;
                Ok(reduce_last_max(vec![lhs, rhs], options.aggregation))
            }
            Self::Always {
                subformula,
                interval,
            } => {
                let trace = subformula.robustness_trace_inner(env, options)?;
                Ok(temporal(trace, *interval, options.aggregation, true))
            }
            Self::Eventually {
                subformula,
                interval,
            } => {
                let trace = subformula.robustness_trace_inner(env, options)?;
                Ok(temporal(trace, *interval, options.aggregation, false))
            }
            Self::Until {
                lhs,
                rhs,
                interval,
                overlap,
            } => Self::until_or_then(env, options, lhs, rhs, *interval, *overlap, false),
            Self::Then {
                lhs,
                rhs,
                interval,
                overlap,
            } => Self::until_or_then(env, options, lhs, rhs, *interval, *overlap, true),
            Self::Integral {
                subformula,
                options: integral_options,
            } => {
                let trace = subformula.robustness_trace_inner(env, options)?;
                Ok(integral(trace, *integral_options))
            }
        }
    }

    fn until_or_then<B: Backend>(
        env: &SignalEnv<B>,
        options: EvalOptions,
        lhs: &Formula,
        rhs: &Formula,
        interval: Interval,
        overlap: bool,
        use_eventually_prefix: bool,
    ) -> Result<Trace<B>> {
        let lhs_trace = lhs.robustness_trace_inner(env, options)?;
        let rhs_formula = if overlap {
            rhs.clone()
        } else {
            rhs.clone().eventually(Interval::closed(0, 1))
        };
        let rhs_trace = rhs_formula.robustness_trace_inner(env, options)?;
        let time = rhs_trace.dims()[1];

        let mut outputs = Vec::with_capacity(time);
        for i in 0..time {
            let candidate_indices = candidate_indices(interval, i);
            let mut candidates = Vec::with_capacity(candidate_indices.len());
            for candidate_index in candidate_indices {
                let rhs_value = select_time(&rhs_trace, candidate_index);
                let lhs_window =
                    lhs_trace
                        .clone()
                        .narrow(1, candidate_index, i - candidate_index + 1);
                let lhs_value = if use_eventually_prefix {
                    maxish3(lhs_window, 1, options.aggregation)
                } else {
                    minish3(lhs_window, 1, options.aggregation)
                };
                candidates.push(reduce_last_min(
                    vec![rhs_value, lhs_value],
                    options.aggregation,
                ));
            }

            if candidates.is_empty() {
                candidates.push(rhs_trace.full_like(LARGE_NEGATIVE).narrow(1, 0, 1));
            }
            outputs.push(reduce_last_max(candidates, options.aggregation));
        }

        Ok(Tensor::cat(outputs, 1))
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Expression(_) => Ok(()),
            Self::Predicate { .. } => Ok(()),
            Self::Not(subformula) => subformula.validate(),
            Self::And(children) | Self::Or(children) => {
                if children.is_empty() {
                    return Err(StlcgError::EmptyFormulaSet);
                }
                children.iter().try_for_each(Self::validate)
            }
            Self::Implies(lhs, rhs)
            | Self::Until { lhs, rhs, .. }
            | Self::Then { lhs, rhs, .. } => {
                lhs.validate()?;
                rhs.validate()
            }
            Self::Always {
                subformula,
                interval,
            }
            | Self::Eventually {
                subformula,
                interval,
            } => {
                interval.validate()?;
                subformula.validate()
            }
            Self::Integral {
                subformula,
                options,
            } => {
                if let Some(interval) = options.interval {
                    interval.validate()?;
                }
                subformula.validate()
            }
        }
    }
}

fn eval_children<B: Backend>(
    children: &[Formula],
    env: &SignalEnv<B>,
    options: EvalOptions,
) -> Result<Vec<Trace<B>>> {
    children
        .iter()
        .map(|child| child.robustness_trace_inner(env, options))
        .collect()
}

fn candidate_indices(interval: Interval, index: usize) -> Vec<usize> {
    match interval {
        Interval::Unbounded => (0..=index).collect(),
        Interval::Closed { start, end } if index < end => Vec::new(),
        Interval::Closed { start, end } => ((index - end)..=(index - start)).collect(),
        Interval::From { start } if index < start => Vec::new(),
        Interval::From { start } => (0..=(index - start)).collect(),
    }
}

fn integral<B: Backend>(trace: Trace<B>, options: IntegralOptions) -> Trace<B> {
    match options.interval {
        None => cumulative_integral(trace, options),
        Some(interval) => window_integral(trace, interval, options),
    }
}

fn cumulative_integral<B: Backend>(trace: Trace<B>, options: IntegralOptions) -> Trace<B> {
    let dims = trace.dims();
    let signal = match options.scheme {
        IntegrationScheme::Riemann => trace,
        IntegrationScheme::Trapezoid => {
            let zero = trace.zeros_like().narrow(1, 0, 1);
            if dims[1] <= 1 {
                zero
            } else {
                let left = trace.clone().narrow(1, 0, dims[1] - 1);
                let right = trace.narrow(1, 1, dims[1] - 1);
                Tensor::cat(vec![zero, (left + right) / 2.0], 1)
            }
        }
    };

    let signal = if options.use_relu {
        relu(signal)
    } else {
        signal
    };
    signal.cumsum(1)
}

fn window_integral<B: Backend>(
    trace: Trace<B>,
    interval: Interval,
    options: IntegralOptions,
) -> Trace<B> {
    let time = trace.dims()[1];
    let mut outputs = Vec::with_capacity(time);

    for i in 0..time {
        let offsets: Vec<usize> = match interval {
            Interval::Unbounded => (0..=i).collect(),
            Interval::Closed { start, end } => (start..=end).collect(),
            Interval::From { start } if start > i => vec![start],
            Interval::From { start } => (start..=i).collect(),
        };
        let mut terms = Vec::with_capacity(offsets.len());
        let count = offsets.len();

        for (term_index, offset) in offsets.into_iter().enumerate() {
            let mut value = if offset > i {
                finite_padding(&trace, options.padding)
            } else {
                select_time(&trace, i - offset)
            };
            if matches!(options.scheme, IntegrationScheme::Trapezoid)
                && count > 1
                && (term_index == 0 || term_index == count - 1)
            {
                value = value / 2.0;
            }
            terms.push(value);
        }

        if terms.is_empty() {
            terms.push(trace.zeros_like().narrow(1, 0, 1));
        }
        let window = Tensor::cat(terms, 1);
        let window = if options.use_relu {
            relu(window)
        } else {
            window
        };
        outputs.push(window.sum_dim(1));
    }

    Tensor::cat(outputs, 1)
}

fn finite_padding<B: Backend>(trace: &Trace<B>, padding: PaddingMode) -> Trace<B> {
    match padding {
        PaddingMode::Zero => trace.zeros_like().narrow(1, 0, 1),
        PaddingMode::Same => select_time(trace, 0),
        PaddingMode::Custom(value) => trace.full_like(value).narrow(1, 0, 1),
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expression(expr) => write!(f, "{expr}"),
            Self::Predicate { lhs, op, rhs } => {
                let op = match op {
                    PredicateOp::LessEqual => "<=",
                    PredicateOp::GreaterEqual => ">=",
                    PredicateOp::Equal => "==",
                };
                write!(f, "{lhs} {op} {rhs}")
            }
            Self::Not(subformula) => write!(f, "!({subformula})"),
            Self::And(children) => join_formula(f, children, " && "),
            Self::Or(children) => join_formula(f, children, " || "),
            Self::Implies(lhs, rhs) => write!(f, "({lhs}) => ({rhs})"),
            Self::Always {
                subformula,
                interval,
            } => write!(f, "always {:?} ({subformula})", interval),
            Self::Eventually {
                subformula,
                interval,
            } => write!(f, "eventually {:?} ({subformula})", interval),
            Self::Until { lhs, rhs, .. } => write!(f, "({lhs}) until ({rhs})"),
            Self::Then { lhs, rhs, .. } => write!(f, "({lhs}) then ({rhs})"),
            Self::Integral {
                subformula,
                options,
            } => write!(f, "integral {:?} ({subformula})", options),
        }
    }
}

impl Not for Formula {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

fn join_formula(f: &mut fmt::Formatter<'_>, children: &[Formula], sep: &str) -> fmt::Result {
    for (index, child) in children.iter().enumerate() {
        if index > 0 {
            write!(f, "{sep}")?;
        }
        write!(f, "({child})")?;
    }
    Ok(())
}
