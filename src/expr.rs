use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use burn::tensor::backend::Backend;

use crate::{Formula, Result, SignalEnv, Trace};

/// Signal expression used in predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Named signal looked up in a [`SignalEnv`].
    Var(String),
    /// Scalar value broadcast to the current trace shape during evaluation.
    Scalar(f64),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Elementwise addition.
    Add(Box<Expr>, Box<Expr>),
    /// Elementwise subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Elementwise multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Elementwise division.
    Div(Box<Expr>, Box<Expr>),
    /// Elementwise absolute value.
    Abs(Box<Expr>),
}

/// Create a named signal expression.
pub fn var(name: impl Into<String>) -> Expr {
    Expr::Var(name.into())
}

/// Create a scalar expression.
pub const fn scalar(value: f64) -> Expr {
    Expr::Scalar(value)
}

impl Expr {
    /// Take the elementwise absolute value of this expression.
    pub fn abs(self) -> Self {
        Self::Abs(Box::new(self))
    }

    /// Build a predicate whose robustness is positive when `self <= rhs`.
    pub fn le(self, rhs: impl Into<Expr>) -> Formula {
        Formula::less_equal(self, rhs.into())
    }

    /// Build a predicate whose robustness is positive when `self >= rhs`.
    pub fn ge(self, rhs: impl Into<Expr>) -> Formula {
        Formula::greater_equal(self, rhs.into())
    }

    /// Build an equality predicate with robustness `-abs(self - rhs)`.
    pub fn eq_value(self, rhs: impl Into<Expr>) -> Formula {
        Formula::equal(self, rhs.into())
    }

    /// Treat this expression directly as a robustness-valued formula.
    pub fn into_formula(self) -> Formula {
        Formula::expression(self)
    }

    pub(crate) fn eval<B: Backend>(&self, env: &SignalEnv<B>) -> Result<Trace<B>> {
        match self {
            Self::Var(name) => env.get(name),
            Self::Scalar(value) => {
                let template = env.template()?;
                Ok(template.full_like(*value))
            }
            Self::Neg(expr) => Ok(expr.eval(env)?.neg()),
            Self::Add(lhs, rhs) => Ok(lhs.eval(env)? + rhs.eval(env)?),
            Self::Sub(lhs, rhs) => Ok(lhs.eval(env)? - rhs.eval(env)?),
            Self::Mul(lhs, rhs) => Ok(lhs.eval(env)? * rhs.eval(env)?),
            Self::Div(lhs, rhs) => Ok(lhs.eval(env)? / rhs.eval(env)?),
            Self::Abs(expr) => Ok(expr.eval(env)?.abs()),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(name) => write!(f, "{name}"),
            Self::Scalar(value) => write!(f, "{value}"),
            Self::Neg(expr) => write!(f, "-({expr})"),
            Self::Add(lhs, rhs) => write!(f, "({lhs} + {rhs})"),
            Self::Sub(lhs, rhs) => write!(f, "({lhs} - {rhs})"),
            Self::Mul(lhs, rhs) => write!(f, "({lhs} * {rhs})"),
            Self::Div(lhs, rhs) => write!(f, "({lhs} / {rhs})"),
            Self::Abs(expr) => write!(f, "abs({expr})"),
        }
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Self::Scalar(value)
    }
}

impl From<f32> for Expr {
    fn from(value: f32) -> Self {
        Self::Scalar(value as f64)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        Self::Scalar(value as f64)
    }
}

impl From<usize> for Expr {
    fn from(value: usize) -> Self {
        Self::Scalar(value as f64)
    }
}

impl Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl Add<f64> for Expr {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Self::Scalar(rhs)
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Sub<f64> for Expr {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::Scalar(rhs)
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Mul<f64> for Expr {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Self::Scalar(rhs)
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::Div(Box::new(self), Box::new(rhs))
    }
}

impl Div<f64> for Expr {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self / Self::Scalar(rhs)
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::Neg(Box::new(self))
    }
}
