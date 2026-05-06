use std::collections::HashMap;

use burn::tensor::backend::Backend;

use crate::{Result, StlcgError, Trace};

/// Named signal tensors used while evaluating formulas.
#[derive(Debug, Clone)]
pub struct SignalEnv<B: Backend> {
    signals: HashMap<String, Trace<B>>,
}

impl<B: Backend> SignalEnv<B> {
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
        }
    }

    pub fn with(mut self, name: impl Into<String>, trace: Trace<B>) -> Self {
        self.insert(name, trace);
        self
    }

    pub fn insert(&mut self, name: impl Into<String>, trace: Trace<B>) {
        self.signals.insert(name.into(), trace);
    }

    pub fn get(&self, name: &str) -> Result<Trace<B>> {
        self.signals
            .get(name)
            .cloned()
            .ok_or_else(|| StlcgError::MissingSignal(name.to_string()))
    }

    pub fn template(&self) -> Result<Trace<B>> {
        self.signals
            .values()
            .next()
            .cloned()
            .ok_or(StlcgError::EmptySignalEnv)
    }

    pub fn validate_compatible_shapes(&self) -> Result<()> {
        let Some((first_name, first)) = self.signals.iter().next() else {
            return Err(StlcgError::EmptySignalEnv);
        };

        let expected = first.dims();
        if expected[1] == 0 {
            return Err(StlcgError::EmptyTimeDimension);
        }

        for (name, trace) in &self.signals {
            let actual = trace.dims();
            if actual != expected {
                return Err(StlcgError::ShapeMismatch {
                    name: name.clone(),
                    expected,
                    actual,
                });
            }
        }

        let _ = first_name;
        Ok(())
    }
}

impl<B: Backend> Default for SignalEnv<B> {
    fn default() -> Self {
        Self::new()
    }
}
