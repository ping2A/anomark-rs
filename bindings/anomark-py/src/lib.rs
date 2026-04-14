//! Python bindings for anomark-rs (character and token Markov models).

use anomark::{
    is_linux_kernel_thread_command, MarkovModel, ModelHandler, TokenMarkovModel, Tokenizer,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// Loaded character-level Markov model (same `.bin` format as `apply-model`).
#[pyclass(name = "CharModel")]
pub struct PyCharModel {
    inner: MarkovModel,
}

#[pymethods]
impl PyCharModel {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let mut inner = ModelHandler::load_model(path).map_err(|e| py_err(e))?;
        if !inner.is_trained() {
            inner.normalize_model_and_compute_prior();
        }
        Ok(Self { inner })
    }

    /// Log-likelihood score for `command` (higher = more typical vs training). Same padding as the CLI.
    fn score(&self, command: &str) -> f64 {
        let padded = format!("{}{}", "~".repeat(self.inner.order), command);
        self.inner.log_likelihood(&padded)
    }

    /// Baseline threshold (95% of prior log); scores below this are “suspect” in the CLI sense.
    fn suspect_threshold(&self) -> f64 {
        ModelHandler::compute_threshold(&self.inner, 95.0)
    }

    fn is_suspect(&self, score: f64) -> bool {
        ModelHandler::is_suspect_command(score, self.suspect_threshold())
    }

    #[getter]
    fn order(&self) -> usize {
        self.inner.order
    }

    #[getter]
    fn prior(&self) -> f64 {
        self.inner.prior
    }

    fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    fn num_contexts(&self) -> usize {
        self.inner.num_contexts()
    }

    fn num_transitions(&self) -> usize {
        self.inner.num_transitions()
    }

    fn alphabet_len(&self) -> usize {
        self.inner.alphabet_len()
    }

    /// Return unusual (ngram, log_prob) pairs for explainability; `threshold_percent` e.g. 95.0.
    fn explain(&self, command: &str, threshold_percent: f64) -> Vec<(String, f64)> {
        let threshold = ModelHandler::compute_threshold(&self.inner, threshold_percent);
        let padded = format!("{}{}", "~".repeat(self.inner.order), command);
        self.inner.explain(&padded, threshold).1
    }
}

/// Loaded token-level Markov model.
#[pyclass(name = "TokenModel")]
pub struct PyTokenModel {
    inner: TokenMarkovModel,
}

#[pymethods]
impl PyTokenModel {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = ModelHandler::load_token_model(path).map_err(|e| py_err(e))?;
        Ok(Self { inner })
    }

    fn score(&self, command: &str) -> f64 {
        self.inner.log_likelihood(command)
    }

    fn suspect_threshold(&self) -> f64 {
        self.inner.prior.ln() * 0.95
    }

    fn is_suspect(&self, score: f64) -> bool {
        score < self.suspect_threshold()
    }

    #[getter]
    fn order(&self) -> usize {
        self.inner.order
    }

    #[getter]
    fn prior(&self) -> f64 {
        self.inner.prior
    }

    fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    fn num_contexts(&self) -> usize {
        self.inner.num_contexts()
    }

    fn num_transitions(&self) -> usize {
        self.inner.num_transitions()
    }

    fn tokenizer_name(&self) -> String {
        match &self.inner.tokenizer {
            Tokenizer::Whitespace => "whitespace".to_string(),
            Tokenizer::PathSegments => "path".to_string(),
            Tokenizer::WhitespaceAndPath => "whitespace_and_path".to_string(),
        }
    }

    fn explain(&self, command: &str, threshold_percent: f64) -> Vec<(String, f64)> {
        let threshold = self.inner.prior.ln() * threshold_percent / 100.0;
        self.inner.explain(command, threshold).1
    }
}

/// True if `cmd` looks like a Linux kernel thread name (e.g. `[kthreadd]`).
#[pyfunction]
#[pyo3(name = "is_kernel_thread")]
fn is_kernel_thread_py(cmd: &str) -> bool {
    is_linux_kernel_thread_command(cmd)
}

/// Python module: `import anomark_rs`
#[pymodule]
fn anomark_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCharModel>()?;
    m.add_class::<PyTokenModel>()?;
    m.add_function(wrap_pyfunction!(is_kernel_thread_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
