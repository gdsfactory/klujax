//! Error type for the `klu` crate.

use core::fmt;

/// Broad category of a [`Error`], useful for callers that map to their own
/// error types (e.g. XLA's `XLA_FFI_Error_Code`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Invalid input (bad shapes, indices, or non-matching handles).
    InvalidArgument,
    /// The matrix is singular.
    Singular,
    /// A KLU/internal failure.
    Internal,
}

/// Error returned by the `klu` entry points.
#[derive(Debug, Clone)]
pub struct Error {
    /// Category of the error.
    pub kind: ErrorKind,
    /// Human-readable message.
    pub message: String,
}

impl Error {
    /// An `InvalidArgument` error.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidArgument,
            message: message.into(),
        }
    }

    /// A `Singular` error.
    pub fn singular(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Singular,
            message: message.into(),
        }
    }

    /// An `Internal` error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

/// Convenience result alias.
pub type Result<T> = core::result::Result<T, Error>;
