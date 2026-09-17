//! Error and status types for the pure-Rust KLU implementation.

/// KLU status codes (a subset of SuiteSparse's `KLU_*` constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum KluStatus {
    /// Success.
    Ok = 0,
    /// Out of memory.
    OutOfMemory = 1,
    /// Invalid input.
    Invalid = 2,
    /// Singular matrix.
    Singular = 4,
    /// Too large / integer overflow.
    TooLarge = 5,
}

/// Errors returned by the KLU entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KluError {
    /// Not yet implemented (skeleton stages).
    NotImplemented,
    /// Invalid argument, with a human-readable message.
    InvalidArgument(String),
    /// The matrix is singular.
    Singular,
    /// Allocation failure.
    OutOfMemory,
    /// Size overflow.
    TooLarge,
}

impl core::fmt::Display for KluError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(f, "not implemented"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::Singular => write!(f, "singular matrix"),
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::TooLarge => write!(f, "matrix too large"),
        }
    }
}

impl std::error::Error for KluError {}

impl KluError {
    /// The corresponding [`KluStatus`] code.
    pub fn status(&self) -> KluStatus {
        match self {
            Self::NotImplemented | Self::InvalidArgument(_) => KluStatus::Invalid,
            Self::Singular => KluStatus::Singular,
            Self::OutOfMemory => KluStatus::OutOfMemory,
            Self::TooLarge => KluStatus::TooLarge,
        }
    }
}
