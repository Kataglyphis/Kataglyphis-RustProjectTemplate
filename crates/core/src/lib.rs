//! Core components for the Kataglyphis project.

//! Core utilities: configuration, detection types, and logging.
//!
//! # Features
//! - Environment-based config with caching
//! - Unified `Detection` struct for inference output
//! - Structured logging

pub mod config;
pub mod detection;
pub mod logging;

pub use config::*;
pub use detection::*;
pub use logging::*;
