// src/cli.rs — Command-line interface definition.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[cfg(feature = "gui_windows")]
use kataglyphis_gui::gui_wgpu::GpuBackend;

#[derive(Parser)]
#[command(name = "kataglyphis")]
#[command(
    about = "Rust project template: CLI tools, optional GUI, ONNX inference, and resource monitoring.",
    long_about = None
)]
pub struct Cli {
    /// Enable lightweight periodic resource logging (CPU/RAM and best-effort GPU on Windows).
    #[arg(long, global = true, default_value_t = false)]
    pub resource_log: bool,

    /// Resource log interval in milliseconds.
    #[arg(long, global = true, default_value_t = 1000)]
    pub resource_log_interval_ms: u64,

    /// Optional file path to append resource log lines to (in addition to standard logging).
    #[arg(long, global = true)]
    pub resource_log_file: Option<PathBuf>,

    /// Include GPU metrics in resource logging (best-effort; Windows only).
    #[arg(
        long,
        global = true,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        action = clap::ArgAction::Set
    )]
    pub resource_log_gpu: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Read the contents of a file and print them to stdout.
    Read {
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Print line / word / byte statistics for a file.
    Stats {
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Launch the GUI (requires a gui feature).
    Gui {
        /// WGPU backend to use.
        /// Note: dx12 is only available on Windows.
        #[cfg(feature = "gui_windows")]
        #[arg(long, value_enum, default_value_t = GpuBackend::Auto)]
        backend: GpuBackend,
        #[cfg(not(feature = "gui_windows"))]
        #[arg(long, default_value = "")]
        backend: String,
    },
}
