// src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;
use std::io::Write;

#[cfg(all(feature = "gui_unix", not(windows)))]
mod gui;
#[cfg(feature = "gui_windows")]
mod gui_windows;

mod logging;
#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
mod person_detection;
mod resource_monitor;
mod utils;

#[derive(Parser)]
#[command(name = "file_tool")]
#[command(about = "A CLI tool that reads and analyzes files.", long_about = None)]
struct Cli {
    /// Enable lightweight periodic resource logging (CPU/RAM and best-effort GPU on Windows).
    #[arg(long, global = true, default_value_t = false)]
    resource_log: bool,

    /// Resource log interval in milliseconds.
    #[arg(long, global = true, default_value_t = 1000)]
    resource_log_interval_ms: u64,

    /// Optional file path to append resource log lines to (in addition to standard logging).
    #[arg(long, global = true)]
    resource_log_file: Option<std::path::PathBuf>,

    /// Include GPU metrics in resource logging (best-effort; Windows only).
    #[arg(
        long,
        global = true,
        default_value_t = true,
        value_parser = clap::value_parser!(bool),
        action = clap::ArgAction::Set
    )]
    resource_log_gpu: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Read {
        #[arg(short, long)]
        path: String,
    },
    Stats {
        #[arg(short, long)]
        path: String,
    },
    Gui {
        /// WGPU backend to use (dx12 | vulkan | primary).
        /// Note: dx12 is only available on Windows.
        #[arg(long, default_value = "dx12")]
        backend: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut logger_builder = env_logger::Builder::from_env(env_logger::Env::default());

    // Loguru-like format: `HH:MM:SS | LEVEL | message`
    logger_builder.format(|buf, record| {
        let ts = chrono::Local::now().format("%H:%M:%S");
        let level = if record.target() == "SUCCESS" {
            "SUCCESS"
        } else {
            match record.level() {
                log::Level::Error => "ERROR",
                log::Level::Warn => "WARN",
                log::Level::Info => "INFO",
                log::Level::Debug => "DEBUG",
                log::Level::Trace => "TRACE",
            }
        };

        writeln!(buf, "{ts} | {level:<8} | {}", record.args())
    });

    // Default logging behavior:
    // - If `RUST_LOG` is set, let the user fully control logging.
    // - Otherwise, default to INFO so loguru-style diagnostics are visible.
    // - Optionally override with `KATAGLYPHIS_LOG_LEVEL` (error|warn|info|debug|trace).
    // - Also suppress very noisy INFO logs from wgpu/naga (e.g. generated shader dumps).
    if std::env::var_os("RUST_LOG").is_none() {
        let level = std::env::var("KATAGLYPHIS_LOG_LEVEL")
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref()
            .and_then(|v| match v {
                "error" => Some(log::LevelFilter::Error),
                "warn" | "warning" => Some(log::LevelFilter::Warn),
                "info" => Some(log::LevelFilter::Info),
                "debug" => Some(log::LevelFilter::Debug),
                "trace" => Some(log::LevelFilter::Trace),
                _ => None,
            })
            .unwrap_or(log::LevelFilter::Info);

        logger_builder.filter_level(level);

        logger_builder.filter_module("wgpu", log::LevelFilter::Warn);
        logger_builder.filter_module("wgpu_core", log::LevelFilter::Warn);
        logger_builder.filter_module("wgpu_hal", log::LevelFilter::Warn);
        logger_builder.filter_module("naga", log::LevelFilter::Warn);
    }
    logger_builder.init();

    let _resource_monitor = if cli.resource_log {
        Some(resource_monitor::ResourceMonitor::start(
            resource_monitor::ResourceMonitorConfig {
                interval: std::time::Duration::from_millis(cli.resource_log_interval_ms.max(100)),
                log_file: cli.resource_log_file.clone(),
                include_gpu: cli.resource_log_gpu,
            },
        ))
    } else {
        None
    };

    match cli.command {
        Commands::Read { path } => {
            info!("Reading file: {}", path);
            let content = utils::read_file(&path).await?;
            println!("{}", content);
        }
        Commands::Stats { path } => {
            let stats = utils::file_stats(&path).await?;
            println!(
                "Lines: {}, Words: {}, Bytes: {}",
                stats.lines, stats.words, stats.bytes
            );
        }
        Commands::Gui { backend } => {
            #[cfg(feature = "gui_windows")]
            {
                // WGPU/Winit GUI (cross-platform)
                gui_windows::run_with_backend(&backend);
            }

            #[cfg(all(feature = "gui_unix", not(windows), not(feature = "gui_windows")))]
            {
                // GTK/GStreamer demo GUI (non-async)
                let _ = backend;
                gui::run();
            }

            #[cfg(not(any(feature = "gui_windows", all(feature = "gui_unix", not(windows)))))]
            {
                eprintln!(
                    "GUI feature is disabled. Enable --features gui_windows (WGPU) or gui_unix (GTK on non-Windows)."
                );
                let _ = backend;
            }
        }
    }
    Ok(())
}
