// src/main.rs
use anyhow::Result;
use clap::Parser;
use log::info;

mod cli;
#[cfg(all(feature = "gui_unix", not(windows)))]
mod gui;
#[cfg(feature = "gui_windows")]
mod gui_wgpu;

mod detection;
mod logging;
#[cfg(onnx)]
mod person_detection;
mod resource_monitor;
mod utils;

use cli::{Cli, Commands};

#[cfg_attr(not(target_arch = "wasm32"), tokio::main)]
#[cfg_attr(target_arch = "wasm32", tokio::main(flavor = "current_thread"))]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    logging::init_logger();

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
                gui_wgpu::run_with_backend(&backend);
            }

            #[cfg(all(feature = "gui_unix", not(windows), not(feature = "gui_windows")))]
            {
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
