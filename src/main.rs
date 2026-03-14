// src/main.rs — Thin CLI driver.  All logic lives in the library crate.
use anyhow::Result;
use clap::Parser;
use log::info;

mod cli;

use cli::{Cli, Commands};
use kataglyphis_rustprojecttemplate::{logging, resource_monitor, utils};

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
            info!("Reading file: {}", path.display());
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
                kataglyphis_rustprojecttemplate::gui_wgpu::run_with_backend(&backend)?;
            }

            #[cfg(all(feature = "gui_unix", not(windows), not(feature = "gui_windows")))]
            {
                let _ = backend;
                kataglyphis_rustprojecttemplate::gui::run()?;
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
