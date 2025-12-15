// src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;

#[cfg(all(feature = "gui_unix", not(windows)))]
mod gui;
#[cfg(feature = "gui_windows")]
mod gui_windows;
mod utils;

#[derive(Parser)]
#[command(name = "file_tool")]
#[command(about = "A CLI tool that reads and analyzes files.", long_about = None)]
struct Cli {
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
    env_logger::init();
    let cli = Cli::parse();

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

            #[cfg(not(any(
                feature = "gui_windows",
                all(feature = "gui_unix", not(windows))
            )))]
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
