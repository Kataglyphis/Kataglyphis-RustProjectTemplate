// src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;

#[cfg(all(feature = "gui_unix", not(windows)))]
mod gui;
#[cfg(all(feature = "gui_windows", target_os = "windows"))]
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
        /// WGPU backend to use on Windows (dx12 | vulkan | primary).
        /// Ignored on non-Windows GUIs.
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
            #[cfg(all(feature = "gui_unix", not(windows)))]
            {
                // GTK/GStreamer demo GUI (non-async)
                let _ = backend;
                gui::run();
            }

            #[cfg(all(feature = "gui_windows", target_os = "windows"))]
            {
                gui_windows::run_with_backend(&backend);
            }

            #[cfg(not(any(
                all(feature = "gui_unix", not(windows)),
                all(feature = "gui_windows", target_os = "windows")
            )))]
            {
                eprintln!(
                    "GUI feature is disabled. Enable --features gui_unix (Linux/macOS) or gui_windows (Windows)."
                );
                let _ = backend;
            }
        }
    }
    Ok(())
}
