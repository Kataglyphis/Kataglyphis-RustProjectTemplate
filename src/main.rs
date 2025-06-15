// src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;

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
    }
    Ok(())
}
