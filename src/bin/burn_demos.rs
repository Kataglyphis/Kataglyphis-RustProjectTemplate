use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "burn-demos", about = "Burn demos for common PyTorch tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Demonstrates PyTorch-like tensor basics: matmul, broadcast, reductions.
    TensorDemo,

    /// Fits y = 3x + 2 (with light noise) using a single Linear layer.
    LinearRegression {
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 50)]
        steps_per_epoch: usize,
        #[arg(long, default_value_t = 0.02)]
        lr: f64,
        #[arg(long, default_value_t = 256)]
        batch_size: usize,
        /// Save loss curve as PNG.
        #[arg(long)]
        plot_path: Option<std::path::PathBuf>,
    },

    /// Learns XOR using a small MLP.
    Xor {
        #[arg(long, default_value_t = 2000)]
        epochs: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f64,
        /// Save loss curve as PNG.
        #[arg(long)]
        plot_path: Option<std::path::PathBuf>,
    },

    /// Trains a deeper MLP on a synthetic 2D "two moons" classification dataset.
    TwoMoons {
        #[arg(long, default_value_t = 200)]
        epochs: usize,
        #[arg(long, default_value_t = 50)]
        steps_per_epoch: usize,
        #[arg(long, default_value_t = 0.01)]
        lr: f64,
        #[arg(long, default_value_t = 256)]
        batch_size: usize,
        #[arg(long, default_value_t = 0.12)]
        noise: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Save loss curve as PNG.
        #[arg(long)]
        plot_path: Option<std::path::PathBuf>,
    },

    /// Builds a tiny YOLO-like conv net and runs a forward pass (optionally a couple train steps).
    YoloTiny {
        #[arg(long, default_value_t = 128)]
        height: usize,
        #[arg(long, default_value_t = 128)]
        width: usize,
        #[arg(long, default_value_t = 20)]
        num_classes: usize,
        #[arg(long, default_value_t = 3)]
        num_anchors: usize,
        #[arg(long, default_value_t = 0)]
        train_steps: usize,
        #[arg(long, default_value_t = 0.001)]
        lr: f64,
    },

    /// Loads models/yolov10m.onnx via ONNX Runtime and runs inference (optional: trains a tiny adapter on outputs).
    OnnxYolov10 {
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        #[arg(long, default_value_t = 1)]
        runs: usize,
        #[arg(long, default_value_t = 3)]
        print_topk: usize,
        #[arg(long, default_value_t = 0)]
        train_adapter_steps: usize,
        #[arg(long, default_value_t = 0.001)]
        lr: f64,
        #[arg(long)]
        model_path: Option<std::path::PathBuf>,
    },

    /// Save/Load demonstration using Burn recorders.
    SaveLoad,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::TensorDemo => {
            kataglyphis_rustprojecttemplate::burn_demos::simple::tensor_demo::<
                kataglyphis_rustprojecttemplate::burn_demos::InferenceBackend,
            >()
        }

        Command::LinearRegression {
            epochs,
            steps_per_epoch,
            lr,
            batch_size,
            plot_path,
        } => kataglyphis_rustprojecttemplate::burn_demos::simple::linear_regression_demo(
            epochs,
            steps_per_epoch,
            lr,
            batch_size,
            plot_path,
        ),

        Command::Xor {
            epochs,
            lr,
            plot_path,
        } => kataglyphis_rustprojecttemplate::burn_demos::simple::xor_demo(epochs, lr, plot_path),

        Command::TwoMoons {
            epochs,
            steps_per_epoch,
            lr,
            batch_size,
            noise,
            seed,
            plot_path,
        } => kataglyphis_rustprojecttemplate::burn_demos::two_moons::two_moons_demo(
            epochs,
            steps_per_epoch,
            lr,
            batch_size,
            noise,
            seed,
            plot_path,
        ),

        Command::YoloTiny {
            height,
            width,
            num_classes,
            num_anchors,
            train_steps,
            lr,
        } => kataglyphis_rustprojecttemplate::burn_demos::simple::yolo_tiny_demo(
            height,
            width,
            num_classes,
            num_anchors,
            train_steps,
            lr,
        ),

        Command::OnnxYolov10 {
            warmup,
            runs,
            print_topk,
            train_adapter_steps,
            lr,
            model_path,
        } => {
            let model_path = model_path.unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
                    .join("yolov10m.onnx")
            });

            let device = <kataglyphis_rustprojecttemplate::burn_demos::TrainingBackend as burn::tensor::backend::Backend>::Device::default();
            kataglyphis_rustprojecttemplate::burn_demos::onnx_yolov10::onnx_yolov10_demo::<
                kataglyphis_rustprojecttemplate::burn_demos::TrainingBackend,
            >(
                &model_path,
                warmup,
                runs,
                print_topk,
                train_adapter_steps,
                lr,
                &device,
            )
            .with_context(|| format!("onnx-yolov10 demo failed (model={})", model_path.display()))
        }

        Command::SaveLoad => kataglyphis_rustprojecttemplate::burn_demos::simple::save_load_demo(),
    }
}
