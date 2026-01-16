use anyhow::Context;
use burn::module::Module;
use burn::nn;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor, TensorData};
use ndarray::{Array4, Axis};
use ort::session::Session;
use std::path::Path;
use std::time::Instant;

pub fn onnx_yolov10_demo<TrainB: AutodiffBackend>(
    model_path: &Path,
    warmup: usize,
    runs: usize,
    print_topk: usize,
    train_adapter_steps: usize,
    lr: f64,
    train_device: &TrainB::Device,
) -> anyhow::Result<()> {
    let mut builder = Session::builder().context("Failed to create ORT SessionBuilder")?;

    #[cfg(all(feature = "onnxruntime_directml", windows))]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .context("Failed to configure ORT DirectML execution provider")?;
    }

    let mut session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?;

    println!("ONNX model loaded: {}", model_path.display());
    for (i, input) in session.inputs.iter().enumerate() {
        println!("input[{i}] name={} type={:?}", input.name, input.input_type);
    }
    for (i, output) in session.outputs.iter().enumerate() {
        println!("output[{i}] name={} type={:?}", output.name, output.output_type);
    }

    // Warmup.
    for i in 0..warmup {
        let input = make_demo_image_1x3x640x640(i as u64);
        let out = run_once(&mut session, input).context("warmup run")?;
        if i == 0 {
            println!("warmup output0 shape: {:?}", out.0);
        }
    }

    // Timed runs.
    let mut last_output = None;
    let start = Instant::now();
    for i in 0..runs.max(1) {
        let input = make_demo_image_1x3x640x640(1234 + i as u64);
        last_output = Some(run_once(&mut session, input).context("timed run")?);
    }
    let elapsed = start.elapsed();

    if runs > 0 {
        println!("runs={runs} elapsed={elapsed:?} per_run={:?}", elapsed / (runs as u32));
    }

    let (shape, out) = last_output.context("missing output")?;
    println!("output0 shape: {:?}", shape);

    // Expected YOLOv10 export: [1, N, 6]
    let out3 = out
        .into_dimensionality::<ndarray::Ix3>()
        .context("expected output0 to have 3 dimensions")?;
    let first_batch = out3.index_axis(Axis(0), 0);
    let topk = print_topk.min(first_batch.len_of(Axis(0)));
    for i in 0..topk {
        let row = first_batch.index_axis(Axis(0), i);
        // xyxy + score + class
        println!(
            "det[{i}] x1={:.1} y1={:.1} x2={:.1} y2={:.1} score={:.3} class={}",
            row[0], row[1], row[2], row[3], row[4], row[5] as i64
        );
    }

    // Optional: train a tiny adapter layer on frozen ONNX outputs.
    if train_adapter_steps > 0 {
        let first = first_batch.to_owned();
        train_adapter::<TrainB>(&first, train_adapter_steps, lr, train_device)?;
    }

    Ok(())
}

fn run_once(
    session: &mut Session,
    input: Array4<f32>,
) -> anyhow::Result<(Vec<usize>, ndarray::ArrayD<f32>)> {
    let input_tensor = ort::value::Tensor::from_array(input).context("create ORT input tensor")?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .context("run ORT")?;

    if outputs.len() == 0 {
        anyhow::bail!("model returned no outputs");
    }

    let out = &outputs[0];
    let (shape, data) = out
        .try_extract_tensor::<f32>()
        .context("extract f32 tensor")?;

    let shape: Vec<usize> = shape
        .iter()
        .map(|d| {
            if *d < 0 {
                anyhow::bail!("ORT output had dynamic/negative dimension: {d}");
            }
            Ok(*d as usize)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let arr = ndarray::ArrayD::from_shape_vec(shape.clone(), data.to_vec())
        .context("reshape output tensor")?;

    Ok((shape, arr))
}

#[derive(Module, Debug)]
struct OutputAdapter<B: Backend> {
    linear: nn::Linear<B>,
}

impl<B: Backend> OutputAdapter<B> {
    fn new(device: &B::Device) -> Self {
        let linear = nn::LinearConfig::new(6, 6).init(device);
        Self { linear }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn make_adapter_batch<B: Backend>(
    device: &B::Device,
    features_300x6: &ndarray::Array2<f32>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let flat: Vec<f32> = features_300x6.iter().copied().collect();
    let x = Tensor::<B, 2>::from_data(TensorData::new(flat.clone(), [features_300x6.nrows(), 6]), device);
    let y = Tensor::<B, 2>::from_data(TensorData::new(flat, [features_300x6.nrows(), 6]), device);
    (x, y)
}

fn train_adapter<TrainB: AutodiffBackend>(
    features: &ndarray::Array2<f32>,
    steps: usize,
    lr: f64,
    device: &TrainB::Device,
) -> anyhow::Result<()> {
    let mut model = OutputAdapter::<TrainB>::new(device);
    let mut optim = AdamConfig::new().init();

    for step in 0..steps {
        let (x, y) = make_adapter_batch::<TrainB>(device, features);
        let pred = model.forward(x);

        let loss = (pred - y).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);

        if step % 5 == 0 {
            println!("adapter step {step}/{steps} loss={:.6}", loss.into_scalar());
        }
    }

    Ok(())
}

fn make_demo_image_1x3x640x640(seed: u64) -> Array4<f32> {
    const B: usize = 1;
    const C: usize = 3;
    const H: usize = 640;
    const W: usize = 640;

    let len = B * C * H * W;
    let mut data = Vec::with_capacity(len);
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (state >> 32) as u32;
        data.push((v as f32) / (u32::MAX as f32));
    }

    Array4::from_shape_vec((B, C, H, W), data).expect("shape should be valid")
}
