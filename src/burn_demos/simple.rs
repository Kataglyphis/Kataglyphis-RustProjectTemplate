use anyhow::Context;
use burn::module::Module;
use burn::nn;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::Backend, Tensor, TensorData};

use crate::burn_demos::{plot, InferenceBackend, TrainingBackend};

pub fn tensor_demo<B: Backend>() -> anyhow::Result<()> {
    let device = B::Device::default();

    let a = Tensor::<B, 2>::from_data(TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]), &device);
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![5.0, 6.0, 7.0, 8.0], [2, 2]), &device);

    let c = a.clone().matmul(b.clone());
    println!("a=\n{:?}", a.to_data());
    println!("b=\n{:?}", b.to_data());
    println!("a@b=\n{:?}", c.to_data());

    let v = Tensor::<B, 2>::from_data(TensorData::new(vec![1.0, 2.0], [1, 2]), &device);
    let d = c + v;
    println!("broadcast add=\n{:?}", d.to_data());

    let sum = d.sum();
    println!("sum={:?}", sum.into_scalar());

    Ok(())
}

#[derive(Module, Debug)]
struct LinearRegressor<B: Backend> {
    linear: nn::Linear<B>,
}

impl<B: Backend> LinearRegressor<B> {
    fn new(device: &B::Device) -> Self {
        let linear = nn::LinearConfig::new(1, 1).init(device);
        Self { linear }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn make_regression_batch<B: Backend>(device: &B::Device, batch: usize, step: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // y = 3x + 2 + noise
    let mut xs = Vec::with_capacity(batch);
    let mut ys = Vec::with_capacity(batch);

    let mut state = (step as u64) ^ 0xD6E8_FEB8_6659_FD93;
    for i in 0..batch {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (state >> 32) as u32;
        let x = (r as f32) / (u32::MAX as f32) * 2.0 - 1.0;

        // small deterministic noise
        let noise = (((i as f32) * 12.9898 + (step as f32) * 78.233).sin() * 43758.5453).fract() * 0.05;
        let y = 3.0 * x + 2.0 + noise;

        xs.push(x);
        ys.push(y);
    }

    let x = Tensor::<B, 2>::from_data(TensorData::new(xs, [batch, 1]), device);
    let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [batch, 1]), device);
    (x, y)
}

pub fn linear_regression_demo(
    epochs: usize,
    steps_per_epoch: usize,
    lr: f64,
    batch_size: usize,
    plot_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let device = <TrainingBackend as Backend>::Device::default();
    let mut model = LinearRegressor::<TrainingBackend>::new(&device);
    let mut optim = AdamConfig::new().init();

    let mut losses = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut loss_sum = 0.0f32;
        for step in 0..steps_per_epoch {
            let (x, y) = make_regression_batch::<TrainingBackend>(&device, batch_size, epoch * steps_per_epoch + step);
            let pred = model.forward(x);
            let loss = (pred - y).powf_scalar(2.0).mean();

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            loss_sum += loss.into_scalar();
        }

        let loss_avg = loss_sum / steps_per_epoch.max(1) as f32;
        losses.push(loss_avg);
        if epoch % 10 == 0 {
            println!("epoch {epoch} loss={loss_avg:.6}");
        }
    }

    if let Some(path) = plot_path {
        plot::plot_loss_curve(path, &losses, "Linear regression loss")?;
    }

    Ok(())
}

#[derive(Module, Debug)]
struct XorNet<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> XorNet<B> {
    fn new(device: &B::Device) -> Self {
        let l1 = nn::LinearConfig::new(2, 16).init(device);
        let l2 = nn::LinearConfig::new(16, 1).init(device);
        Self { l1, l2 }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(x));
        sigmoid(self.l2.forward(x))
    }
}

fn xor_dataset<B: Backend>(device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let x = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let x = Tensor::<B, 2>::from_data(TensorData::new(x, [4, 2]), device);
    let y = Tensor::<B, 2>::from_data(TensorData::new(y, [4, 1]), device);
    (x, y)
}

pub fn xor_demo(epochs: usize, lr: f64, plot_path: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let device = <TrainingBackend as Backend>::Device::default();
    let mut model = XorNet::<TrainingBackend>::new(&device);
    let mut optim = AdamConfig::new().init();

    let (x, y) = xor_dataset::<TrainingBackend>(&device);
    let mut losses = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let pred = model.forward(x.clone());

        let eps = 1e-6;
        let pred = pred.clamp(eps, 1.0 - eps);
        let one_minus_y = y.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
        let one_minus_pred = pred.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
        let loss = -(y.clone() * pred.clone().log() + one_minus_y * one_minus_pred.log()).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);

        let l = loss.into_scalar();
        losses.push(l);
        if epoch % 200 == 0 {
            println!("epoch {epoch} loss={l:.6}");
        }
    }

    if let Some(path) = plot_path {
        plot::plot_loss_curve(path, &losses, "XOR loss")?;
    }

    Ok(())
}

pub fn yolo_tiny_demo(
    height: usize,
    width: usize,
    num_classes: usize,
    num_anchors: usize,
    train_steps: usize,
    lr: f64,
) -> anyhow::Result<()> {
    use crate::burn_demos::yolo::YoloTiny;

    let device = <TrainingBackend as Backend>::Device::default();
    let mut model = YoloTiny::<TrainingBackend>::new(&device, num_classes, num_anchors);
    let mut optim = AdamConfig::new().init();

    let x = YoloTiny::<TrainingBackend>::demo_input(&device, 1, height, width);

    // Forward pass.
    let y = model.forward(x.clone());
    println!("yolo_tiny output shape: {:?}", y.dims());

    // Optional tiny training steps (dummy loss).
    for step in 0..train_steps {
        let pred = model.forward(x.clone());
        let loss = pred.mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);
        println!("train step {step}/{train_steps} loss={:.6}", loss.into_scalar());
    }

    Ok(())
}

pub fn save_load_demo() -> anyhow::Result<()> {
    let device = <InferenceBackend as Backend>::Device::default();

    // Create a tiny model, run forward, save record.
    let model = LinearRegressor::<InferenceBackend>::new(&device);
    let x = Tensor::<InferenceBackend, 2>::from_data(TensorData::new(vec![1.0], [1, 1]), &device);
    let y0 = model.forward(x.clone()).to_data();

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let path = std::path::Path::new("./tmp-linear-regressor.bin");
    let record = model.clone().into_record();
    recorder
        .record(record, path.to_path_buf())
        .context("record model")?;

    // Load into a fresh model.
    let record = recorder
        .load(path.to_path_buf(), &device)
        .context("load model record")?;

    let model2 = LinearRegressor::<InferenceBackend>::new(&device).load_record(record);
    let y1 = model2.forward(x).to_data();

    println!("save/load ok: before={:?} after={:?}", y0, y1);
    Ok(())
}
