use anyhow::Context;
use burn::module::{AutodiffModule, Module};
use burn::nn;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::Backend, Tensor, TensorData};

use crate::burn_demos::{plot, InferenceBackend, TrainingBackend};

#[derive(Module, Debug)]
struct DeepClassifier<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
    l3: nn::Linear<B>,
}

impl<B: Backend> DeepClassifier<B> {
    fn new(device: &B::Device, hidden: usize) -> Self {
        let l1 = nn::LinearConfig::new(2, hidden).init(device);
        let l2 = nn::LinearConfig::new(hidden, hidden).init(device);
        let l3 = nn::LinearConfig::new(hidden, 1).init(device);
        Self { l1, l2, l3 }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        sigmoid(self.l3.forward(x))
    }
}

#[derive(Clone)]
struct TwoMoonsDataset {
    x: Vec<f32>,
    y: Vec<f32>,
    n: usize,
}

impl TwoMoonsDataset {
    fn generate(n: usize, noise: f32, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);

        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);

        // Half points on each moon.
        let n0 = n / 2;
        let n1 = n - n0;

        // Moon 0: (cos t, sin t)
        for _ in 0..n0 {
            let t = rng.next_f32() * std::f32::consts::PI;
            let mut px = t.cos();
            let mut py = t.sin();

            // Noise.
            px += rng.next_normal() * noise;
            py += rng.next_normal() * noise;

            x.push(px);
            x.push(py);
            y.push(0.0);
        }

        // Moon 1: (1 - cos t, 1 - sin t) shifted
        for _ in 0..n1 {
            let t = rng.next_f32() * std::f32::consts::PI;
            let mut px = 1.0 - t.cos();
            let mut py = 1.0 - t.sin() - 0.5;

            px += rng.next_normal() * noise;
            py += rng.next_normal() * noise;

            x.push(px);
            x.push(py);
            y.push(1.0);
        }

        Self { x, y, n }
    }

    fn batch<B: Backend>(
        &self,
        device: &B::Device,
        batch_size: usize,
        step: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut xb = Vec::with_capacity(batch_size * 2);
        let mut yb = Vec::with_capacity(batch_size);

        // Deterministic-ish cycling through samples.
        for i in 0..batch_size {
            let idx = (step * batch_size + i) % self.n;
            xb.push(self.x[idx * 2]);
            xb.push(self.x[idx * 2 + 1]);
            yb.push(self.y[idx]);
        }

        let x = Tensor::<B, 2>::from_data(TensorData::new(xb, [batch_size, 2]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(yb, [batch_size, 1]), device);
        (x, y)
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32)
    }

    fn next_normal(&mut self) -> f32 {
        // Box-Muller.
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn accuracy_from_sigmoid(pred: &[f32], target: &[f32]) -> f32 {
    let mut correct = 0usize;
    let n = pred.len().min(target.len()).max(1);
    for i in 0..n {
        let p = pred[i] >= 0.5;
        let t = target[i] >= 0.5;
        if p == t {
            correct += 1;
        }
    }
    correct as f32 / n as f32
}

fn eval_two_moons_accuracy(
    model: &DeepClassifier<TrainingBackend>,
    dataset: &TwoMoonsDataset,
) -> anyhow::Result<f32> {
    let infer_device = <InferenceBackend as Backend>::Device::default();
    let infer_model = model.valid().to_device(&infer_device);

    let x_all = Tensor::<InferenceBackend, 2>::from_data(
        TensorData::new(dataset.x.clone(), [dataset.n, 2]),
        &infer_device,
    );

    let y_all = Tensor::<InferenceBackend, 2>::from_data(
        TensorData::new(dataset.y.clone(), [dataset.n, 1]),
        &infer_device,
    )
    .to_data();

    let pred = infer_model.forward(x_all).to_data();

    let pred = pred
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("TensorData cast failed: {e:?}"))?;

    let target = y_all
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("TensorData cast failed: {e:?}"))?;

    Ok(accuracy_from_sigmoid(pred, target))
}

fn train_two_moons_epoch(
    mut model: DeepClassifier<TrainingBackend>,
    optim: &mut impl Optimizer<DeepClassifier<TrainingBackend>, TrainingBackend>,
    dataset: &TwoMoonsDataset,
    device: &<TrainingBackend as Backend>::Device,
    epoch: usize,
    steps_per_epoch: usize,
    lr: f64,
    batch_size: usize,
) -> (DeepClassifier<TrainingBackend>, f32) {
    let mut loss_sum = 0.0f32;

    for step in 0..steps_per_epoch {
        let (x, y) = dataset.batch::<TrainingBackend>(device, batch_size, epoch * steps_per_epoch + step);
        let pred = model.forward(x);

        // Binary cross-entropy (manual; keeps deps minimal).
        let eps = 1e-6;
        let pred = pred.clamp(eps, 1.0 - eps);
        let one_minus_y = y.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
        let one_minus_pred = pred.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
        let loss = -(y.clone() * pred.clone().log() + one_minus_y * one_minus_pred.log()).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);
        loss_sum += loss.into_scalar();

        if step % 10 == 0 {
            println!("epoch {epoch} step {step}/{steps_per_epoch} loss={:.6}", loss_sum / (step + 1) as f32);
        }
    }

    let loss_avg = loss_sum / steps_per_epoch.max(1) as f32;
    (model, loss_avg)
}

pub fn two_moons_demo(
    epochs: usize,
    steps_per_epoch: usize,
    lr: f64,
    batch_size: usize,
    noise: f32,
    seed: u64,
    plot_path: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let device = <TrainingBackend as Backend>::Device::default();
    let dataset = TwoMoonsDataset::generate(2000, noise, seed);

    let mut model = DeepClassifier::<TrainingBackend>::new(&device, 64);
    let mut optim = AdamConfig::new().init();

    let mut losses = Vec::with_capacity(epochs);
    for epoch in 0..epochs {
        let (m, loss) = train_two_moons_epoch(model, &mut optim, &dataset, &device, epoch, steps_per_epoch, lr, batch_size);
        model = m;
        losses.push(loss);

        if epoch % 10 == 0 {
            let acc = eval_two_moons_accuracy(&model, &dataset).context("eval accuracy")?;
            println!("epoch {epoch} loss={loss:.6} acc={acc:.3}");
        }
    }

    let acc = eval_two_moons_accuracy(&model, &dataset).context("final eval accuracy")?;
    println!("final acc={acc:.3}");

    if let Some(path) = plot_path {
        plot::plot_loss_curve(path, &losses, "Two Moons loss")?;
    }

    Ok(())
}
