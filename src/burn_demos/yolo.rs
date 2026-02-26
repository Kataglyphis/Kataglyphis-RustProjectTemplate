use burn::module::Module;
use burn::nn;
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Module, Debug)]
pub struct YoloTiny<B: Backend> {
    c1: nn::conv::Conv2d<B>,
    c2: nn::conv::Conv2d<B>,
    c3: nn::conv::Conv2d<B>,
    c4: nn::conv::Conv2d<B>,
    c5: nn::conv::Conv2d<B>,
    head: nn::conv::Conv2d<B>,
}

impl<B: Backend> YoloTiny<B> {
    /// A tiny YOLO-like detector head.
    ///
    /// Output tensor shape: [batch, anchors * (5 + num_classes), grid_h, grid_w]
    pub fn new(device: &B::Device, num_classes: usize, num_anchors: usize) -> Self {
        let out_channels = num_anchors * (5 + num_classes);

        let c1 = nn::conv::Conv2dConfig::new([3, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let c2 = nn::conv::Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let c3 = nn::conv::Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let c4 = nn::conv::Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let c5 = nn::conv::Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let head = nn::conv::Conv2dConfig::new([256, out_channels], [1, 1]).init(device);

        Self {
            c1,
            c2,
            c3,
            c4,
            c5,
            head,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.c1.forward(x));
        let x = relu(self.c2.forward(x));
        let x = relu(self.c3.forward(x));
        let x = relu(self.c4.forward(x));
        let x = relu(self.c5.forward(x));
        self.head.forward(x)
    }

    pub fn demo_input(
        device: &B::Device,
        batch: usize,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let mut rng = Lcg::new(42);
        let mut data = Vec::with_capacity(batch * 3 * height * width);
        for _ in 0..(batch * 3 * height * width) {
            data.push(rng.next_f32());
        }
        Tensor::<B, 4>::from_data(TensorData::new(data, [batch, 3, height, width]), device)
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
}
