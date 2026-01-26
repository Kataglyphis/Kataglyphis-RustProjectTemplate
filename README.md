<div align="center">
  <a href="https://jonasheinle.de">
    <img src="images/logo.png" alt="logo" width="200" />
  </a>

  <h1>🦀 Kataglyphis-RustProjectTemplate 🦀</h1>

  <h4>Collecting Rust best practices.</h4>
</div>
  
<div align="center">
  <a href="https://jonasheinle.de">
    <img src="images/Rust.gif" alt="Rust" width="400" />
  </a>
</div>

[![Rust workflow on Ubuntu-24.04 (x86_64/ARM)](https://github.com/Kataglyphis/Kataglyphis-RustProjectTemplate/actions/workflows/rust_ubuntu24_04.yml/badge.svg)](https://github.com/Kataglyphis/Kataglyphis-RustProjectTemplate/actions/workflows/rust_ubuntu24_04.yml)
[![Rust workflow on Windows 2025](https://github.com/Kataglyphis/Kataglyphis-RustProjectTemplate/actions/workflows/rust_windows2025.yml/badge.svg)](https://github.com/Kataglyphis/Kataglyphis-RustProjectTemplate/actions/workflows/rust_windows2025.yml)

For **__official docs__** follow this [link](https://rust.jonasheinle.de).

<!-- [![Linux build](https://github.com/Kataglyphis/GraphicsEngineVulkan/actions/workflows/Linux.yml/badge.svg)](https://github.com/Kataglyphis/GraphicsEngineVulkan/actions/workflows/Linux.yml)
[![Windows build](https://github.com/Kataglyphis/GraphicsEngineVulkan/actions/workflows/Windows.yml/badge.svg)](https://github.com/Kataglyphis/GraphicsEngineVulkan/actions/workflows/Windows.yml)
[![TopLang](https://img.shields.io/github/languages/top/Kataglyphis/GraphicsEngineVulkan)]() -->
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/paypalme/JonasHeinle)
[![Twitter](https://img.shields.io/twitter/follow/Cataglyphis_?style=social)](https://twitter.com/Cataglyphis_)
 
## Table of Contents

- [About The Project](#about-the-project)
  - [Key Features](#key-features)
  - [Dependencies](#dependencies)
  - [Useful tools](#useful-tools)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Tests](#tests)
- [Run](#run)
- [Docs](#docs)
- [Updates](#updates)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Literature](#literature)

## About The Project

This project is about collecting experience in Rust.

### Key Features

- Features are to be adjusted to your own project needs.

<div align="center">


|            Category           |           Feature                             |  Implement Status  |
|-------------------------------|-----------------------------------------------|:------------------:|
|  **Packaging agnostic**   | Binary only deployment                            |         ✔️         |
|                               | Lore ipsum                                   |         ✔️         |
|  **Lore ipsum agnostic**   |                                               |                    |
|                               | LORE IPSUM                            |         ✔️         |
|                               |
|                               | Advanced unit testing                         |         🔶         |
|                               | Advanced performance testing                  |         🔶         |
|                               | Advanced fuzz testing                         |         🔶         |

</div>

**Legend:**
- ✔️ - completed  
- 🔶 - in progress  
- ❌ - not started

### Dependencies
This enumeration also includes submodules.
<!-- * [Vulkan 1.3](https://www.vulkan.org/) -->

If you just want the newest versions allowed by your current constraints (updates Cargo.lock only):

Update all:
```bash
cargo update

Update versions in Cargo.toml
cargo install cargo-edit
cargo upgrade --dry-run --verbose
cargo upgrade --incompatible
```

### Useful tools

* [cargo-outdated](https://github.com/kbknapp/cargo-outdated)
<!-- * [cppcheck](https://cppcheck.sourceforge.io/) -->

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

### Installation

1. Clone the repo
   ```bash
   git clone --recurse-submodules git@github.com:Kataglyphis/Kataglyphis-RustProjectTemplate.git
   ```
 
## Tests

<!-- ROADMAP -->
## Run
```bash
cargo run -- read --path ../README.md
```

### Windows: GStreamer + ONNX Overlay (WGPU)

Build + Run (CPU via tract):

```bash
cargo run --bin kataglyphis_rustprojecttemplate --features gui_windows,onnx_tract -- gui --backend dx12
```

Build + Run (ONNX Runtime + DirectML):

```bash
cargo run --bin kataglyphis_rustprojecttemplate --features gui_windows,onnxruntime_directml -- gui --backend dx12
```

Build + Run (ONNX Runtime + CUDA, NVIDIA):

```bash
# PowerShell
$env:KATAGLYPHIS_ORT_DEVICE="cuda"
cargo run --bin kataglyphis_rustprojecttemplate --features gui_windows,onnxruntime_cuda -- gui --backend dx12

# CMD
set KATAGLYPHIS_ORT_DEVICE=cuda
cargo run --bin kataglyphis_rustprojecttemplate --features gui_windows,onnxruntime_cuda -- gui --backend dx12
```

Optional environment variables:

- `KATAGLYPHIS_ONNX_MODEL` – Pfad zum ONNX-Modell (Default: models/yolov10m.onnx)
- `KATAGLYPHIS_ONNX_BACKEND` – `tract` oder `ort` (Default: automatisch)
- `KATAGLYPHIS_ORT_DEVICE` – `cpu` | `auto` | `cuda` (Default: `cpu`)
- `KATAGLYPHIS_PREPROCESS` – `letterbox` | `stretch` (Default: `stretch`)
- `KATAGLYPHIS_SWAP_XY` – setze `1`, falls die Modell-Ausgabe X/Y vertauscht (Default: `0`)
- `KATAGLYPHIS_SCORE_THRESHOLD` – Score-Schwelle für Erkennung (Default: `0.5`)
- `KATAGLYPHIS_INFER_EVERY_MS` – Inferenz-Intervall in ms (Default: `100`, `0` = jedes Frame)

CUDA Hinweise:
- Benötigt NVIDIA-Treiber + CUDA/cuDNN Runtime auf dem System.
- Wenn CUDA-Init fehlschlägt, kann `KATAGLYPHIS_ORT_DEVICE=auto` genutzt werden (fällt auf CPU zurück).

Overlay:
- Zeigt FPS, Inferenz-Latenz, CPU/RSS und eine CPU-Historie.
- Inferenz kann im Overlay ein-/ausgeschaltet werden.

### Resource usage logging (CPU/GPU/RAM)

```bash
cargo run --features gui_windows,onnxruntime_directml -- --resource-log --resource-log-interval-ms 1000 --resource-log-gpu=true gui
```

Optional: zusätzlich in Datei schreiben

```bash
cargo run --features gui_windows,onnxruntime_directml -- --resource-log --resource-log-file .\resource.log gui
```

### Burn / PyTorch-Replacement Demos

Diese Demos sind als separates Binary integriert und per Feature gated.

```bash
cargo run --features burn_demos --bin burn-demos -- --help
```

Beispiele:

```bash
cargo run --features burn_demos --bin burn-demos -- tensor-demo

cargo run --features burn_demos --bin burn-demos -- linear-regression --epochs 50 --steps-per-epoch 50 --lr 0.02 --batch-size 256

cargo run --features burn_demos --bin burn-demos -- xor --epochs 2000 --lr 0.05

cargo run --features burn_demos --bin burn-demos -- two-moons --epochs 200 --steps-per-epoch 50 --lr 0.01 --batch-size 256

# ONNX Runtime YOLOv10m Demo (Default model: models/yolov10m.onnx)
cargo run --features burn_demos --bin burn-demos -- onnx-yolov10 --runs 1 --print-topk 3
```

### Windows
```bash
cargo run --features gui_windows -- gui --backend dx12

# Vulkan backend
cargo run --features gui_windows -- gui --backend vulkan

# Auto-select (wgpu PRIMARY)
cargo run --features gui_windows -- gui --backend primary
```

### Linux
```bash
# WGPU (recommended)
cargo run --features gui_linux -- gui --backend vulkan

# GTK demo
cargo run --features gui_unix -- gui
```

## Docs
```bash
cargo doc --open
```
## Updates

How to update all installed packages:

1. Install updater:
  ```bash
  cargo install cargo-update
  ```
2. Now update all packages:
  ```bash
  cargo install-update -a
  ```

## Cameras

```bash
sudo v4l2-ctl --list-formats-ext -d /dev/video0
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
gst-launch-1.0 videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! autovideosink
```


## Roadmap
Upcoming :)
<!-- See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues). -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

<!-- CONTACT -->
## Contact

Jonas Heinle - [@Cataglyphis_](https://twitter.com/Cataglyphis_) - jonasheinle@googlemail.com

Project Link: [https://github.com/Kataglyphis/...](https://github.com/Kataglyphis/...)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

<!-- Thanks for free 3D Models: 
* [Morgan McGuire, Computer Graphics Archive, July 2017 (https://casual-effects.com/data)](http://casual-effects.com/data/)
* [Viking room](https://sketchfab.com/3d-models/viking-room-a49f1b8e4f5c4ecf9e1fe7d81915ad38) -->

## Literature 

Some very helpful literature, tutorials, etc. 

<!-- CMake/C++
* [Cpp best practices](https://github.com/cpp-best-practices/cppbestpractices)

Vulkan
* [Udemy course by Ben Cook](https://www.udemy.com/share/102M903@JMHgpMsdMW336k2s5Ftz9FMx769wYAEQ7p6GMAPBsFuVUbWRgq7k2uY6qBCG6UWNPQ==/)
* [Vulkan Tutorial](https://vulkan-tutorial.com/)
* [Vulkan Raytracing Tutorial](https://developer.nvidia.com/rtx/raytracing/vkray)
* [Vulkan Tutorial; especially chapter about integrating imgui](https://frguthmann.github.io/posts/vulkan_imgui/)
* [NVidia Raytracing tutorial with Vulkan](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
* [Blog from Sascha Willems](https://www.saschawillems.de/)

Physically Based Shading
* [Advanced Global Illumination by Dutre, Bala, Bekaert](https://www.oreilly.com/library/view/advanced-global-illumination/9781439864951/)
* [The Bible: PBR book](https://pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models)
* [Real shading in Unreal engine 4](https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf)
* [Physically Based Shading at Disney](https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf)
* [RealTimeRendering](https://www.realtimerendering.com/)
* [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs](https://hal.inria.fr/hal-01024289/)
* [Sampling the GGX Distribution of Visible Normals](https://pdfs.semanticscholar.org/63bc/928467d760605cdbf77a25bb7c3ad957e40e.pdf)

Path tracing
* [NVIDIA Path tracing Tutorial](https://github.com/nvpro-samples/vk_mini_path_tracer/blob/main/vk_mini_path_tracer/main.cpp) -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->