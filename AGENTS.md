# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project layout

Cargo workspace (`Cargo.toml` at the root is both the workspace and the root package `kataglyphis_rustprojecttemplate` — a lib with `cdylib`/`staticlib`/`rlib` crate types plus the feature-gated `burn-demos` bin):

- `crates/core` — core config/detection/logging (`kataglyphis_core`)
- `crates/telemetry` — resource monitoring (`kataglyphis_telemetry`; has the unit tests)
- `crates/inference` — ONNX backends, feature-gated (`onnx_tract`, `onnxruntime`, `onnxruntime_directml`, `onnxruntime_cuda`)
- `crates/gui` — feature-gated GUI (`gui_windows`, `gui_linux`, `gui_wgpu`, `gui_unix`)
- `crates/webgpu_renderer` - WebGPU (wgpu) glTF renderer, native + wasm32/browser (`kataglyphis_webgpu_renderer`): PBR, cascaded shadows, SSAO, bloom, skinning, animations, LOD
- `crates/cli` — the CLI binary; its bin target is named `kataglyphis_rustprojecttemplate` (read/stats/gui subcommands; `stats --path <file>`)
- `tests/` — root-package integration tests (`integration.rs`) and proptest fuzz tests (`fuzz_test.rs`)
- `ExternalLib/Kataglyphis-ContainerHub` — git submodule and **the ground truth for every container and PowerShell concern**. See the section below before writing any helper.

## ContainerHub is the ground truth

Anything to do with containers, Dockerfiles, CI plumbing or PowerShell belongs to the submodule. **Search it before writing a helper.** Everything below was written locally first and later found to already exist there — usually in a better form, twice with a bug the local copy did not have:

| Need | Use | Not |
| --- | --- | --- |
| `docker.exe` discovery (Stevedore) | `Resolve-DockerExe` | a hand-rolled candidate list |
| `--isolation process` and friends | `Get-ContainerIsolationArgs` | inline flags |
| Container teardown | `Remove-BuildContainerSafe` | `docker rm -f` (misses the wcifs teardown lock) |
| Bind-mount probe, artifact delivery | `Test-ContainerBindMount`, `Test-BuildArtifactsDelivered` | assuming a green build delivered something |
| SDK tools (makeappx, signtool) | `Resolve-WindowsSdkToolPath` | `Get-ChildItem -Recurse` over the Kits tree |
| MSIX manifest tokens | `Expand-XmlTemplateTokens` | `-replace` — see below |
| XML escaping, placeholder PNGs | `ConvertTo-XmlEscapedText`, `New-TransparentPng` | local redefinitions |
| Config access | `Get-OrDefault`, `Get-ConfigValue` (`WindowsConfig.Common`) | copies |
| Build logging and steps | `New-BuildContext`, `Invoke-BuildStep`, `Invoke-BuildExternal`, `Write-BuildLog*` | ad-hoc `Write-Host` wrappers |
| Version normalising | `ConvertTo-NormalizedVersion` (pwsh), `version_util.sh --normalize` (bash) | a second implementation |
| CI job plumbing | the composite actions under `.github/actions/` | hand-written `docker run` blocks |
| In-container cargo steps | `linux/scripts/02-toolchain/rust/cargo_*.sh` | inline cargo invocations |

**Never expand a manifest template with `-replace`.** PowerShell treats the replacement side as a substitution template, so a value containing `$&` re-inserts the whole matched token. A description of ``Renderer $& x`` produced `Desc="Renderer __MSIX_DESCRIPTION__amp; x"` — the literal token, shipped into the manifest. `Expand-XmlTemplateTokens` uses an ordinal `[string].Replace` and escapes each value itself.

Two caveats:

- **Nested module imports are module-private.** `WindowsBuild.Common` importing `WindowsScripts.Shared` does not re-export it to you; import each module you call into directly, or you get a "command not found" the first time that code path runs.
- **Editing the submodule is allowed** (it is the same owner), but it is consumed by other repos. Change it there, push, then move this repo's submodule pointer — do not fork behaviour locally.

Nothing here needs Windows PowerShell 5.1 semantics: every script carries `#requires -Version 7.0` and CI invokes `pwsh`.

## Build & test (host)

```bash
cargo build --workspace --locked                      # dev/debug
cargo build --workspace --locked --profile profile    # custom: release + debuginfo
cargo build --workspace --locked --release            # fat LTO, codegen-units 1, panic=abort, stripped
cargo test  --workspace --locked                      # unit + integration + proptest fuzz + doc tests
```

Run the lint gate before pushing — CI runs exactly these two commands and both are hard failures:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Default features are empty — GUI and ONNX code only compiles with explicit `--features` (see README "Run"). "Fuzz" testing = proptest in `tests/fuzz_test.rs`; there is no cargo-fuzz/libFuzzer target.

Known-benign warning: cargo reports a pdb/filename collision because the root **lib** and the CLI **bin** are both named `kataglyphis_rustprojecttemplate`. It may become a hard cargo error someday; renaming one target is the fix.

## Build & test in the Stevedore Windows container

Driver: `Scripts\Windows\Container\Invoke-StevedoreBuild.ps1` (add `-Test` to also run the test suite; `-TestOnly` to skip building). It stages sources off the Dev Drive, runs the in-container scripts (`rust-build-all.ps1`, `rust-test-all.ps1`) in `ghcr.io/kataglyphis/kataglyphis_beschleuniger:winamd64`, and copies artifacts back to `target\container\<profile>` plus gitignored root mirrors `debug\`, `profile\`, `release\`.

Hard-won host facts (verified 2026-07-17; full background in the submodule's `docs/windows-builds.md`):

- **Use Stevedore's `docker.exe`, never nerdctl** — but do not hard-code the path: `Resolve-DockerExe` already checks `$env:DOCKER_EXE`, both Stevedore locations and then PATH. nerdctl is excluded deliberately; it talks to containerd, not this lane's engine.
- **Dev Drive (ReFS) volumes refuse bind mounts** with *"Der Dateisystem-Minifilter kann nicht an das Entwicklervolume angefügt werden"* unless `fsutil devdrv setfiltersallowed bindFlt, wcifs` has been run (elevated) and the volume remounted. The driver sidesteps this by staging to `%LOCALAPPDATA%\Temp`.
- **Run containers with `--isolation process`** for the full host CPU count; Hyper-V isolation exposes only 2 CPUs. Get the flags from `Get-ContainerIsolationArgs` rather than inline. Mount targets must be paths that do not already exist in the image (e.g. `C:\ws-mnt`).
- **Keep every build write container-local** (`CARGO_TARGET_DIR=C:\ct`, `CARGO_HOME=C:\ch`): the wcifs/bindFlt skew on this host breaks create-then-rename in image-layer dirs and two-path ops on bind mounts. Plain copies through the mount work; renames/moves may not. `docker cp` is unreliable — persist results via the mount.
- **A dying docker CLI is not a dying build**: the client pipe intermittently drops (transient hcsshim/ttrpc flakiness) while the container keeps running. Check `docker inspect` container state before concluding failure; run containers **named and without `--rm`** so logs and state survive. Tear them down with `Remove-BuildContainerSafe` — a bare `docker rm -f` can return while the wcifs teardown still holds the name, and the next run then fails on the clash.
- **Everything here is `pwsh` (PowerShell 7+).** Every script under `Scripts/Windows/` carries `#requires -Version 7.0` and CI invokes `pwsh`, so any surviving "Windows PowerShell 5.1" comment is wrong — those scripts would not start under it. Keep `$ErrorActionPreference` at `Continue` in the in-container scripts and check `$LASTEXITCODE` manually anyway (native-command stderr handling has shifted across PowerShell versions; the explicit check has not), and tee important output to the mounted scratch dir so a dropped docker client cannot lose it.
- **Still not adopted:** `Test-BuildArtifactsDelivered` (a green build is not proof of delivery — it `docker exec`s the container, so it must run before teardown) and `Test-ContainerBindMount`. Both would fit `Invoke-StevedoreBuild.ps1`; neither could be exercised from the Linux verification box.

## Verified baselines (container, 32 CPUs)

**2026-08-07, winamd64, rustc 1.97.1** — `Invoke-StevedoreBuild.ps1 -MemoryGb 32`:

- Builds: debug 1m35s, profile 1m32s, release 1m12s — all three green. Release binary verified on the host: `stats --path README.md` → `Lines: 476, Words: 1905, Bytes: 20104`.
- Tests: the 8 that predate `crates/webgpu_renderer` still pass (3 integration, 1 proptest, 4 telemetry). **`kataglyphis_webgpu_renderer --lib` cannot start in this container**: `exit code 0xc0000135, STATUS_DLL_NOT_FOUND`, before `main`, so no test runs.

  Not a regression from the wgpu 30 upgrade. The old "8 passed / 0 failed" baseline was recorded on 2026-07-17, and the renderer crate landed on 2026-07-18 — the container test lane has therefore *never* run with that crate present. The image is Server Core with no GPU stack; a wgpu-linked binary needs graphics DLLs it does not ship.

  Consequence: `Invoke-StevedoreBuild.ps1 -Test` fails as a whole. Use `-TestOnly` knowing the renderer will abort, or scope it (`cargo test --workspace --exclude kataglyphis_webgpu_renderer`), until the image carries the missing DLLs.

**2026-07-17** (superseded, kept because it is what the 8-test figure refers to): builds debug 1m11s / profile 1m31s / release 1m08s; tests 8 passed / 0 failed, 1 doc-test ignored — measured before `crates/webgpu_renderer` existed.

## Continuous integration

Two workflows, both building inside ContainerHub images rather than on the runner:

| Lane | Workflow | Runs when | Image |
| --- | --- | --- | --- |
| Linux x86_64 | `rust_ubuntu24_04.yml` | every push/PR to `main`/`develop` | `ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest-cross` |
| Linux arm64 | same | opt-in: `[build-arm]` in the HEAD commit message, or `workflow_dispatch` | same |
| Windows | `rust_windows2025.yml` | opt-in: `[build-win]` in the HEAD commit message, or `workflow_dispatch` | `…:winamd64` |

**A green tick without the opt-in marker says nothing about that lane** — the workflow reports `skipped`, which the badge renders the same as passing.

Facts that cost real debugging time:

- **The ARM lane cannot currently go green.** `:latest-cross` resolves to an amd64-only manifest list, so the pull dies with `no matching manifest for linux/arm64/v8` before any build step. Repair is a ContainerHub-side job (`build-runtime-manifest.sh --repair --push-manifest`); until then, leave `[build-arm]` off.
- **Never call ContainerHub's `cargo_fmt_clippy.sh` from a workflow.** Its first line is `rustup component add rustfmt`, and the runtime image deliberately ships **no rustup** (rustfmt/clippy are baked in at image-build time) — so it exits 127 before cargo ever runs. Invoke `cargo fmt`/`cargo clippy` directly. This was masked by `continue-on-error: true` for months and let an entire crate reach `main` unformatted and with 12 clippy errors.
- **The container runs as uid 1001, not root.** `apt-get` fails with `Permission denied`, so a workflow step cannot install system packages — whatever the image lacks, it lacks. And `CARGO_HOME=/usr/local/cargo` is root-owned, so every writing cargo step needs `-e CARGO_HOME=/tmp/cargo-home`.
- **The lint gate runs default features on purpose.** `--all-features` would need GTK4 headers and the ORT libs, which the image has not got and uid 1001 cannot install.

Lint the workflows locally with the submodule's pinned, SHA-verified actionlint (works from Git Bash on Windows):

```bash
bash ExternalLib/Kataglyphis-ContainerHub/linux/scripts/lint-workflows.sh .
```

The trailing `.` is load-bearing: without it the script lints ContainerHub's own workflows instead of this repo's, and reports green either way.

## Verifying locally on a Windows box (no MSVC required)

This repo's dev machines commonly lack the MSVC "C++ build tools" workload. Without it **nothing links**, and Git Bash makes the failure baffling: `/usr/bin/link.exe` is coreutils' `link`, which shadows the MSVC linker on PATH and dies with `link: missing operand after '\377\376'` (it is being handed rustc's UTF-16 response file). That is an environment fault, never a code fault.

The fastest real fix is to verify in WSL against the CI's own target platform:

```bash
# once, as root inside the distro
apt-get install -y build-essential pkg-config curl git libssl-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk-4-dev
# optional: a software Vulkan adapter so the headless golden tests actually run
apt-get install -y mesa-vulkan-drivers vulkan-tools
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/ri.sh
sh /tmp/ri.sh -y --profile minimal --default-toolchain 1.97.1 -c rustfmt -c clippy
```

Then, from the repo root, with `CARGO_TARGET_DIR` pointed at a **Linux-native** path (never the 9p-mounted Windows tree, which is glacial and already holds MSVC artifacts):

```bash
export CARGO_TARGET_DIR=/root/kt
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
KATAGLYPHIS_REQUIRE_GPU=1 cargo test --workspace --locked
```

`KATAGLYPHIS_REQUIRE_GPU` is the important one: without it `GpuContext::headless_or_skip()` silently returns `None` and the whole golden-test suite "passes" having rendered nothing. Set it and a missing adapter becomes a panic, so green *proves* the tests ran.

## Feature combinations and their system dependencies

Default features are empty, so `cargo build` needs nothing. Each optional feature pulls system libraries that must already exist — **the CI container runs as uid 1001 and cannot `apt-get install` them**:

| Feature | Needs | In the CI image? |
| --- | --- | --- |
| `gstreamer` (crates/media) | GStreamer dev files | **Yes** — source-built into `/opt/gstreamer`, on `PKG_CONFIG_PATH`. Do *not* install the distro `libgstreamer*-dev`: the image purges those on purpose. |
| `gui_linux` | GStreamer + wgpu (pure Rust) | **Yes** — despite the name it does not use GTK; it is the wgpu path. |
| `gui_unix` | `libgtk-4-dev` | **No, by design.** The foreign-arch GTK dev chain pulls target-side Python and breaks cross builds on `python3-minimal`'s postinst. This feature cannot be built against `latest-cross`. |
| `onnxruntime`, `burn_demos` | `libssl-dev` (via `openssl-sys`) | **Yes** — via ContainerHub's `package-lists.sh`. |

On a plain Ubuntu box (e.g. the WSL recipe above) you *do* need the distro packages, because nothing there provides the source-built stack. That difference is exactly why "install the -dev package" is the wrong instinct when the image is involved.

## Known gaps

- **No CI lane builds any optional feature.** The Linux lane builds default features; the Windows lane builds `gui_windows,onnxruntime_directml` and is itself opt-in. So `crates/media` and the burn demos have no automated coverage — that is how the GStreamer version skew (since fixed) survived unnoticed. The `feature-matrix` job in `rust_ubuntu24_04.yml` closes this, but only once the build image ships the three package groups in the table above.

- **No CI lane has a GPU, so the golden tests never actually run.** `GpuContext::headless_or_skip()` returns `None` and every one of the ~40 headless render tests reports as passed having drawn nothing. This is not theoretical: running them for real (WSL + llvmpipe, 2026-08-07) surfaced a **pre-existing, deterministic rendering bug**:

  ```
  a_non_uniform_instance_scale_shades_like_the_same_node_scale
  crates/webgpu_renderer/tests/skinned_bounds.rs
  node-scale and instance-scale must shade the same, 987 pixels differ  (threshold: 40)
  ```

  Confirmed pre-existing by re-running it against a pristine export of `a618287`: byte-identical 987. Everything else in the suite (~340 tests) passes. The failing path is the instanced normal transform — the generated `src/shaders/forward.wgsl` applies `instance_cofactor_0` to `worldNormal_0`, and the two shading paths disagree where they must agree.

  **Fix it upstream, not here.** The `.wgsl` files in `src/shaders/` are checked-in *generated artifacts*; there is no `.slang` file in this repo, and the code comments reference the C++ engine's `forward.slang` by line number (e.g. `cascades.rs` → `forward.slang:151`). Hand-editing the generated WGSL would desynchronise it from its source.

  Until a CI runner has an adapter, a software one makes these tests real: install `mesa-vulkan-drivers` and set `KATAGLYPHIS_REQUIRE_GPU=1` so a missing adapter fails loudly instead of skipping silently.

## The 2026-08-07 graphics-stack upgrade

`wgpu` 29→30, `naga` 29→30, `egui`/`egui-wgpu`/`egui-winit` 0.35→0.36, `glam` 0.30→0.33 and `pollster` 0.4→1.0 moved as one coupled set (`egui-wgpu` 0.35 pins `wgpu ^29`, 0.36 pins `^30`, so none of them could move alone). What changed, so the next person does not have to rediscover it:

- **`VertexState::buffers` is `&[Option<VertexBufferLayout>]`** — a slot can now be left unbound without shifting the ones after it.
- **`BufferSlice::get_mapped_range` returns `Result`.** Seven call sites. Only `render_to_pixels_with_format` returns `Result` and propagates; the other six are in functions whose caller has already awaited the map, so they `expect` with a message naming that invariant.
- **Presentation moved from `SurfaceTexture::present(self)` to `Queue::present(&self, texture)`.** Three sites, two of which the default Linux build never compiles (one is `cfg(wasm32)`, one is behind a GUI feature) — check them by hand or with the `feature-matrix` job.
- **`RequestAdapterOptions::apply_limit_buckets`** (new, no default): rounds reported adapter limits to coarse presets so a host exposing wgpu to *untrusted* content cannot fingerprint the machine. This renderer is the trusted application, so it is `false` — real limits, as wgpu 29 had.
- **`SurfaceConfiguration::color_space`** (new, no default): set to `SurfaceColorSpace::Auto`, the type's own default and the only value guaranteed supported for every format in `SurfaceCapabilities::formats`. Anything else (an HDR space) needs a capability check first.
- **glam 0.33 moved the camera constructors off `Mat4`** and split them by clip-space convention: `opengl` (NDC Z −1..1), `directx` (Z 0..1, Y up), `vulkan` (Z 0..1, Y down). **`directx` is the one that matches** — it reproduces the old `Mat4::perspective_rh`/`orthographic_rh`/`perspective_infinite_rh` bit for bit. That was verified by compiling both against glam 0.33.3 and comparing the matrices, not inferred from the names: the `vulkan` module is Y-**down**, and picking it would have flipped the image with no compile error. The old methods are deprecated but still present, so `-D warnings` is what forces the migration.

Guard this with the golden tests, not with the compiler: a clip-space or Y-axis mistake compiles perfectly and only shows up in pixels. See the GPU note above for how to make them actually run.

The upgrade was verified rendering-neutral: 333 tests pass, the only failure is the pre-existing instance-normal bug, and it still reports **exactly 987 differing pixels** — the same figure as before the upgrade. That number is the useful signal here; a changed clip-space or flipped Y would have moved it.

## Conventions

- Version pins/single sources of truth follow the ContainerHub ecosystem; don't duplicate what the submodule documents — link to it.
- Never commit build outputs: `/target`, root `/debug`, `/profile`, `/release` are gitignored.
