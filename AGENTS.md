# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project layout

Cargo workspace (`Cargo.toml` at the root is both the workspace and the root package `kataglyphis_rustprojecttemplate` — a lib with `cdylib`/`staticlib`/`rlib` crate types plus the feature-gated `burn-demos` bin):

- `crates/core` — core config/detection/logging (`kataglyphis_core`)
- `crates/telemetry` — resource monitoring (`kataglyphis_telemetry`; has the unit tests)
- `crates/inference` — ONNX backends, feature-gated (`onnx_tract`, `onnxruntime`, `onnxruntime_directml`, `onnxruntime_cuda`)
- `crates/gui` — feature-gated GUI (`gui_windows`, `gui_linux`, `gui_wgpu`, `gui_unix`)
- `crates/cli` — the CLI binary; its bin target is named `kataglyphis_rustprojecttemplate` (read/stats/gui subcommands; `stats --path <file>`)
- `tests/` — root-package integration tests (`integration.rs`) and proptest fuzz tests (`fuzz_test.rs`)
- `ExternalLib/Kataglyphis-ContainerHub` — git submodule; canonical container/toolchain docs live there (`docs/windows-builds.md`). Do not edit it from this repo.

## Build & test (host)

```bash
cargo build --workspace --locked                      # dev/debug
cargo build --workspace --locked --profile profile    # custom: release + debuginfo
cargo build --workspace --locked --release            # fat LTO, codegen-units 1, panic=abort, stripped
cargo test  --workspace --locked                      # unit + integration + proptest fuzz + doc tests
```

Default features are empty — GUI and ONNX code only compiles with explicit `--features` (see README "Run"). "Fuzz" testing = proptest in `tests/fuzz_test.rs`; there is no cargo-fuzz/libFuzzer target.

Known-benign warning: cargo reports a pdb/filename collision because the root **lib** and the CLI **bin** are both named `kataglyphis_rustprojecttemplate`. It may become a hard cargo error someday; renaming one target is the fix.

## Build & test in the Stevedore Windows container

Driver: `Scripts\Windows\Container\Invoke-StevedoreBuild.ps1` (add `-Test` to also run the test suite; `-TestOnly` to skip building). It stages sources off the Dev Drive, runs the in-container scripts (`rust-build-all.ps1`, `rust-test-all.ps1`) in `ghcr.io/kataglyphis/kataglyphis_beschleuniger:winamd64`, and copies artifacts back to `target\container\<profile>` plus gitignored root mirrors `debug\`, `profile\`, `release\`.

Hard-won host facts (verified 2026-07-17; full background in the submodule's `docs/windows-builds.md`):

- **Use Stevedore's `docker.exe`** (`D:\Stevedore\bin\docker.exe` or `%ProgramFiles%\Stevedore\bin\docker.exe`), never nerdctl.
- **Dev Drive (ReFS) volumes refuse bind mounts** with *"Der Dateisystem-Minifilter kann nicht an das Entwicklervolume angefügt werden"* unless `fsutil devdrv setfiltersallowed bindFlt, wcifs` has been run (elevated) and the volume remounted. The driver sidesteps this by staging to `%LOCALAPPDATA%\Temp`.
- **Run containers with `--isolation process`** for the full host CPU count; Hyper-V isolation exposes only 2 CPUs. Mount targets must be paths that do not already exist in the image (e.g. `C:\ws-mnt`).
- **Keep every build write container-local** (`CARGO_TARGET_DIR=C:\ct`, `CARGO_HOME=C:\ch`): the wcifs/bindFlt skew on this host breaks create-then-rename in image-layer dirs and two-path ops on bind mounts. Plain copies through the mount work; renames/moves may not. `docker cp` is unreliable — persist results via the mount.
- **A dying docker CLI is not a dying build**: the client pipe intermittently drops (transient hcsshim/ttrpc flakiness) while the container keeps running. Check `docker inspect` container state before concluding failure; run containers **named and without `--rm`** so logs and state survive.
- In-container scripts run under Windows PowerShell 5.1: keep `$ErrorActionPreference` at `Continue` and check `$LASTEXITCODE` manually (native stderr under `Stop` becomes terminating errors), and tee important output to the mounted scratch dir so a dropped client can't lose it.

## Verified baselines (2026-07-17, container, 32 CPUs / 48 GB)

- Builds: debug 1m11s, profile 1m31s, release 1m08s — all green; binaries verified on the host (`stats --path README.md`).
- Tests (debug): **8 passed / 0 failed** — 3 integration, 1 proptest fuzz case, 4 telemetry unit; 1 doc-test ignored.

## Conventions

- Version pins/single sources of truth follow the ContainerHub ecosystem; don't duplicate what the submodule documents — link to it.
- Never commit build outputs: `/target`, root `/debug`, `/profile`, `/release` are gitignored.
