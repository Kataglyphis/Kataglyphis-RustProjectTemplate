# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project layout

Cargo workspace (`Cargo.toml` at the root is both the workspace and the root package `oxidant` — a lib with `cdylib`/`staticlib`/`rlib` crate types plus the feature-gated `burn-demos` bin):

- `crates/core` — core config/detection/logging (`kataglyphis_core`)
- `crates/telemetry` — resource monitoring (`kataglyphis_telemetry`; has the unit tests)
- `crates/inference` — ONNX backends, feature-gated (`onnx_tract`, `onnxruntime`, `onnxruntime_directml`, `onnxruntime_cuda`)
- `crates/gui` — feature-gated GUI (`gui_windows`, `gui_linux`, `gui_wgpu`, `gui_unix`)
- `crates/webgpu_renderer` - WebGPU (wgpu) glTF renderer, native + wasm32/browser (`kataglyphis_webgpu_renderer`): PBR, cascaded shadows, SSAO, bloom, skinning, animations, LOD
- `crates/cli` — the CLI binary; its bin target is named `kataglyphis_cli` (read/stats/gui subcommands; `stats --path <file>`). It was renamed from `oxidant` on 2026-08-07 — see the pdb note below.
- `tests/` — root-package integration tests (`integration.rs`) and proptest fuzz tests (`fuzz_test.rs`)
- `third_party/ContainerHub` — git submodule and **the ground truth for every container and PowerShell concern**. See the section below before writing any helper.

## ContainerHub is the ground truth

Anything to do with containers, Dockerfiles, CI plumbing or PowerShell belongs to the submodule. **Search it before writing a helper.**

**Do not re-derive host knowledge here — read it there.** Everything about
Stevedore, Rancher Desktop, wcifs/bindFlt and the container hosts is already
written down, in more depth than this file should carry. Start at
[`third_party/ContainerHub/docs/INDEX.md`](third_party/ContainerHub/docs/INDEX.md)
— it maps topic → owning document, so one hop survives upstream reorganisation.

The entries this repo reaches for most:

| Question | Document |
| --- | --- |
| How do I set up a Windows host for Stevedore? (services, `docker-users`, CNI nat conf, pwsh, gates) | `docs/windows-host-setup.md` |
| Why does a layer commit fail / what is process isolation doing? wcifs, bindFlt, the `ActivateLayer 0x20` bug | `docs/windows-builds.md` |
| Dev Drive filter setup, bind mount vs tar-pipe, container reuse | `docs/windows-container-build-performance.md` |
| How do I run Linux containers on Windows? | `docs/rancher-desktop-linux-containers.md` |
| Which image, which tag, which engine? | `docs/adopting-in-a-new-project.md` |
| Why did my lane not run? | `docs/ci-build-triggers.md` |

When something in this file contradicts one of those, **the submodule wins**.
That has happened twice, both times because a procedure was retyped here instead
of linked: this file once claimed Dev Drive volumes *refuse* bind mounts (they
work; it is create-then-rename that fails), and it carried a `fsutil devdrv`
command that was missing `/volume` and split its filter list on a space — so it
could never have worked. Both are why § *Build & test in the Stevedore Windows
container* now links rather than restates.

## Linux containers locally (Rancher Desktop)

Read `docs/rancher-desktop-linux-containers.md` first. The essentials as they apply here: the image is **always** `:latest-cross`, Rancher defaults to the **containerd** engine so it is `nerdctl --namespace default` rather than `docker`, and from Git Bash `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'` is mandatory or the mount argument is mangled. Point `CARGO_HOME` at a writable path (the image's is root-owned); a named volume keeps the registry warm between runs.

Two consumer-specific traps, both hit on 2026-08-07:

- **A CRLF checkout breaks it before anything runs.** The scripts are executed by bash inside the container; a `\r` makes it fail with `set: pipefail\r: invalid option name`, which names neither the file nor line endings. `.gitattributes` now pins `*.sh` to LF in both repos, but git does not rewrite an existing checkout: `git ls-files -z '*.sh' | xargs -0 rm -f && git checkout -- .`
- **The image's Rust may be older than its own pin.** See "Known gaps" — `latest-cross` shipped Ubuntu's rustc 1.93.1 while `versions.env` pinned 1.97.1, which surfaced as a dependency's MSRV error, not as an image problem. Fixed in ContainerHub; check `rustc --version` in the container if a build fails on an MSRV floor. Everything below was written locally first and later found to already exist there — usually in a better form, twice with a bug the local copy did not have:

All paths below are relative to `third_party/ContainerHub/`.

| Need | Use | Defined in | Not |
| --- | --- | --- | --- |
| `docker.exe` discovery (Stevedore) | `Resolve-DockerExe` | [`windows/scripts/modules/WindowsContainerBuild.Reuse.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsContainerBuild.Reuse.psm1) | a hand-rolled candidate list |
| `--isolation process` and friends | `Get-ContainerIsolationArgs` | same file | inline flags |
| Container teardown | `Remove-BuildContainerSafe` | same file | `docker rm -f` (misses the wcifs teardown lock) |
| Bind-mount probe, artifact delivery | `Test-ContainerBindMount`, `Test-BuildArtifactsDelivered` | same file | assuming a green build delivered something |
| SDK tools (makeappx, signtool) | `Resolve-WindowsSdkToolPath` | [`windows/scripts/modules/WindowsMsix.Common.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsMsix.Common.psm1) | `Get-ChildItem -Recurse` over the Kits tree |
| MSIX manifest tokens | `Expand-XmlTemplateTokens` | same file | `-replace` — see below |
| XML escaping, placeholder PNGs | `ConvertTo-XmlEscapedText`, `New-TransparentPng` | same file | local redefinitions |
| Config access | `Get-OrDefault`, `Get-ConfigValue` | [`windows/scripts/modules/WindowsConfig.Common.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsConfig.Common.psm1) | copies |
| Build logging and steps | `New-BuildContext`, `Invoke-BuildStep`, `Invoke-BuildExternal`, `Write-BuildLog*` | [`windows/scripts/modules/WindowsBuild.Common.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsBuild.Common.psm1) | ad-hoc `Write-Host` wrappers |
| Tool guards, version normalising (pwsh) | `Assert-Command`, `ConvertTo-NormalizedVersion` | [`windows/scripts/modules/WindowsScripts.Shared.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsScripts.Shared.psm1) | a second implementation |
| Logging inside a container | `Start-ContainerLog`, `Write-ContainerLog`, `Invoke-ContainerLoggedCommand` | [`windows/scripts/modules/WindowsContainerLog.Common.psm1`](third_party/ContainerHub/windows/scripts/modules/WindowsContainerLog.Common.psm1) | a `Say`/`Run-Logged` pair per script |
| CI version stamping (bash) | `version_util.sh --github-env` / `--resolve-ci` / `--normalize` | [`linux/scripts/02-toolchain/rust/version_util.sh`](third_party/ContainerHub/linux/scripts/02-toolchain/rust/version_util.sh) | re-reading VERSION.txt yourself |
| In-container cargo steps | `cargo_debug.sh`, `cargo_release.sh`, `cargo_test.sh`, `cargo_coverage.sh`, … | [`linux/scripts/02-toolchain/rust/`](third_party/ContainerHub/linux/scripts/02-toolchain/rust) | inline cargo invocations |
| Linux packaging (tar/deb/AppImage/Flatpak) | `package_archive.sh` | [`linux/scripts/06-packaging/package_archive.sh`](third_party/ContainerHub/linux/scripts/06-packaging/package_archive.sh) | bespoke packaging |
| CI job plumbing | `prepare-linux-ci-host`, `run-in-linux-container`, `run-in-windows-container`, `clone-into-short-path`, `cleanup-disk-space`, `assert-docker-disk-space` | [`.github/actions/`](third_party/ContainerHub/.github/actions) | hand-written `docker run` blocks |
| Linting workflows locally | `lint-workflows.sh <root>` (pinned, SHA-verified actionlint) | [`linux/scripts/lint-workflows.sh`](third_party/ContainerHub/linux/scripts/lint-workflows.sh) | bootstrapping your own |
| Agentic loop | config + runner templates | [`shared/agentic-loop/templates/`](third_party/ContainerHub/shared/agentic-loop/templates) | writing one from scratch |
| Bash helpers (logging, retry, SHA'd downloads, parallelism) | `logging.sh`, `downloads.sh`, `parallelism.sh`, … | [`linux/scripts/01-core/`](third_party/ContainerHub/linux/scripts/01-core) | new implementations |

**One caveat about `cargo_fmt_clippy.sh`**: it is the one script in that rust directory this repo must *not* call — its first line is `rustup component add rustfmt`, and neither image can satisfy that offline. Call `cargo fmt` / `cargo clippy` directly. See the CI section.

**Never expand a manifest template with `-replace`.** PowerShell treats the replacement side as a substitution template, so a value containing `$&` re-inserts the whole matched token. A description of ``Renderer $& x`` produced `Desc="Renderer __MSIX_DESCRIPTION__amp; x"` — the literal token, shipped into the manifest. `Expand-XmlTemplateTokens` uses an ordinal `[string].Replace` and escapes each value itself.

Two caveats:

- **Nested module imports are module-private.** `WindowsBuild.Common` importing `WindowsScripts.Shared` does not re-export it to you; import each module you call into directly, or you get a "command not found" the first time that code path runs.
- **Editing the submodule is allowed** (it is the same owner), but it is consumed by other repos. Change it there, push, then move this repo's submodule pointer — do not fork behaviour locally.

Nothing here needs Windows PowerShell 5.1 semantics: every script carries `#requires -Version 7.0` and CI invokes `pwsh`.

## Where the documentation actually lives

This repo owns only `AGENTS.md`, `README.md`, `BACKLOG.md` and `crates/webgpu_renderer/README.md`. There is **no `docs/` directory here**, which matters because the renderer source refers to six design documents as if there were:

`docs/renderer-bounds-invariant.md`, `docs/gpu-golden-testing.md`, `docs/model-loading.md`, `docs/shader-sharing.md`, `docs/webgpu-gltf-rust-plan.md`, `docs/webgpu-srgb-audit.md`

They live in the **parent** repository, `Kataglyphis-BeschleunigerBallett/docs/` — the comments say "repo root" and mean the superproject's root, one level above this submodule. Worth knowing twice over: `bounds.rs` calls `renderer-bounds-invariant.md` the checklist for not repeating eight identical bugs, and if this template is ever used standalone those six references dangle with nothing to point at. From `crates/webgpu_renderer/` the correct relative prefix is `../../../../docs/` (four levels: crate → crates → repo → ExternalLib → superproject); `../../../` lands in `ExternalLib/` and was wrong in that README until 2026-08-07.

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

**The pdb collision is fixed — do not undo it by renaming the bin back.** Cargo used to warn that the root **lib** and the CLI **bin**, both named `oxidant`, wrote the same `oxidant.pdb` (it comes from the lib's `cdylib` crate type, not the rlib), and that this *"may become a hard error in the future"* — [rust-lang/cargo#6313](https://github.com/rust-lang/cargo/issues/6313).

The **bin** was renamed, not the lib, and that direction was deliberate: the C++ Vulkan engine imports the lib through Corrosion/cxxbridge (the generated `oxidant_bridge` target in the parent repo's CMake), so `[lib] name` decides DLL/LIB filenames that another repository depends on.

What moved with the bin: `Msix.Binary` and `Msi.OutputName` in `scripts/windows/Build-Windows.config.psd1`, `File Name=` in `wix/main.wxs`, `BINARY_FILE` in the Ubuntu workflow and `BINARY` in the Windows one, the `-Binary` default in `Run-AppProfiles.ps1`, and `--bin` in `scripts/linux/run-person-detection.sh`.

**`[lib] name` did change later, on 2026-09-05**, when the repository became OxidANT: `[package] name` and `[lib] name` are both `oxidant` now, so the artefacts are `oxidant.dll` / `liboxidant.so` / `liboxidant.a`. That is exactly the outside-this-repo break the paragraph above warns about, and it was only safe because every consumer was updated in the same commit — Inference-Engine's Cargokit wiring, podspecs, `Get-WindowsBuildConfig.ps1` and the committed `frb_generated.dart` loader stem, plus BeschleunigerBallett's Corrosion import. Renaming it again means finding those consumers again; they are not discoverable from inside this repository.

The two `BINARY` variables still mean different things. In the **Windows** workflow it is the executable (`kataglyphis_cli`). In the **Ubuntu** workflow it is `oxidant`, and it names *both* the tarball and the file inside it: `package_archive.sh` copies `target/release/$BINARY_FILE` to `$ArchiveDir/$Binary`. So `BINARY_FILE` is the cargo artefact, `BINARY` is what a user ends up invoking.

## Build & test in the Stevedore Windows container

Driver: `scripts\windows\Container\Invoke-StevedoreBuild.ps1` (add `-Test` to also run the test suite; `-TestOnly` to skip building). It **bind-mounts this repository straight into the container** as `C:\ws-mnt`, runs the in-container scripts (`rust-build-all.ps1`, `rust-test-all.ps1`) in `ghcr.io/kataglyphis/kataglyphis_beschleuniger:winamd64`, and the artifacts land directly in `target\container\<profile>`, mirrored to the gitignored root `debug\`, `profile\`, `release\`.

Mounting the repo — not a copy of it — is the default, ReFS Dev Drive or not.
It also means `third_party/` is present inside the container, so anything
importing ContainerHub modules (e.g. `Build-Windows.ps1`) works without special
staging.

**The host-side mechanics are upstream's, not this repo's.** Why writes through
a bind mount fail while reads succeed, how to allow the Dev Drive filters, what
`--isolation process` does to the CPU count, the wcifs teardown lock, the
transient hcsshim client-pipe drops, and why the Windows lane is Stevedore's
`docker.exe` rather than nerdctl: all in
[`docs/windows-builds.md`](third_party/ContainerHub/docs/windows-builds.md)
and
[`docs/windows-container-build-performance.md`](third_party/ContainerHub/docs/windows-container-build-performance.md).
Read those before changing the driver. **Do not copy their commands back into
this file** — the last copy of the `fsutil devdrv` line that lived here was
malformed and stayed that way through several edits.

What is specific to *this* repo, because cargo is what makes it bite:

- **Every build write stays container-local** — `CARGO_TARGET_DIR=C:\ct`,
  `CARGO_HOME=C:\ch`. Cargo's create-then-rename is exactly the pattern a bind
  mount rejects, so only a plain artifact copy crosses the mount at the end.
  That copy direction works; do not "simplify" it into a rename or a `docker cp`.
- **`-StageSources`** restores the old robocopy-to-`%LOCALAPPDATA%\Temp` path for
  a host whose Dev Drive filters were never allowed (symptom: `docker run` exits
  immediately with *"Der Dateisystem-Minifilter kann nicht an das
  Entwicklervolume angefügt werden"*). The permanent fix is a host setting — see
  *Dev Drive filter setup* in the performance doc above; `-StageSources` is the
  workaround, not the cure.
- **Mount target must not already exist in the image** — hence `C:\ws-mnt`.
- **Get container flags from the modules, never inline**: `Resolve-DockerExe`,
  `Get-ContainerIsolationArgs`, `Remove-BuildContainerSafe`. The last one matters
  most: a bare `docker rm -f` can return while teardown still holds the name, and
  the next run fails on the clash.
- **Run containers named and without `--rm`** so logs and state survive a dropped
  client, and tee important output to the mounted scratch dir.
- **Everything here is `pwsh` (PowerShell 7+).** Every script under
  `scripts/windows/` carries `#requires -Version 7.0`, so any surviving "Windows
  PowerShell 5.1" comment is wrong — those scripts would not start under it. Keep
  `$ErrorActionPreference` at `Continue` in the in-container scripts and check
  `$LASTEXITCODE` manually anyway.
- **Still not adopted:** `Test-BuildArtifactsDelivered` (a green build is not
  proof of delivery — it `docker exec`s the container, so it must run before
  teardown) and `Test-ContainerBindMount`. Both would fit
  `Invoke-StevedoreBuild.ps1`; neither could be exercised from the Linux
  verification box.

## Verified baselines (container, 32 CPUs)

**2026-08-07, winamd64, rustc 1.97.1** — `Invoke-StevedoreBuild.ps1 -MemoryGb 32`:

- Builds: debug 1m35s, profile 1m32s, release 1m12s — all three green. Release binary verified on the host: `stats --path README.md` → `Lines: 476, Words: 1905, Bytes: 20104`.
- Tests: the 8 that predate `crates/webgpu_renderer` still pass (3 integration, 1 proptest, 4 telemetry). **`kataglyphis_webgpu_renderer` is excluded from the container run** — `scripts/windows/container/rust-test-all.ps1` passes `--exclude kataglyphis_webgpu_renderer` and logs that it did. Its test binaries exit `0xc0000135` (`STATUS_DLL_NOT_FOUND`) before `main`, because linking wgpu with the `gles` backend makes the executable import `opengl32.dll`, which Server Core does not ship. The loader resolves that import, so no runtime flag helps; without the exclusion the whole `cargo test --workspace` crashed and reported nothing. `gles` stays on purpose (OpenGL fallback for hosts without Vulkan/DX12) — run `cargo test -p kataglyphis_webgpu_renderer --locked` on a desktop Windows machine instead.

  Not a regression from the wgpu 30 upgrade. The old "8 passed / 0 failed" baseline was recorded on 2026-07-17, and the renderer crate landed on 2026-07-18 — the container test lane has therefore *never* run with that crate present. The image is Server Core with no GPU stack; a wgpu-linked binary needs graphics DLLs it does not ship.

  `Invoke-StevedoreBuild.ps1 -Test` used to fail as a whole because of this. It no longer does — the exclusion lives in `rust-test-all.ps1`, so the container lane is green and reports 8 passed. Drop the `--exclude` once the image carries the missing DLLs.

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
bash third_party/ContainerHub/linux/scripts/lint-workflows.sh .
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

**Every feature path lints clean** — measured 2026-08-07 on Ubuntu 24.04 with rustc 1.97.1, `cargo clippy --all-targets --locked --features <set> -- -D warnings`:

| Feature set | Result |
| --- | --- |
| default (what CI lints) | clean |
| `gstreamer,gui_linux,onnxruntime,onnx_tract` | clean |
| `gui_unix` | clean |
| `burn_demos` | clean |

Worth stating plainly because none of the non-default rows has *ever* been linted in CI: the Linux lane lints default features (which are empty) and the Windows lane's fmt/clippy silently skip. `crates/media`, `crates/gui` and the ONNX paths are unguarded, not neglected — the `feature-matrix` job exists to keep it that way.

Note the feature names belong to the **root package**. `cargo clippy --workspace --features gstreamer` fails with *"package `kataglyphis_gui` does not have feature `gstreamer`"* because `--workspace` applies the list to every member; drop `--workspace` to scope it to the root.

## Packaging: what actually happens when you run it

Verified 2026-08-07 by running `Build-Windows.ps1 -SkipTests` in `:winamd64`:

- **MSIX works.** `Kataglyphis.RustProjectTemplate_2.3.4.0_x64.msix`, 51.69 MB, manifest with every token substituted. `makeappx.exe` resolves via `Resolve-WindowsSdkToolPath` to `Windows Kits\10\bin\10.0.26100.0\x64\`. The identity became `Kataglyphis.OxidANT` on 2026-09-05, so a build today writes `Kataglyphis.OxidANT_<VERSION>_x64.msix`; the old filename stands here because it is what that run actually produced. Windows treats the two identities as different apps, so an installation predating that date is not upgraded — it has to be uninstalled first, see the MSIX section of the README.
- **MSI works, but only since the WiX v4 migration** (2026-08-07). It had never produced a file. Two independent faults, both masked by the step being optional:
  1. `cargo wix -p kataglyphis_cli` looks for WXS files inside the package it was pointed at (`crates/cli/wix/`); this repo keeps its single WiX source at the workspace root. `Msi.WxsFile` had been sitting unread in the config the whole time.
  2. Even with the path fixed, **cargo-wix cannot drive this image.** 0.3.9 is its newest release and it shells out to WiX v3's `candle.exe`/`light.exe`. ContainerHub installs **WiX 4.0.6** as a dotnet tool — a single `wix.exe`, no candle — so it failed with *"The compiler application ('candle') does not exist at the 'C:\WiX' path"*.

  `Build-Windows.ps1` now calls `wix.exe build` directly (resolved from `$env:WIX`, then PATH) and `wix/main.wxs` is **WiX v4 schema**: `<Package>` instead of `<Product>` + inner `<Package>`, `<SummaryInformation>`, `<StandardDirectory>` instead of the `TARGETDIR` nesting, `Bitness='always64'` for `Win64='yes'`, `AllowAbsent` for `Absent`, and `<ui:WixUI>` for `<UIRef>`. `WixUI_FeatureTree` needs `-ext WixToolset.UI.wixext`, which the image already ships (4.0.4). Paths that move with the build — the binary follows `CARGO_TARGET_DIR` — go in as `-d Version= / ExeSource= / LicenseRtf=` preprocessor variables, so the WXS never assumes a `target\release` beside the workspace root.

  **If you touch this: cargo-wix is not an option again unless it gains WiX 4 support.** Check its releases before reintroducing it.

**Packaging and security steps are now `Invoke-BuildStep -Critical`, not `Invoke-BuildOptional`.** That matters because ContainerHub's `Invoke-BuildOptional` is `try { & $Script } catch { Write-BuildLogWarning }` and **never registers the step with the build context** — so it cannot appear in the summary at all. The pre-fix run reported **"7 steps, 7 succeeded, 0 failed (100% success rate)"** while MSI *and* the license check had failed. If you see a suspiciously perfect summary, that percentage covers only the `Invoke-BuildStep` steps; read the WARNING lines.

`cargo-deny licenses` also failed on that run (advisories, bans and sources passed) and was equally invisible. Fixed by allowing `BSL-1.0` in `deny.toml` — xxhash-rust via cubecl-common → burn; it was the only rejection.

`CARGO_TARGET_DIR` may be absolute — the in-container scripts set `C:\ct`. `Build-Windows.ps1` now handles that (`IsPathRooted`); before, `Join-Path` produced `C:\...\workspace\C:\ct\msix-staging` and MSIX died on "The filename, directory name, or volume label syntax is incorrect".

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

## The 2026-08-07 tract 0.22 → 0.23 migration

Dependabot offered this as `build(deps): bump tract-onnx from 0.22.3 to 0.23.4`. It is **not** a drop-in bump — it breaks in four separate ways, none of which the PR title suggests, and only `crates/inference/src/person_detection/{mod,tract_backend}.rs` are affected (the `onnx_tract` feature is off by default, so nothing else notices).

- **`SimplePlan` is gone from the prelude.** It was renamed to `RunnableModel`; the alias to use is `TypedRunnableModel`, and it is **fully applied** — `pub type TypedRunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>>`. Passing it a generic argument (the old third `TypedModel` parameter) fails with *"type alias takes 0 generic arguments but 1 generic argument was supplied"*.
- **`run` takes `self: &Arc<Self>`.** A `Box<TractPlan>` does not resolve the method at all — the error is a bare *"no method named `run`"*, which reads like a missing trait import and is not.
- **`into_runnable()` already returns an `Arc`.** So the Arc is neither ours to add nor to strip; `load_tract_model` returns `Arc<TractPlan>` and the `Backend::Tract` variant stores it directly.
- **`Tensor::as_slice` was removed.** The safe replacement is `to_plain_array_view::<f32>()`, which errors unless the storage is plain *and* the datum type matches — the same two conditions the old call checked. rustc's *"there is a method `slice` with a similar name"* suggestion points somewhere else entirely; do not follow it.

The lockfile also gains `tract-extra`, `tract-pulse`, `tract-pulse-opl`, `tract-transformers` and `typeid`. `cargo deny check licenses` passes with them (verified, exit 0) — no new `deny.toml` allowances were needed.

## Do not let `cargo update` take zune-core to 0.5.2

`zune-core` is held at **0.5.1** in `Cargo.lock` on purpose. 0.5.2 breaks `zune-jpeg` 0.5.15:

```
error: macro expansion ends with an incomplete expression: expected expression
  --> zune-jpeg-0.5.15/src/mcu_prog.rs:463:17
error: could not compile `zune-jpeg` (lib) due to 1 previous error
```

zune-jpeg consumes a macro from zune-core, and 0.5.2 changed it. **There is no forward fix**: 0.5.15 is zune-jpeg's newest release and 0.5.2 is zune-core's, so the two are incompatible at their respective tips. Both arrive transitively (via `image`, into the renderer), so nothing in a `Cargo.toml` pins them — only the lockfile does.

A bare `cargo update` reintroduces it silently, and it only shows up in a **release** build of the full workspace; `cargo test` and `cargo check -p ...` stay green because they never reach that crate. If you run `cargo update`, put it back:

```bash
cargo update -p zune-core --precise 0.5.1
```

Re-check when zune-jpeg publishes past 0.5.15.

## Conventions

- Version pins/single sources of truth follow the ContainerHub ecosystem; don't duplicate what the submodule documents — link to it.
- Never commit build outputs: `/target`, root `/debug`, `/profile`, `/release` are gitignored.
