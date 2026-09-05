# Runs INSIDE the Windows container: full `cargo test` (debug profile) for the
# workspace -- unit tests, integration tests (tests/integration.rs), and the
# proptest fuzz suite (tests/fuzz_test.rs), plus doc tests.
# Same wcifs-safe layout as rust-build-all.ps1: writes only to C:\ct / C:\ch,
# everything logged to the mounted C:\host-scratch.
#requires -Version 7.0

$ProgressPreference = 'SilentlyContinue'
# See rust-build-all.ps1: the driver stages this module into the scratch mount
# because third_party is excluded from the sources copied into the container.
Import-Module 'C:\host-scratch\WindowsContainerLog.Common.psm1' -Force
Start-ContainerLog -Path 'C:\host-scratch\in-container-test.log'

Write-ContainerLog "=== Rust container test run (debug): unit + integration + fuzz(proptest) + doc ==="
Write-ContainerLog "cpus: $env:NUMBER_OF_PROCESSORS"
[void](Invoke-ContainerLoggedCommand 'rustc -vV')

New-Item -ItemType Directory -Force -Path C:\ct, C:\ch | Out-Null
$env:CARGO_TARGET_DIR = 'C:\ct'
$env:CARGO_HOME = 'C:\ch'
Set-Location C:\ws-mnt

# kataglyphis_webgpu_renderer is excluded HERE ONLY, and only because of the
# container image -- not because its tests are broken.
#
# Any test binary of that crate links wgpu, and wgpu's `gles` backend makes the
# executable import opengl32.dll at load time. Windows Server Core does not ship
# opengl32.dll, so the process dies with 0xc0000135 (STATUS_DLL_NOT_FOUND)
# before main() ever runs -- there is no runtime switch that avoids it, because
# the import is resolved by the loader. Letting it run just turns the whole
# `cargo test --workspace` into a crash with no test results at all.
#
# We keep the `gles` feature deliberately: it is the OpenGL fallback for
# machines without Vulkan/DX12. These tests are meant to run on a desktop
# Windows host, where opengl32.dll exists:
#
#   cargo test -p kataglyphis_webgpu_renderer --locked
#
# Revisit if the base image ever gains opengl32.dll, or if wgpu stops importing
# it eagerly.
Write-ContainerLog 'SKIPPING kataglyphis_webgpu_renderer: Server Core has no opengl32.dll (see comment in this script). Run it on a desktop Windows host.'

$code = Invoke-ContainerLoggedCommand 'cargo test --workspace --locked --exclude kataglyphis_webgpu_renderer'
if ($code -ne 0) { Write-ContainerLog "TESTS FAILED (exit $code)"; exit $code }
Write-ContainerLog 'ALL TESTS PASSED (kataglyphis_webgpu_renderer excluded -- see above)'
exit 0

