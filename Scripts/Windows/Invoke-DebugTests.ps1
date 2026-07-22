param(
  [string]$Package = 'kataglyphis_rustprojecttemplate'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$testSteps = @(
  @{ Name = 'Unit tests'; Args = @('test', '--package', $Package, '--lib') },
  @{ Name = 'Integration tests'; Args = @('test', '--package', 'kataglyphis_cli', '--test', 'integration') },
  @{ Name = 'Fuzz tests'; Args = @('test', '--package', $Package, '--test', 'fuzz_test') },
  # GPU-dependent golden tests skip themselves when no adapter is present
  # (e.g. inside the CI container); loader/camera tests always run.
  # LIB tests only in the Windows container. The GPU integration-test
  # binaries (gpu_timing, headless, occlusion, ...) link wgpu and die at
  # PROCESS START with STATUS_DLL_NOT_FOUND (0xc0000135) on servercore - a
  # missing graphics DLL, so their own "skip without an adapter" guard never
  # runs. The lib tests (100+ pure-CPU tests) load fine and are the coverage
  # this container can actually provide; the GPU tests run on Linux CI and on
  # dev boxes with real adapters. Same approach as the C++ engine's
  # "CPU-only tests inside the container" step.
  @{ Name = 'WebGPU renderer tests (lib only; GPU test bins cannot start on servercore)'; Args = @('test', '--package', 'kataglyphis_webgpu_renderer', '--lib') }
)

foreach ($step in $testSteps) {
  Write-Host "==> $($step.Name)"
  & cargo @($step.Args)
  if ($LASTEXITCODE -ne 0) {
    throw "$($step.Name) failed with exit code $LASTEXITCODE."
  }
}
