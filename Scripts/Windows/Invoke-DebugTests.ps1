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
  @{ Name = 'WebGPU renderer tests'; Args = @('test', '--package', 'kataglyphis_webgpu_renderer') }
)

foreach ($step in $testSteps) {
  Write-Host "==> $($step.Name)"
  & cargo @($step.Args)
  if ($LASTEXITCODE -ne 0) {
    throw "$($step.Name) failed with exit code $LASTEXITCODE."
  }
}
