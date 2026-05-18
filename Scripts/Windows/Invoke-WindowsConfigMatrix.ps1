param(
  [string[]]$Configurations = @('all'),
  [string[]]$Profiles = @('debug', 'profile', 'release'),
  [string[]]$AppArgs = @('stats', '--path', 'README.md')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-Configurations([object[]]$Matrix, [string[]]$RequestedConfigurations) {
  $normalizedConfigurations = @()

  foreach ($requestedConfiguration in $RequestedConfigurations) {
    if ([string]::IsNullOrWhiteSpace($requestedConfiguration)) {
      continue
    }

    foreach ($configurationPart in ($requestedConfiguration -split ',')) {
      $trimmedConfiguration = $configurationPart.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmedConfiguration)) {
        $normalizedConfigurations += $trimmedConfiguration
      }
    }
  }

  if ($normalizedConfigurations.Count -eq 1 -and $normalizedConfigurations[0] -eq 'all') {
    return $Matrix
  }

  $resolved = @()
  foreach ($requested in $normalizedConfigurations) {
    $match = $Matrix | Where-Object { $_.Name -eq $requested }
    if ($null -eq $match) {
      $valid = ($Matrix | ForEach-Object { $_.Name }) -join ', '
      throw "Unsupported configuration '$requested'. Valid values: $valid."
    }
    $resolved += $match
  }

  return $resolved
}

$runScript = Join-Path $PSScriptRoot 'Run-AppProfiles.ps1'
if (-not (Test-Path $runScript)) {
  throw "Required script not found: $runScript"
}

$configurationMatrix = @(
  @{ Name = 'base'; Features = '' },
  @{ Name = 'gui_windows'; Features = 'gui_windows' },
  @{ Name = 'gui_windows_onnx_tract'; Features = 'gui_windows,onnx_tract' },
  @{ Name = 'gui_windows_onnxruntime_directml'; Features = 'gui_windows,onnxruntime_directml' },
  @{ Name = 'gui_windows_onnxruntime_cuda'; Features = 'gui_windows,onnxruntime_cuda' }
)

$resolvedConfigurations = Resolve-Configurations -Matrix $configurationMatrix -RequestedConfigurations $Configurations

foreach ($configuration in $resolvedConfigurations) {
  $featureLabel = if ([string]::IsNullOrWhiteSpace($configuration.Features)) { '<none>' } else { $configuration.Features }
  Write-Host "==> Configuration: $($configuration.Name) (features: $featureLabel)"

  & $runScript -Profiles $Profiles -Features $configuration.Features -AppArgs $AppArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Configuration '$($configuration.Name)' failed."
  }
}
