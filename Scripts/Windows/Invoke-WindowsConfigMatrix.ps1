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

# RunApp = $false for every GUI-featured configuration: those binaries die at
# process load in the headless servercore CI container (no display, GUI/ONNX
# runtime DLLs absent) before main() runs - the observable symptom is an
# instant nonzero exit with zero output. Building them is the CI-provable
# part; the plain 'base' binary proves the run path.
#
# Profiles = debug-only for the GUI configurations, for two measured reasons:
# the optimized 'profile' build of gui_windows+onnx_tract dies with
# "rustc-LLVM ERROR: out of memory" on the 16 GB runner (4 parallel LLVM
# backends over wgpu/egui/tract in an opt+debuginfo profile), and the full
# 5-config x 3-profile matrix ran 3.5 hours before that. Debug proves the
# CI-provable part ("this feature set compiles on Windows"); 'base' keeps
# covering all three profiles including optimized codegen.
$configurationMatrix = @(
  @{ Name = 'base'; Features = ''; RunApp = $true; Profiles = @('debug', 'profile', 'release') },
  @{ Name = 'gui_windows'; Features = 'gui_windows'; RunApp = $false; Profiles = @('debug') },
  @{ Name = 'gui_windows_onnx_tract'; Features = 'gui_windows,onnx_tract'; RunApp = $false; Profiles = @('debug') },
  @{ Name = 'gui_windows_onnxruntime_directml'; Features = 'gui_windows,onnxruntime_directml'; RunApp = $false; Profiles = @('debug') },
  @{ Name = 'gui_windows_onnxruntime_cuda'; Features = 'gui_windows,onnxruntime_cuda'; RunApp = $false; Profiles = @('debug') }
)

$resolvedConfigurations = Resolve-Configurations -Matrix $configurationMatrix -RequestedConfigurations $Configurations

foreach ($configuration in $resolvedConfigurations) {
  $featureLabel = if ([string]::IsNullOrWhiteSpace($configuration.Features)) { '<none>' } else { $configuration.Features }
  Write-Host "==> Configuration: $($configuration.Name) (features: $featureLabel)"

  $buildOnly = -not $configuration.RunApp
  # A configuration's own profile list wins over the script parameter; the
  # parameter stays the default for configs that don't restrict themselves.
  $configProfiles = if ($configuration.ContainsKey('Profiles')) { $configuration.Profiles } else { $Profiles }
  & $runScript -Profiles $configProfiles -Features $configuration.Features -AppArgs $AppArgs -BuildOnly:$buildOnly
  if ($LASTEXITCODE -ne 0) {
    throw "Configuration '$($configuration.Name)' failed."
  }
}
