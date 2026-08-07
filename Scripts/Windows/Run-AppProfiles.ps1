param(
#requires -Version 7.0

  [string[]]$Profiles = @('debug', 'profile', 'release'),
  [string]$Features = '',
  [string[]]$AppArgs = @('stats', '--path', 'README.md'),
  [string]$Package = 'kataglyphis_cli',
  [string]$Binary = 'kataglyphis_cli',
  [string]$TargetDir = '',
  # Build the binary but skip launching it. Needed for GUI-featured builds in
  # the headless servercore CI container, where the process dies at load time
  # (missing display/DLLs) before main() ever runs.
  [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-Profiles([string[]]$RequestedProfiles) {
  $knownProfiles = @('debug', 'profile', 'release')
  $normalizedProfiles = @()

  foreach ($requestedProfile in $RequestedProfiles) {
    if ([string]::IsNullOrWhiteSpace($requestedProfile)) {
      continue
    }

    foreach ($profilePart in ($requestedProfile -split ',')) {
      $trimmedProfile = $profilePart.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmedProfile)) {
        $normalizedProfiles += $trimmedProfile
      }
    }
  }

  if ($normalizedProfiles.Count -eq 1 -and $normalizedProfiles[0] -eq 'all') {
    return $knownProfiles
  }

  foreach ($buildProfile in $normalizedProfiles) {
    if ($buildProfile -notin $knownProfiles) {
      throw "Unsupported profile '$buildProfile'. Valid values: $($knownProfiles -join ', ')."
    }
  }

  return $normalizedProfiles
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$resolvedProfiles = Resolve-Profiles -RequestedProfiles $Profiles
$resolvedTargetDir = if ([string]::IsNullOrWhiteSpace($TargetDir)) { 'target' } else { $TargetDir }
$featureLabel = if ([string]::IsNullOrWhiteSpace($Features)) { '<none>' } else { $Features }

foreach ($buildProfile in $resolvedProfiles) {
  $buildArgs = @('build', '--package', $Package, '--bin', $Binary)
  $profileDir = 'debug'

  switch ($buildProfile) {
    'profile' {
      $buildArgs += @('--profile', 'profile')
      $profileDir = 'profile'
    }
    'release' {
      $buildArgs += '--release'
      $profileDir = 'release'
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($Features)) {
    $buildArgs += @('--features', $Features)
  }

  Write-Host "==> Building $Binary [$buildProfile] with features: $featureLabel"
  & cargo @buildArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Build failed for profile '$buildProfile' and features '$featureLabel'."
  }

  $binaryPath = Join-Path $repoRoot "$resolvedTargetDir\$profileDir\$Binary.exe"
  if (-not (Test-Path $binaryPath)) {
    throw "Built binary not found: $binaryPath"
  }

  if ($BuildOnly) {
    Write-Host "==> Build-only for features '$featureLabel': skipping the app run."
    continue
  }

  Write-Host "==> Running $Binary [$buildProfile] with args: $($AppArgs -join ' ')"
  & $binaryPath @AppArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed for profile '$buildProfile' and features '$featureLabel' (exit $LASTEXITCODE)."
  }
}

