param(
  [string[]]$Profiles = @('debug', 'profile', 'release'),
  [string]$Features = '',
  [string[]]$AppArgs = @('stats', '--path', 'README.md'),
  [string]$Package = 'kataglyphis_cli',
  [string]$Binary = 'kataglyphis_rustprojecttemplate',
  [string]$TargetDir = ''
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

  foreach ($profile in $normalizedProfiles) {
    if ($profile -notin $knownProfiles) {
      throw "Unsupported profile '$profile'. Valid values: $($knownProfiles -join ', ')."
    }
  }

  return $normalizedProfiles
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repoRoot

$resolvedProfiles = Resolve-Profiles -RequestedProfiles $Profiles
$resolvedTargetDir = if ([string]::IsNullOrWhiteSpace($TargetDir)) { 'target' } else { $TargetDir }
$featureLabel = if ([string]::IsNullOrWhiteSpace($Features)) { '<none>' } else { $Features }

foreach ($profile in $resolvedProfiles) {
  $buildArgs = @('build', '--package', $Package, '--bin', $Binary)
  $profileDir = 'debug'

  switch ($profile) {
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

  Write-Host "==> Building $Binary [$profile] with features: $featureLabel"
  & cargo @buildArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Build failed for profile '$profile' and features '$featureLabel'."
  }

  $binaryPath = Join-Path $repoRoot "$resolvedTargetDir\$profileDir\$Binary.exe"
  if (-not (Test-Path $binaryPath)) {
    throw "Built binary not found: $binaryPath"
  }

  Write-Host "==> Running $Binary [$profile] with args: $($AppArgs -join ' ')"
  & $binaryPath @AppArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed for profile '$profile' and features '$featureLabel'."
  }
}
