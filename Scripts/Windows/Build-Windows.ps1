<#
.SYNOPSIS
  Windows build and packaging script for Rust projects.
  Similar pattern to Kataglyphis-BeschleunigerBallett's Build-Windows.ps1

.DESCRIPTION
  - Uses ContainerHub's WindowsBuild.Common.psm1 for structured logging.
  - Runs cargo build, test, lint via the ContainerHub Rust build script.
  - Packages MSIX using local config and template.
#>

param(
  [string[]]$Configurations = @('all'),
  [switch]$SkipMsix,
  [switch]$SkipMsi,
  [switch]$SkipBuild,
  [switch]$SkipTests,
  [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-OrDefault([string]$Value, [string]$DefaultValue) {
  if ([string]::IsNullOrWhiteSpace($Value)) { return $DefaultValue }
  return $Value
}

function Get-ConfigValue {
  param(
    [Parameter(Mandatory)]
    $Config,
    [Parameter(Mandatory)]
    [string]$Path
  )

  $cursor = $Config
  foreach ($segment in ($Path -split '\.')) {
    if ($null -eq $cursor) { return $null }
    try {
      $cursor = $cursor[$segment]
    } catch {
      return $null
    }
  }

  return $cursor
}

function Assert-Command([string]$Name, [string]$InstallHint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name not found. $InstallHint"
  }
}

function Resolve-Executable([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  $sdkRoots = @(
    "C:/Program Files (x86)/Windows Kits/10/bin",
    "C:/Program Files/Windows Kits/10/bin"
  )

  foreach ($root in $sdkRoots) {
    if (-not (Test-Path $root)) { continue }
    $candidates = Get-ChildItem -Path $root -Recurse -Filter ("{0}.exe" -f $Name) -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\x64\\" -or $_.FullName -match "/x64/" } |
      Sort-Object FullName -Descending

    if ($candidates -and $candidates.Count -gt 0) {
      $fso = New-Object -ComObject Scripting.FileSystemObject
      return $fso.GetFile($candidates[0].FullName).ShortPath
    }
  }
  return $null
}

function Normalize-Version([string]$RawVersion) {
  $segments = $RawVersion.Split('.')
  if ($segments.Count -eq 3) { return "$RawVersion.0" }
  if ($segments.Count -ne 4) { throw "Version '$RawVersion' is invalid. Use Major.Minor.Build or Major.Minor.Build.Revision" }
  return $RawVersion
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$containerHubModulesRoot = Join-Path $repoRoot 'ExternalLib\Kataglyphis-ContainerHub\windows\scripts\modules'

$buildModule = Join-Path $containerHubModulesRoot 'WindowsBuild.Common.psm1'
if (-not (Test-Path $buildModule)) {
  throw "Required module not found: $buildModule"
}

Import-Module $buildModule -Force

$defaultConfigPath = Join-Path $PSScriptRoot 'Build-Windows.config.psd1'
$configPath = Get-OrDefault $env:BUILD_WINDOWS_CONFIG $defaultConfigPath
if (-not (Test-Path $configPath)) {
  throw "Build config not found: $configPath"
}
$config = Import-PowerShellDataFile -Path $configPath

$workspaceRootEnvVar = Get-OrDefault $env:WORKSPACE_ROOT_ENV (Get-ConfigValue -Config $config -Path 'Build.WorkspaceRootEnv')
$workspaceEnvItem = Get-Item -Path "Env:$workspaceRootEnvVar" -ErrorAction SilentlyContinue
$workspaceRootFromEnv = if ($null -ne $workspaceEnvItem) { $workspaceEnvItem.Value } else { $null }
$workspaceRoot = Get-OrDefault $workspaceRootFromEnv $repoRoot
$workspacePath = Resolve-WorkspacePath -Path $workspaceRoot

$logDir = Get-OrDefault $env:BUILD_LOG_DIR (Get-ConfigValue -Config $config -Path 'Build.LogDir')

$cargoTargetDir = Get-OrDefault $env:CARGO_TARGET_DIR (Get-ConfigValue -Config $config -Path 'Build.CargoTargetDir')
$cargoFeatures = Get-OrDefault $env:CARGO_FEATURES ((Get-ConfigValue -Config $config -Path 'Build.CargoFeatures') -join ',')

$binary = Get-OrDefault $env:BINARY (Get-ConfigValue -Config $config -Path 'Msix.Binary')

$msixName = Get-OrDefault $env:MSIX_PACKAGE_NAME (Get-ConfigValue -Config $config -Path 'Msix.PackageName')
$msixPublisher = Get-OrDefault $env:MSIX_PUBLISHER (Get-ConfigValue -Config $config -Path 'Msix.Publisher')
$msixPublisherDisplayName = Get-OrDefault $env:MSIX_PUBLISHER_DISPLAY_NAME (Get-ConfigValue -Config $config -Path 'Msix.PublisherDisplayName')
$msixDisplayName = Get-OrDefault $env:MSIX_DISPLAY_NAME (Get-ConfigValue -Config $config -Path 'Msix.DisplayName')
$msixDescription = Get-OrDefault $env:MSIX_DESCRIPTION (Get-ConfigValue -Config $config -Path 'Msix.Description')
$msixVersion = Get-OrDefault $env:MSIX_VERSION (Get-ConfigValue -Config $config -Path 'Msix.Version')
$msixMinVersion = Get-OrDefault $env:MSIX_MIN_VERSION (Get-ConfigValue -Config $config -Path 'Msix.MinVersion')

$context = New-BuildContext -Workspace $workspacePath -LogDir $logDir -StopOnError

try {
  Open-BuildLog -Context $context

  Write-BuildLog -Context $context -Message "Workspace: $workspacePath"
  Write-BuildLog -Context $context -Message "Binary: $binary"
  Write-BuildLog -Context $context -Message "MSIX: $msixName"

  $fastBuildDir = Initialize-BuildCacheEnvironment -Context $context
  $isolatedWorkspace = Join-Path $fastBuildDir "workspace"

  Invoke-BuildStep -Context $context -StepName 'Sync Source' -Critical -Script {
    Sync-BuildArtifacts -Context $context -Source $workspacePath -Destination $isolatedWorkspace -ExcludeCommonRustAndCppCache
  } | Out-Null

  $originalWorkspacePath = $workspacePath
  $workspacePath = $isolatedWorkspace
  Set-Location -Path $workspacePath

  $scoopShims = "C:\Users\ContainerAdministrator\scoop\shims"
  if (-not ($env:PATH -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ieq $scoopShims })) {
    Write-BuildLog -Context $context -Message "Prepending scoop shims to PATH: $scoopShims"
    $env:PATH = "$scoopShims;$env:PATH"
  }

  Invoke-BuildStep -Context $context -StepName 'Verify Toolchain' -Critical -Script {
    Assert-Command -Name 'cargo' -InstallHint 'Install Rust toolchain via rustup'
    Invoke-BuildExternal -Context $context -File 'rustup' -Parameters @('--version') | Out-Null
    Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('--version') | Out-Null
  } | Out-Null

  if ($Clean) {
    Invoke-BuildStep -Context $context -StepName 'Clean Build Artifacts' -Script {
      Write-BuildLog -Context $context -Message "Cleaning cargo build artifacts..."
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('clean') | Out-Null

      $flutterExe = Get-Command flutter -ErrorAction SilentlyContinue
      if ($flutterExe) {
        Write-BuildLog -Context $context -Message "Cleaning Flutter build artifacts..."
        Invoke-BuildExternal -Context $context -File 'flutter' -Parameters @('clean') | Out-Null
      } else {
        Write-BuildLogWarning -Context $context -Message "Flutter not found, skipping Flutter clean"
      }
    } | Out-Null
  }

  if (-not $SkipBuild) {
    Invoke-BuildStep -Context $context -StepName 'Security Checks (audit & deny)' -Script {
    try {
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('install', '--locked', 'cargo-audit', 'cargo-deny') | Out-Null
    } catch {
      Write-BuildLogWarning -Context $context -Message "Failed to install cargo-audit/cargo-deny: $_"
    }

    try {
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('audit') | Out-Null
    } catch {
      Write-BuildLogWarning -Context $context -Message "cargo audit failed or not available: $_"
    }

    try {
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('deny', 'check', 'advisories', 'licenses', 'bans', 'sources') | Out-Null
    } catch {
      Write-BuildLogWarning -Context $context -Message "cargo deny checks failed or not available: $_"
    }
    } | Out-Null

    Invoke-BuildStep -Context $context -StepName 'Format Check' -Critical -Script {
      Invoke-BuildExternal -Context $context -File 'rustup' -Parameters @('component', 'add', 'rustfmt') | Out-Null
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('fmt', '--all', '--', '--check') | Out-Null
    } | Out-Null

    Invoke-BuildStep -Context $context -StepName 'Linting (cargo clippy)' -Critical -Script {
      Invoke-BuildExternal -Context $context -File 'rustup' -Parameters @('component', 'add', 'clippy') | Out-Null
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('clippy', '--all-targets', '--all-features', '--', '-D', 'warnings') | Out-Null
    } | Out-Null

    if (-not $SkipTests) {
      Invoke-BuildStep -Context $context -StepName 'Unit Tests' -Critical -Script {
        $testParams = @('test', '--all', '--verbose')
        if (-not [string]::IsNullOrWhiteSpace($cargoFeatures)) {
          $testParams += @('--features', $cargoFeatures)
        }
        Invoke-BuildExternal -Context $context -File 'cargo' -Parameters $testParams | Out-Null
      } | Out-Null
    }

    Invoke-BuildStep -Context $context -StepName 'Release Build' -Critical -Script {
      $buildParams = @('build', '--release', '--package', 'kataglyphis_cli', '--bin', $binary)
      if (-not [string]::IsNullOrWhiteSpace($cargoFeatures)) {
        $buildParams += @('--features', $cargoFeatures)
      }
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters $buildParams | Out-Null
    } | Out-Null
  }

  if (-not $SkipMsix) {
    Invoke-BuildOptional -Context $context -Name 'MSIX Packaging' -Script {
      $makeappxPath = Resolve-Executable -Name 'makeappx'
      if (-not $makeappxPath) {
        throw 'makeappx.exe not found. Install Windows SDK or add it to PATH.'
      }

      $resolvedVersion = $msixVersion
      $versionFile = Join-Path $workspacePath 'version.txt'
      if (Test-Path $versionFile) {
        $resolvedVersion = (Get-Content -Path $versionFile).Trim()
        if ($resolvedVersion -notmatch '\.' ) {
          $resolvedVersion = "$resolvedVersion.0.0"
        }
      }
      if ($resolvedVersion -match '^v') {
        $resolvedVersion = $resolvedVersion.Substring(1)
      }
      $resolvedVersion = Normalize-Version $resolvedVersion

      $cargoTargetFullPath = Join-Path $workspacePath $cargoTargetDir
      $releaseDir = Join-Path $cargoTargetFullPath 'release'

      $msixStaging = Join-Path $cargoTargetFullPath 'msix-staging'
      $assetsDir = Join-Path $msixStaging 'Assets'
      if (Test-Path $msixStaging) {
        Remove-Item $msixStaging -Recurse -Force
      }
      New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null

      $exePath = Join-Path $releaseDir "$binary.exe"
      if (-not (Test-Path $exePath)) {
        throw "Expected executable not found: $exePath"
      }

      Write-BuildLog -Context $context -Message "Copying binary and DLLs..."
      Copy-Item $exePath -Destination $msixStaging -Force
      Get-ChildItem -Path $releaseDir -Filter '*.dll' -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName -Destination $msixStaging -Force }

      $resourcesSource = Join-Path $workspacePath 'resources'
      if (Test-Path $resourcesSource) {
        Write-BuildLog -Context $context -Message "Copying resources from $resourcesSource"
        Copy-Item $resourcesSource -Destination (Join-Path $msixStaging 'resources') -Recurse -Force
      }

      $logoPath = Join-Path $workspacePath 'images\logo.png'
      if (-not (Test-Path $logoPath)) {
        $logoPath = Join-Path $workspacePath 'ExternalLib\Kataglyphis-ContainerHub\images\logo.png'
      }
      if (Test-Path $logoPath) {
        Write-BuildLog -Context $context -Message "Copying logos from $logoPath"
        Copy-Item $logoPath -Destination (Join-Path $assetsDir 'StoreLogo.png') -Force
        Copy-Item $logoPath -Destination (Join-Path $assetsDir 'Square44x44Logo.png') -Force
        Copy-Item $logoPath -Destination (Join-Path $assetsDir 'Square150x150Logo.png') -Force
        Copy-Item $logoPath -Destination (Join-Path $assetsDir 'Wide310x150Logo.png') -Force
      } else {
        function New-TransparentPng {
          param(
            [Parameter(Mandatory)]
            [string]$Path,
            [Parameter(Mandatory)]
            [int]$Width,
            [Parameter(Mandatory)]
            [int]$Height
          )
          Add-Type -AssemblyName System.Drawing
          $bmp = New-Object System.Drawing.Bitmap($Width, $Height)
          $gfx = [System.Drawing.Graphics]::FromImage($bmp)
          try {
            $gfx.Clear([System.Drawing.Color]::Transparent)
            $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
          } finally {
            $gfx.Dispose()
            $bmp.Dispose()
          }
        }
        Write-BuildLogWarning -Context $context -Message "Logo file not found, generating transparent placeholders"
        New-TransparentPng -Path (Join-Path $assetsDir 'StoreLogo.png') -Width 50 -Height 50
        New-TransparentPng -Path (Join-Path $assetsDir 'Square44x44Logo.png') -Width 44 -Height 44
        New-TransparentPng -Path (Join-Path $assetsDir 'Square150x150Logo.png') -Width 150 -Height 150
        New-TransparentPng -Path (Join-Path $assetsDir 'Wide310x150Logo.png') -Width 310 -Height 150
      }

      $manifestTemplateRel = Get-ConfigValue -Config $config -Path 'Msix.ManifestTemplate'
      $manifestTemplatePath = if ([System.IO.Path]::IsPathRooted($manifestTemplateRel)) { $manifestTemplateRel } else { Join-Path $workspacePath $manifestTemplateRel }
      if (-not (Test-Path $manifestTemplatePath)) {
        throw "MSIX manifest template not found: $manifestTemplatePath"
      }

      $exeRelPath = "$binary.exe"
      $templateContent = Get-Content -Path $manifestTemplatePath -Raw -Encoding UTF8

      function ConvertTo-XmlEscapedText {
        param([AllowNull()][string]$Value)
        if ($null -eq $Value) { return '' }
        return [System.Security.SecurityElement]::Escape($Value)
      }

      $manifestXml = $templateContent
      $manifestXml = $manifestXml -replace '__MSIX_NAME__', (ConvertTo-XmlEscapedText $msixName)
      $manifestXml = $manifestXml -replace '__MSIX_PUBLISHER__', (ConvertTo-XmlEscapedText $msixPublisher)
      $manifestXml = $manifestXml -replace '__MSIX_VERSION__', (ConvertTo-XmlEscapedText $resolvedVersion)
      $manifestXml = $manifestXml -replace '__MSIX_MIN_VERSION__', (ConvertTo-XmlEscapedText $msixMinVersion)
      $manifestXml = $manifestXml -replace '__MSIX_DISPLAY_NAME__', (ConvertTo-XmlEscapedText $msixDisplayName)
      $manifestXml = $manifestXml -replace '__MSIX_PUBLISHER_DISPLAY_NAME__', (ConvertTo-XmlEscapedText $msixPublisherDisplayName)
      $manifestXml = $manifestXml -replace '__MSIX_DESCRIPTION__', (ConvertTo-XmlEscapedText $msixDescription)
      $manifestXml = $manifestXml -replace '__EXE_REL_PATH__', (ConvertTo-XmlEscapedText $exeRelPath)
      $manifestXml = $manifestXml -replace '__STORE_LOGO_REL__', 'Assets/StoreLogo.png'
      $manifestXml = $manifestXml -replace '__LOGO44_REL__', 'Assets/Square44x44Logo.png'
      $manifestXml = $manifestXml -replace '__LOGO150_REL__', 'Assets/Square150x150Logo.png'

      Set-Content -Path (Join-Path $msixStaging 'AppxManifest.xml') -Value $manifestXml -Encoding UTF8

      $distDir = Join-Path $workspacePath 'dist\msix'
      New-Item -ItemType Directory -Path $distDir -Force | Out-Null

      $packageFile = Join-Path $distDir "$msixName`_$resolvedVersion`_x64.msix"
      if (Test-Path $packageFile) {
        Remove-Item $packageFile -Force
      }

      Write-BuildLog -Context $context -Message "Creating MSIX package: $packageFile"
      Invoke-BuildExternal -Context $context -File $makeappxPath -Parameters @('pack', '/d', $msixStaging, '/p', $packageFile, '/o') | Out-Null

      Write-BuildLogSuccess -Context $context -Message "MSIX package created: $packageFile"
    }
  }

  # MSI Packaging with cargo-wix
  $msiEnabled = Get-ConfigValue -Config $config -Path 'Msi.Enabled'
  if (-not $SkipMsi -and $msiEnabled) {
    Invoke-BuildOptional -Context $context -Name 'MSI Packaging' -Script {
      Write-BuildLog -Context $context -Message "Installing cargo-wix..."
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('install', 'cargo-wix') | Out-Null

      $resolvedVersion = $msixVersion
      $versionFile = Join-Path $workspacePath 'version.txt'
      if (Test-Path $versionFile) {
        $resolvedVersion = (Get-Content -Path $versionFile).Trim()
      }
      if ($resolvedVersion -match '^v') {
        $resolvedVersion = $resolvedVersion.Substring(1)
      }
      # MSI version must be X.Y.Z format (3 components max for cargo-wix)
      $versionParts = $resolvedVersion.Split('.')
      if ($versionParts.Count -gt 3) {
        $resolvedVersion = "$($versionParts[0]).$($versionParts[1]).$($versionParts[2])"
      }

      $msiOutputName = Get-OrDefault $env:MSI_OUTPUT_NAME (Get-ConfigValue -Config $config -Path 'Msi.OutputName')
      if ([string]::IsNullOrWhiteSpace($msiOutputName)) {
        $msiOutputName = $binary
      }

      $msiDistDir = Join-Path $workspacePath 'dist\msi'
      New-Item -ItemType Directory -Path $msiDistDir -Force | Out-Null

      $msiFile = Join-Path $msiDistDir "$msiOutputName-$resolvedVersion-x64.msi"

      Write-BuildLog -Context $context -Message "Creating MSI package: $msiFile"

      # Run cargo-wix to create the MSI
      $wixParams = @('wix', '--no-build', '--nocapture', '-p', 'kataglyphis_cli', '--output', $msiFile)
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters $wixParams | Out-Null

      Write-BuildLogSuccess -Context $context -Message "MSI package created: $msiFile"
    }
  }

  Invoke-BuildStep -Context $context -StepName 'Sync Artifacts' -Critical -Script {
    $distSource = Join-Path $workspacePath 'dist'
    $distDest = Join-Path $originalWorkspacePath 'dist'
    if (Test-Path $distSource) {
      Write-BuildLog -Context $context -Message "Syncing distribution artifacts to $distDest"
      Sync-BuildArtifacts -Context $context -Source $distSource -Destination $distDest
    }
    $targetSource = Join-Path $workspacePath $cargoTargetDir
    $targetDest = Join-Path $originalWorkspacePath $cargoTargetDir
    if (Test-Path $targetSource) {
      Write-BuildLog -Context $context -Message "Syncing cargo target directory to $targetDest"
      Sync-BuildArtifacts -Context $context -Source $targetSource -Destination $targetDest -ExcludeCommonRustAndCppCache
    }
  } | Out-Null

  Write-BuildLogSuccess -Context $context -Message 'Windows build completed.'
} finally {
  Write-BuildSummary -Context $context
  Close-BuildLog -Context $context
}

if ($context.Results.Failed.Count -gt 0) {
  throw "Windows build completed with failures ($($context.Results.Failed.Count) steps failed)."
}
