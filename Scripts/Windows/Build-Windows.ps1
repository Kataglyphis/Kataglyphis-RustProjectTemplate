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
#requires -Version 7.0

  [string[]]$Configurations = @('all'),
  [switch]$SkipMsix,
  [switch]$SkipMsi,
  [switch]$SkipBuild,
  [switch]$SkipTests,
  [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Get-OrDefault and Get-ConfigValue used to be defined here, byte-identical to
# ContainerHub's WindowsConfig.Common.psm1. They now come from that module -
# see the import block below.

# Assert-Command comes from ContainerHub's WindowsScripts.Shared.psm1, and SDK
# tool lookup from WindowsMsix.Common's Resolve-WindowsSdkToolPath (both
# imported below). The Resolve-Executable that used to sit here recursed the
# whole Windows Kits tree; the module version consults VsDevCmd's
# WindowsSdkVerBinPath / WindowsSDKVersion first and scans newest-first.

# Robocopy-backed tree sync. This script has always CALLED Sync-BuildArtifacts
# but no ContainerHub module version has ever DEFINED it - the packaging path
# simply never executed until the 2026-07-22 lane peel reached it, and it died
# on the first call. Defined locally with the semantics the call sites assume.
function Sync-BuildArtifacts {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [object] $Context,
    [Parameter(Mandatory = $true)] [string] $Source,
    [Parameter(Mandatory = $true)] [string] $Destination,
    [switch] $ExcludeCommonRustAndCppCache
  )

  if (-not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
  }

  # /E subdirs, /MT multithreaded, /R:1 /W:1 no retry-hangs on locked files,
  # /FFT coarse timestamps (bind mounts), /NOOFFLOAD no copy-offload over the
  # VM boundary - same rationale as Sync-FastLocalArtifactsToHost in
  # WindowsFlutter.Common.psm1.
  $robocopyArgs = @(
    $Source, $Destination,
    '/E', '/MT:16', '/R:1', '/W:1', '/FFT', '/NOOFFLOAD',
    '/NFL', '/NDL', '/NJH', '/NJS', '/nc', '/ns', '/np'
  )
  if ($ExcludeCommonRustAndCppCache) {
    $robocopyArgs += @('/XD', 'target', '.git', 'node_modules', 'build-clangcl-debug', 'build-clangcl-release', 'build-clangcl-profile')
  }

  & robocopy.exe @robocopyArgs > $null 2>&1
  $robocopyExit = $LASTEXITCODE
  # Robocopy exit codes are a BITMASK, not a severity scale:
  #   1 copied, 2 extra, 4 mismatch, 8 some files could not be copied,
  #   16 serious error / nothing copied.
  # Only 16 means the mirror did not happen. Bit 8 (exit 9 = 8+1 in CI) fires
  # routinely on a live bind-mounted tree - a transient lock on one file while
  # the rest copy fine - and treating it as fatal killed the packaging step.
  # The sibling Sync-FastLocalArtifactsToHost ignores the code entirely for
  # the same reason; this at least warns on a partial copy and fails only on
  # the catastrophic bit.
  if ($robocopyExit -ge 16) {
    throw "Sync-BuildArtifacts failed (robocopy exit $robocopyExit, serious error): '$Source' -> '$Destination'"
  }
  if (($robocopyExit -band 8) -ne 0) {
    Write-Warning "Sync-BuildArtifacts: robocopy exit $robocopyExit - some files could not be copied (likely a transient lock); continuing."
  }
  # Do not leak robocopy's nonzero success codes into callers that treat
  # $LASTEXITCODE as pass/fail.
  $global:LASTEXITCODE = 0
}

# Normalize-Version now comes from ContainerHub's WindowsScripts.Shared.psm1 as
# ConvertTo-NormalizedVersion ("Normalize" is not an approved PowerShell verb,
# and an unapproved one in a shared module warns on every import). It sits
# beside the bash twin (version_util.sh --normalize) it has to agree with.

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$containerHubModulesRoot = Join-Path $repoRoot 'ExternalLib\Kataglyphis-ContainerHub\windows\scripts\modules'

$buildModule = Join-Path $containerHubModulesRoot 'WindowsBuild.Common.psm1'
if (-not (Test-Path $buildModule)) {
  throw "Required module not found: $buildModule"
}

Import-Module $buildModule -Force

# WindowsBuild.Common imports WindowsScripts.Shared for ITS OWN internals,
# but nested module imports are module-private in PowerShell - the caller
# never sees Resolve-WorkspacePath, and this script died on exactly that the
# first time CI ever reached it (every earlier lane failure sat upstream).
# Import the shared module directly.
$sharedModule = Join-Path $containerHubModulesRoot 'WindowsScripts.Shared.psm1'
if (-not (Test-Path $sharedModule)) {
  throw "Required module not found: $sharedModule"
}
Import-Module $sharedModule -Force

# Get-OrDefault / Get-ConfigValue live here. They were previously copy-pasted
# into this file, character for character - so a fix to either copy silently
# missed the other. Same nested-import caveat as above: WindowsBuild.Common
# importing this module does not re-export it to us.
$configModule = Join-Path $containerHubModulesRoot 'WindowsConfig.Common.psm1'
if (-not (Test-Path $configModule)) {
  throw "Required module not found: $configModule"
}
Import-Module $configModule -Force

# The whole MSIX path below was a second implementation of this module:
# SDK tool lookup, XML escaping, token expansion and the transparent-PNG
# placeholder all already live here.
$msixModule = Join-Path $containerHubModulesRoot 'WindowsMsix.Common.psm1'
if (-not (Test-Path $msixModule)) {
  throw "Required module not found: $msixModule"
}
Import-Module $msixModule -Force

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

    # No try/catch around these two. Swallowing them into a warning is how
    # `licenses FAILED` shipped unnoticed: cargo-deny rejected xxhash-rust's
    # BSL-1.0 on every single build and the step still reported success. A
    # security gate that cannot fail is not a gate. Findings belong in
    # deny.toml (allow the licence, or ignore the advisory with a reason) -
    # not in a catch block here.
    Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('audit') | Out-Null
    Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('deny', 'check', 'advisories', 'licenses', 'bans', 'sources') | Out-Null
    } | Out-Null

    # NEVER `rustup component add` here. The image's rustup is offline - its
    # dist server is a file:// mirror that setup-rust-toolchain.ps1 deletes
    # after installing - so the call can only ever fail, and the previous
    # skip-on-failure made both gates decorative: each finished in ~0.1s and
    # reported success, so neither had run even once (measured 2026-08-07).
    # Call the components directly and let a failure BE a failure. If they are
    # missing the image is wrong, not the code; ContainerHub now installs them
    # with `-c rustfmt -c clippy` and asserts them at image-build time.
    Invoke-BuildStep -Context $context -StepName 'Format Check' -Critical -Script {
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters @('fmt', '--all', '--', '--check') | Out-Null
    } | Out-Null

    # Default features on purpose. --all-features pulls onnxruntime_cuda and
    # onnxruntime_directml, which need vendor SDKs this image has not got; the
    # feature-matrix CI job lints the combinations that are actually buildable.
    Invoke-BuildStep -Context $context -StepName 'Linting (cargo clippy)' -Critical -Script {
      $clippyParams = @('clippy', '--all-targets')
      if (-not [string]::IsNullOrWhiteSpace($cargoFeatures)) {
        $clippyParams += @('--features', $cargoFeatures)
      }
      $clippyParams += @('--', '-D', 'warnings')
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters $clippyParams | Out-Null
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

  # Invoke-BuildStep, NOT Invoke-BuildOptional. The latter is
  # `try { & $Script } catch { Write-BuildLogWarning }` and never registers the
  # step with the context, so packaging failures appeared in neither the
  # SUCCEEDED nor the FAILED list - a run with a broken MSI still printed
  # "7 steps, 7 succeeded, 0 failed (100% success rate)". If packaging was
  # asked for and it breaks, that is a failure and the summary must say so.
  if (-not $SkipMsix) {
    Invoke-BuildStep -Context $context -StepName 'MSIX Packaging' -Critical -Script {
      # Resolve-WindowsSdkToolPath, not a local recursive scan of the Kits
      # tree: it takes VsDevCmd's WindowsSdkVerBinPath / WindowsSDKVersion
      # first and only falls back to scanning, newest version first. The 8.3
      # short path the old local helper returned is not needed - the path goes
      # to Invoke-BuildExternal as a -File argument, never spliced into a
      # command line, so spaces are already safe.
      $makeappxPath = Resolve-WindowsSdkToolPath `
        -ToolName 'makeappx.exe' `
        -OverridePath (Get-ConfigValue -Config $config -Path 'Msix.MakeAppxPath')
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
      $resolvedVersion = ConvertTo-NormalizedVersion $resolvedVersion

      # CARGO_TARGET_DIR is a standard cargo variable and is commonly ABSOLUTE
      # (the in-container scripts here set C:\ct). PowerShell's Join-Path does
      # not collapse that the way Path.Combine would - `Join-Path 'C:\a' 'C:\b'`
      # yields 'C:\a\C:\b' - and MSIX packaging then died on
      # "The filename, directory name, or volume label syntax is incorrect".
      # Same IsPathRooted idiom this script already uses for the manifest
      # template path.
      $cargoTargetFullPath = if ([System.IO.Path]::IsPathRooted($cargoTargetDir)) {
        $cargoTargetDir
      } else {
        Join-Path $workspacePath $cargoTargetDir
      }
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
        # New-TransparentPng comes from WindowsMsix.Common; it used to be
        # redefined inline right here, inside the else branch.
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

      # Expand-XmlTemplateTokens (WindowsMsix.Common) instead of a chain of
      # `-replace`. That is a BUG FIX, not just deduplication: `-replace`
      # treats its replacement as a substitution TEMPLATE, so a value
      # containing '$&' or '$1' - a description or display name is free text -
      # would be silently rewritten into the manifest. The module uses an
      # ordinal [string].Replace and escapes each value for XML itself.
      $manifestXml = Expand-XmlTemplateTokens -Template $templateContent -TokenMap @{
        '__MSIX_NAME__'                   = $msixName
        '__MSIX_PUBLISHER__'              = $msixPublisher
        '__MSIX_VERSION__'                = $resolvedVersion
        '__MSIX_MIN_VERSION__'            = $msixMinVersion
        '__MSIX_DISPLAY_NAME__'           = $msixDisplayName
        '__MSIX_PUBLISHER_DISPLAY_NAME__' = $msixPublisherDisplayName
        '__MSIX_DESCRIPTION__'            = $msixDescription
        '__EXE_REL_PATH__'                = $exeRelPath
      }
      $manifestXml = Expand-XmlTemplateTokens -Template $manifestXml -TokenMap @{
        '__STORE_LOGO_REL__' = 'Assets/StoreLogo.png'
        '__LOGO44_REL__'     = 'Assets/Square44x44Logo.png'
        '__LOGO150_REL__'    = 'Assets/Square150x150Logo.png'
      }

      Set-Content -Path (Join-Path $msixStaging 'AppxManifest.xml') -Value $manifestXml -Encoding UTF8

      $distDir = Join-Path $workspacePath 'dist\msix'
      New-Item -ItemType Directory -Path $distDir -Force | Out-Null

      $packageFile = Join-Path $distDir "$msixName`_$resolvedVersion`_x64.msix"
      if (Test-Path $packageFile) {
        Remove-Item $packageFile -Force
      }

      Write-BuildLog -Context $context -Message "Creating MSIX package: $packageFile"
      Invoke-BuildExternal -Context $context -File $makeappxPath -Parameters @('pack', '/d', $msixStaging, '/p', $packageFile, '/o') | Out-Null

      # Do not announce a package that is not there. A green tool exit is not
      # proof of delivery - the same rule Test-BuildArtifactsDelivered exists
      # for on the container side.
      if (-not (Test-Path $packageFile)) {
        throw "makeappx reported success but produced no file at $packageFile"
      }
      Write-BuildLogSuccess -Context $context -Message "MSIX package created: $packageFile"
    }
  }

  # MSI Packaging with cargo-wix
  $msiEnabled = Get-ConfigValue -Config $config -Path 'Msi.Enabled'
  if (-not $SkipMsi -and $msiEnabled) {
    Invoke-BuildStep -Context $context -StepName 'MSI Packaging' -Critical -Script {
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

      # cargo-wix looks for WXS files under <package-root>/wix/. With
      # `-p kataglyphis_cli` that is crates/cli/wix/, which does not exist -
      # this repo keeps the single WiX source at the WORKSPACE root. The result
      # was "Error[2] (Generic): There are no WXS files to create an installer"
      # on every run, hidden because the step ran under Invoke-BuildOptional.
      #
      # Msi.WxsFile has been in Build-Windows.config.psd1 all along and was
      # never read; cargo-wix takes the .wxs as a positional INPUT argument, so
      # pass it explicitly rather than relying on directory discovery.
      $wxsRel = Get-OrDefault (Get-ConfigValue -Config $config -Path 'Msi.WxsFile') 'wix/main.wxs'
      $wxsPath = if ([System.IO.Path]::IsPathRooted($wxsRel)) { $wxsRel } else { Join-Path $workspacePath $wxsRel }
      if (-not (Test-Path $wxsPath)) {
        throw "WiX source not found: $wxsPath (Msi.WxsFile = '$wxsRel'). cargo-wix cannot build an installer without it."
      }

      $wixParams = @('wix', '--no-build', '--nocapture', '-p', 'kataglyphis_cli', '--output', $msiFile, $wxsPath)
      Invoke-BuildExternal -Context $context -File 'cargo' -Parameters $wixParams | Out-Null

      if (-not (Test-Path $msiFile)) {
        throw "cargo-wix reported success but produced no file at $msiFile"
      }

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

