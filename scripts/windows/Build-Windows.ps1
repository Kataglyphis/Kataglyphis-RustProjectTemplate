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

  # NOTE: there is deliberately no -Configurations here. One used to be
  # declared and nothing ever read it, so `-Configurations gui_windows` was
  # accepted and silently ignored. The feature-matrix concept lives in
  # Invoke-WindowsConfigMatrix.ps1, which implements it properly and drives
  # Run-AppProfiles.ps1 per configuration. Use that script instead of
  # reintroducing the parameter here.
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

# Normalize-Version now comes from ContainerHub's WindowsScripts.Shared.psm1 as
# ConvertTo-NormalizedVersion ("Normalize" is not an approved PowerShell verb,
# and an unapproved one in a shared module warns on every import). It sits
# beside the bash twin (version_util.sh --normalize) it has to agree with.

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
# One bootstrap resolves every module ContainerHub-first, with
# scripts/windows/modules/ as the project-specific fallback. It replaces four
# near-identical hard-coded import blocks; a module that moves upstream is now
# picked up without editing this script, and a missing submodule reports the
# exact `git submodule update` command instead of a bare path.
#
# Import-BuildModule pulls WindowsScripts.Shared in whether or not it is listed,
# which is what the four blocks below had each worked around by hand: a nested
# Import-Module inside a .psm1 binds into THAT module's private scope and never
# reaches this session, so importing only WindowsBuild.Common left
# Resolve-WorkspacePath undefined.
. (Join-Path $PSScriptRoot 'Resolve-BuildModule.ps1')

Import-BuildModule @(
  'WindowsScripts.Shared'   # Assert-Command, ConvertTo-NormalizedVersion, Resolve-WorkspacePath
  'WindowsBuild.Common'     # build context/log/step primitives, Sync-BuildArtifacts
  'WindowsConfig.Common'    # Get-OrDefault, Get-ConfigValue
  'WindowsMsix.Common'      # Resolve-WindowsSdkToolPath, ConvertTo-XmlSafeText, New-TransparentImage
)

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
        $logoPath = Join-Path $workspacePath 'third_party\ContainerHub\images\logo.png'
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

  # MSI packaging with WiX Toolset v4, driving wix.exe directly.
  #
  # NOT cargo-wix. 0.3.9 is its newest release and it shells out to WiX v3's
  # candle.exe + light.exe, neither of which exists here: ContainerHub installs
  # WiX 4.0.6 as a dotnet tool (a single wix.exe) in
  # windows/scripts/setup-scoop-tools.ps1 and points WIX=C:\WiX at it in
  # windows/Dockerfile.base. Every MSI run therefore died with
  # "The compiler application ('candle') does not exist at the 'C:\WiX' path",
  # which went unnoticed while this step still ran as optional. Calling wix.exe
  # keeps the image on one WiX generation instead of adding a second.
  $msiEnabled = Get-ConfigValue -Config $config -Path 'Msi.Enabled'
  if (-not $SkipMsi -and $msiEnabled) {
    Invoke-BuildStep -Context $context -StepName 'MSI Packaging' -Critical -Script {
      $wixExe = $null
      if (-not [string]::IsNullOrWhiteSpace($env:WIX)) {
        $candidate = Join-Path $env:WIX 'wix.exe'
        if (Test-Path $candidate) { $wixExe = $candidate }
      }
      if (-not $wixExe) {
        $wixExe = (Get-Command 'wix.exe' -ErrorAction SilentlyContinue).Source
      }
      if (-not $wixExe) {
        throw "WiX v4 (wix.exe) not found. Looked under `$env:WIX ('$env:WIX') and on PATH. The container image installs it via ContainerHub's windows/scripts/setup-scoop-tools.ps1."
      }
      Write-BuildLog -Context $context -Message "Using WiX: $wixExe"

      $resolvedVersion = $msixVersion
      $versionFile = Join-Path $workspacePath 'version.txt'
      if (Test-Path $versionFile) {
        $resolvedVersion = (Get-Content -Path $versionFile).Trim()
      }
      if ($resolvedVersion -match '^v') {
        $resolvedVersion = $resolvedVersion.Substring(1)
      }
      # MSI ProductVersion is limited to major.minor.build
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

      # Msi.WxsFile has been in Build-Windows.config.psd1 all along and was
      # never read - the old cargo-wix call let it look for WXS files under
      # crates/cli/wix/, which does not exist, so it also failed with
      # "There are no WXS files to create an installer".
      $wxsRel = Get-OrDefault (Get-ConfigValue -Config $config -Path 'Msi.WxsFile') 'wix/main.wxs'
      $wxsPath = if ([System.IO.Path]::IsPathRooted($wxsRel)) { $wxsRel } else { Join-Path $workspacePath $wxsRel }
      if (-not (Test-Path $wxsPath)) {
        throw "WiX source not found: $wxsPath (Msi.WxsFile = '$wxsRel')."
      }

      # Same IsPathRooted guard as the MSIX step: CARGO_TARGET_DIR is usually
      # absolute in the container (C:\ct), and Join-Path would mangle it.
      $msiCargoTargetFullPath = if ([System.IO.Path]::IsPathRooted($cargoTargetDir)) {
        $cargoTargetDir
      } else {
        Join-Path $workspacePath $cargoTargetDir
      }
      $msiExePath = Join-Path (Join-Path $msiCargoTargetFullPath 'release') "$binary.exe"
      if (-not (Test-Path $msiExePath)) {
        throw "Expected executable not found: $msiExePath"
      }

      $licenseRel = Get-OrDefault (Get-ConfigValue -Config $config -Path 'Msi.LicenseFile') 'wix/License.rtf'
      $licenseRtf = if ([System.IO.Path]::IsPathRooted($licenseRel)) { $licenseRel } else { Join-Path $workspacePath $licenseRel }
      if (-not (Test-Path $licenseRtf)) {
        throw "License file not found: $licenseRtf (Msi.LicenseFile = '$licenseRel', referenced by $wxsPath)."
      }

      # Msi.ProductName and Msi.Manufacturer were declared in the config and
      # never read, while those same two strings sat hard-coded in the WXS --
      # two sources of truth, where editing the config silently did nothing.
      # They are preprocessor variables now, so the config is the only one.
      $msiProductName = Get-OrDefault (Get-ConfigValue -Config $config -Path 'Msi.ProductName') $msixDisplayName
      $msiManufacturer = Get-OrDefault (Get-ConfigValue -Config $config -Path 'Msi.Manufacturer') $msixPublisherDisplayName
      if ([string]::IsNullOrWhiteSpace($msiProductName)) {
        throw "Msi.ProductName is empty and Msix.DisplayName gave no fallback; $wxsPath requires it."
      }
      if ([string]::IsNullOrWhiteSpace($msiManufacturer)) {
        throw "Msi.Manufacturer is empty and Msix.PublisherDisplayName gave no fallback; $wxsPath requires it."
      }

      # The WXS takes every moving value as a preprocessor variable so it never
      # has to assume a target\release next to the workspace root, and never
      # duplicates a string the config already owns.
      $wixParams = @(
        'build',
        '-arch', 'x64',
        '-ext', 'WixToolset.UI.wixext',
        '-d', "Version=$resolvedVersion",
        '-d', "ExeSource=$msiExePath",
        '-d', "LicenseRtf=$licenseRtf",
        '-d', "ProductName=$msiProductName",
        '-d', "Manufacturer=$msiManufacturer",
        '-out', $msiFile,
        $wxsPath
      )
      Invoke-BuildExternal -Context $context -File $wixExe -Parameters $wixParams | Out-Null

      if (-not (Test-Path $msiFile)) {
        throw "wix.exe reported success but produced no file at $msiFile"
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

