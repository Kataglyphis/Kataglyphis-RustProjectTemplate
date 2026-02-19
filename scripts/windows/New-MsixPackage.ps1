param(
    [string]$Workspace = (Resolve-Path "$PSScriptRoot\..\..\").Path,
    [string]$Binary = "kataglyphis_rustprojecttemplate",
    [string]$Features = "gui_windows,onnxruntime_directml",
    [string]$PackageName = "Kataglyphis.RustProjectTemplate",
    [string]$Publisher = "CN=Kataglyphis",
    [string]$PublisherDisplayName = "Kataglyphis",
    [string]$DisplayName = "Kataglyphis RustProjectTemplate",
    [string]$Description = "Kataglyphis Rust desktop application",
    [string]$Version = "0.1.0.0",
    [string]$OutputDirectory = "dist\msix",
    [string]$CargoTargetDir = "target-msix",
    [string]$CertificatePath,
    [string]$CertificatePassword,
    [switch]$CreateTestCertificate,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

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
        if (-not (Test-Path $root)) {
            continue
        }

        $candidates = Get-ChildItem -Path $root -Recurse -Filter ("{0}.exe" -f $Name) -ErrorAction SilentlyContinue |
            Where-Object {
                $_.FullName -match "\\x64\\" -or $_.FullName -match "/x64/"
            } |
            Sort-Object FullName -Descending

        if ($candidates -and $candidates.Count -gt 0) {
            return $candidates[0].FullName
        }
    }

    return $null
}

function Normalize-Version([string]$RawVersion) {
    $segments = $RawVersion.Split('.')
    if ($segments.Count -eq 3) {
        return "$RawVersion.0"
    }
    if ($segments.Count -ne 4) {
        throw "Version '$RawVersion' is invalid. Use Major.Minor.Build or Major.Minor.Build.Revision"
    }
    return $RawVersion
}

function New-PasswordSecureString([string]$Password) {
    if ([string]::IsNullOrWhiteSpace($Password)) {
        throw "A non-empty -CertificatePassword is required for certificate export/signing"
    }
    return ConvertTo-SecureString -String $Password -AsPlainText -Force
}

Assert-Command -Name "cargo" -InstallHint "Install Rust toolchain via rustup"

$makeappxExe = Resolve-Executable -Name "makeappx"
if ([string]::IsNullOrWhiteSpace($makeappxExe)) {
    throw "makeappx not found. Install Windows SDK / MSIX Packaging Tools"
}

$signtoolExe = Resolve-Executable -Name "signtool"
if ([string]::IsNullOrWhiteSpace($signtoolExe)) {
    throw "signtool not found. Install Windows SDK / Signing Tools"
}

Write-Host "Using makeappx: $makeappxExe"
Write-Host "Using signtool: $signtoolExe"

$Workspace = (Resolve-Path $Workspace).Path
$Version = Normalize-Version $Version

$resolvedCargoTargetDir = Join-Path $Workspace $CargoTargetDir
$env:CARGO_TARGET_DIR = $resolvedCargoTargetDir
$env:CARGO_INCREMENTAL = "0"

Push-Location $Workspace
try {
    if (-not $SkipBuild) {
        Write-Host "Building release binary..."
        cargo build --release --bin $Binary --features $Features --jobs 1
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    }

    $releaseDir = Join-Path $resolvedCargoTargetDir "release"
    $exePath = Join-Path $releaseDir "$Binary.exe"
    if (-not (Test-Path $exePath)) {
        throw "Binary not found: $exePath"
    }

    $outDir = Join-Path $Workspace $OutputDirectory
    $stagingRoot = Join-Path $outDir "staging"
    $assetsDir = Join-Path $stagingRoot "Assets"

    if (Test-Path $stagingRoot) {
        Remove-Item $stagingRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null

    Copy-Item $exePath -Destination $stagingRoot -Force

    Get-ChildItem -Path $releaseDir -Filter "*.dll" -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName -Destination $stagingRoot -Force }

    $resourcesSource = Join-Path $Workspace "resources"
    if (Test-Path $resourcesSource) {
        Copy-Item $resourcesSource -Destination (Join-Path $stagingRoot "resources") -Recurse -Force
    }

    $logoSource = Join-Path $Workspace "images\logo.png"
    if (-not (Test-Path $logoSource)) {
        throw "Logo file not found at $logoSource"
    }

    Copy-Item $logoSource -Destination (Join-Path $assetsDir "StoreLogo.png") -Force
    Copy-Item $logoSource -Destination (Join-Path $assetsDir "Square44x44Logo.png") -Force
    Copy-Item $logoSource -Destination (Join-Path $assetsDir "Square150x150Logo.png") -Force
    Copy-Item $logoSource -Destination (Join-Path $assetsDir "Wide310x150Logo.png") -Force

    $manifestTemplatePath = Join-Path $Workspace "packaging\msix\AppxManifest.template.xml"
    if (-not (Test-Path $manifestTemplatePath)) {
        throw "Manifest template not found: $manifestTemplatePath"
    }

    $manifestContent = Get-Content $manifestTemplatePath -Raw
    $manifestContent = $manifestContent.Replace("__PACKAGE_NAME__", $PackageName)
    $manifestContent = $manifestContent.Replace("__PUBLISHER__", $Publisher)
    $manifestContent = $manifestContent.Replace("__VERSION__", $Version)
    $manifestContent = $manifestContent.Replace("__DISPLAY_NAME__", $DisplayName)
    $manifestContent = $manifestContent.Replace("__PUBLISHER_DISPLAY_NAME__", $PublisherDisplayName)
    $manifestContent = $manifestContent.Replace("__DESCRIPTION__", $Description)
    $manifestContent = $manifestContent.Replace("__EXECUTABLE__", "$Binary.exe")
    Set-Content -Path (Join-Path $stagingRoot "AppxManifest.xml") -Value $manifestContent -Encoding utf8

    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $packageFile = Join-Path $outDir ("{0}_{1}_x64.msix" -f $PackageName, $Version)
    if (Test-Path $packageFile) {
        Remove-Item $packageFile -Force
    }

    Write-Host "Packing MSIX..."
    & $makeappxExe pack /d $stagingRoot /p $packageFile /o | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "makeappx pack failed" }

    if ($CreateTestCertificate) {
        if ([string]::IsNullOrWhiteSpace($CertificatePath)) {
            $CertificatePath = Join-Path $outDir "$PackageName.testcert.pfx"
        }

        if ([string]::IsNullOrWhiteSpace($CertificatePassword)) {
            throw "-CreateTestCertificate requires -CertificatePassword"
        }

        Write-Host "Creating self-signed test certificate..."
        $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $Publisher -CertStoreLocation "Cert:\CurrentUser\My"
        $securePassword = New-PasswordSecureString -Password $CertificatePassword
        Export-PfxCertificate -Cert $cert -FilePath $CertificatePath -Password $securePassword | Out-Null

        Write-Host "Signing package with generated test certificate..."
        & $signtoolExe sign /fd SHA256 /f $CertificatePath /p $CertificatePassword $packageFile | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "signtool sign failed for generated certificate" }
    }
    elseif (-not [string]::IsNullOrWhiteSpace($CertificatePath)) {
        if ([string]::IsNullOrWhiteSpace($CertificatePassword)) {
            throw "-CertificatePassword is required when -CertificatePath is provided"
        }

        Write-Host "Signing package with provided certificate..."
        & $signtoolExe sign /fd SHA256 /f $CertificatePath /p $CertificatePassword $packageFile | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "signtool sign failed" }
    }
    else {
        Write-Warning "MSIX created but not signed. Installation will typically fail until it is signed."
    }

    Write-Host "MSIX package ready: $packageFile"
}
finally {
    Pop-Location
}
