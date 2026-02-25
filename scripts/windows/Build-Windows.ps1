<#
.SYNOPSIS
  Build script for running inside the Windows container. Matches the behavior of the inline script
  previously passed via -EncodedCommand.

.DESCRIPTION
  - Ensures Scoop shims are in PATH (common container user: ContainerAdministrator).
  - Verifies rustup/cargo install.
  - Runs cargo audit and cargo deny security checks.
  - Adds rustfmt & clippy components.
  - Runs formatting check, clippy (treat warnings as errors), tests, benches and release build.
  - Accepts optional parameters or reads environment variables BINARY and VERSION.

.EXAMPLE
  # from inside container
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Build-Windows.ps1

  # pass parameters
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Build-Windows.ps1 -Workspace C:\workspace -Binary mybin -Version 1.2.3
#>

param(
    [string]$Workspace = $env:WORKSPACE,         # optional working directory; defaults to current if not provided
    [string]$Binary    = $env:BINARY,            # optional - read from environment if present
    [string]$Version   = $env:VERSION            # optional - read from environment if present
)

# Fail fast on any error
$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host "==== $text ===="
}

try {
    # If a workspace was provided, switch there
    if ($null -ne $Workspace -and $Workspace -ne "") {
        Write-Host "Changing directory to workspace: $Workspace"
        Set-Location -Path $Workspace
    } else {
        Write-Host "No Workspace parameter provided; using current directory: $(Get-Location)"
    }

    # Ensure Scoop shims are in PATH (common path inside many Windows containers)
    $scoopShims = "C:\Users\ContainerAdministrator\scoop\shims"
    if (-not ($env:PATH -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ieq $scoopShims })) {
        Write-Host "Prepending scoop shims to PATH: $scoopShims"
        $env:PATH = "$scoopShims;$env:PATH"
    } else {
        Write-Host "Scoop shims already in PATH."
    }

    Write-Header "Environment"
    Write-Host "Workspace:    $((Get-Location).Path)"
    Write-Host "BINARY:       $Binary"
    Write-Host "VERSION:      $Version"
    Write-Host "PATH (truncated): $($env:PATH.Substring(0, [Math]::Min(200, $env:PATH.Length)))`n"

    # Verify Rust toolchain presence
    Write-Header "Verifying Rust installation..."
    & rustup --version
    & cargo --version

    # Security checks
    Write-Header "Installing security tools and running cargo audit / cargo deny"
    & cargo install --locked cargo-audit cargo-deny
    & cargo audit
    & cargo deny check advisories licenses bans sources

    # Add rustfmt component and run formatting check
    Write-Header "Ensuring rustfmt component and running cargo fmt --check"
    & rustup component add rustfmt
    & cargo fmt --all -- --check

    # Add clippy and run linter
    Write-Header "Ensuring clippy component and running cargo clippy"
    & rustup component add clippy
    & cargo clippy --all-targets --all-features -- -D warnings

    # Run tests
    Write-Header "Running cargo test (verbose)"
    & cargo test --all --verbose

    # Run benchmarks (note: requires benches configured and possibly nightly for some setups)
    Write-Header "Running cargo bench"
    & cargo bench

    # Final release build
    Write-Header "Running cargo build --release"
    & cargo build --release

    Write-Header "Build script completed successfully"
    exit 0
}
catch {
    Write-Host ""
    Write-Error "Build script failed: $($_.Exception.Message)"
    if ($_.InvocationInfo -ne $null) {
        Write-Host "At: $($_.InvocationInfo.ScriptName) Line: $($_.InvocationInfo.ScriptLineNumber)"
    }
    # For CI it's useful to return a non-zero exit code
    exit 1
}
