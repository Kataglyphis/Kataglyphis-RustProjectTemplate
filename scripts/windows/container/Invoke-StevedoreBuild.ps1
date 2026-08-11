<#
.SYNOPSIS
    Builds (and optionally tests) this Rust workspace inside the Kataglyphis
    ContainerHub Windows developer image via Stevedore's docker.exe.

.DESCRIPTION
    Runs cargo for all three profiles -- dev (debug), profile (release +
    debuginfo), release (fat LTO) -- inside the container, then copies the
    artifacts back to <repo>\target\container\<profile> and mirrors them to the
    repo root (<repo>\debug, \profile, \release; gitignored).

    Host quirks this script works around (verified 2026-07-17, see
    ExternalLib\Kataglyphis-ContainerHub\docs\windows-builds.md):
    - Dev Drive (ReFS) sources cannot be bind-mounted unless bindFlt/wcifs are
      allowed on the volume ("Der Dateisystem-Minifilter kann nicht an das
      Entwicklervolume angefügt werden"). The sources are therefore staged to a
      non-Dev-Drive location (default: %LOCALAPPDATA%\Temp) and mounted from
      there. Durable alternative (elevated, then remount):
        fsutil devdrv setfiltersallowed bindFlt, wcifs
    - --isolation process is required for full host CPU count (Hyper-V = 2).
    - All build writes stay container-local (CARGO_TARGET_DIR=C:\ct,
      CARGO_HOME=C:\ch) -- wcifs/bindFlt break create-then-rename on image
      layers and two-path ops on bind mounts. Artifacts come back via plain
      copies through the mount (done by the in-container scripts).
    - The docker CLI intermittently drops its pipe mid-run while the container
      keeps working, so the container is named (not --rm) and this script waits
      on the actual container state, not the client exit code.

.PARAMETER Test
    Also run the full test suite (cargo test --workspace --locked: unit +
    integration + proptest fuzz + doc tests) at the debug profile.

.PARAMETER BuildOnly / TestOnly
    Restrict to one phase (default: build; add -Test for both).

.EXAMPLE
    pwsh -ExecutionPolicy Bypass -File .\scripts\windows\Container\Invoke-StevedoreBuild.ps1 -Test
#>
param(
#requires -Version 7.0

    [string]$Docker = '',
    [string]$Image = 'ghcr.io/kataglyphis/kataglyphis_beschleuniger:winamd64',
    # Scratch root for the in-container scripts and their logs. Small, and on a
    # non-Dev-Drive volume because it is written to constantly.
    [string]$StagingDir = (Join-Path $env:LOCALAPPDATA 'Temp\kataglyphis-rust-container'),
    # Copy the sources to $StagingDir\ws and mount THAT, instead of mounting the
    # repository directly. The direct mount is the default and is what CI-like
    # runs should use; this is the escape hatch for a host where bindFlt refuses
    # the repo volume (see the comment on $ws below for how to tell).
    [switch]$StageSources,
    [switch]$Test,
    [switch]$TestOnly,
    [int]$MemoryGb = 48,
    [string]$ContainerName = 'kata-rust-build'
)

# NB: EAP stays 'Continue' -- native-command stderr handling has shifted across
# PowerShell versions, so this script turns
# terminating errors under 'Stop' (documented ContainerHub trap). Exit codes
# are checked explicitly instead.
$ProgressPreference = 'SilentlyContinue'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path

# Container plumbing comes from ContainerHub, which is the ground truth for it:
# Stevedore's docker.exe lookup, the isolation flags, and the wcifs-tolerant
# container removal were all reimplemented here before.
$containerHubModules = Join-Path $repoRoot 'ExternalLib\Kataglyphis-ContainerHub\windows\scripts\modules'
$reuseModule = Join-Path $containerHubModules 'WindowsContainerBuild.Reuse.psm1'
if (-not (Test-Path $reuseModule)) {
    throw "Required module not found: $reuseModule (run: git submodule update --init --recursive)"
}
Import-Module $reuseModule -Force

# Resolve-DockerExe checks the same candidates this script used to hard-code
# ($env:DOCKER_EXE, both Stevedore locations, then PATH) and throws with an
# install hint instead of a bare Get-Command failure. nerdctl is deliberately
# not a candidate: it talks to containerd, not the Windows lane's engine.
$Docker = Resolve-DockerExe -Override $Docker
Write-Host "Using docker: $Docker"

# ---- workspace: mount the repository itself ----
#
# The repo is bind-mounted straight into the container, Dev Drive or not.
# Verified on this host 2026-08-07 against a ReFS D: - the tree is readable and
# the build runs. The older behaviour (robocopy the sources to
# %LOCALAPPDATA%\Temp and mount the copy) is still available via -StageSources.
#
# The distinction that matters is READ vs WRITE, not the filesystem. Per the
# submodule's docs/windows-builds.md, bindFlt rejects copySync/renameSync with
# errno 3, so create-then-rename through the mount fails - which is why every
# build write already goes to container-local C:\ct and C:\ch. The only thing
# crossing the mount is the artifact copy at the end of rust-build-all.ps1, and
# plain copies do work.
#
# If a host DOES refuse it, `docker run --mount type=bind,source=<repo>,...`
# fails immediately with "Der Dateisystem-Minifilter kann nicht an das
# Entwicklervolume angefügt werden". The permanent fix is one elevated
# `fsutil devdrv setfiltersallowed bindFlt, wcifs` plus a remount; -StageSources
# is the stopgap. Note `fsutil devdrv query` needs elevation itself, so a
# failing query proves nothing - try the mount.
$scratch = Join-Path $StagingDir 'scratch'
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

if ($StageSources) {
    $ws = Join-Path $StagingDir 'ws'
    New-Item -ItemType Directory -Force -Path $ws | Out-Null
    Write-Host "Staging sources -> $ws (-StageSources)"
    robocopy $repoRoot $ws /MIR /XD target .git .vs out dist debug profile release /XF *.msix /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy staging failed ($LASTEXITCODE)" }
} else {
    $ws = $repoRoot
    Write-Host "Mounting repository directly -> $ws"
}

Copy-Item (Join-Path $PSScriptRoot 'rust-build-all.ps1'), (Join-Path $PSScriptRoot 'rust-test-all.ps1') -Destination $scratch -Force

# The in-container scripts import their logging from here. Keep staging it into
# the scratch mount rather than reading it off the workspace mount: that keeps
# the two in-container scripts working identically under -StageSources, whose
# robocopy still has no reason to carry the whole submodule.
$containerLogModule = Join-Path $containerHubModules 'WindowsContainerLog.Common.psm1'
if (-not (Test-Path $containerLogModule)) {
    throw "Required module not found: $containerLogModule"
}
Copy-Item $containerLogModule -Destination $scratch -Force

function Invoke-ContainerScript {
    param([Parameter(Mandatory)][string]$Script, [Parameter(Mandatory)][string]$Label)
    # Remove-BuildContainerSafe, not a bare `docker rm -f`: on this host the
    # wcifs teardown can still hold the container after rm returns, and the
    # helper detects that and says so instead of letting the next `docker run`
    # fail on a name clash.
    [void](Remove-BuildContainerSafe -DockerExe $Docker -Name $ContainerName)
    # --isolation process is required for the full host CPU count (Hyper-V
    # isolation exposes 2). Centralised so this lane cannot drift from the
    # others that need the same flag.
    $isolationArgs = Get-ContainerIsolationArgs -Isolation 'process' -MemoryGb $MemoryGb
    Write-Host "`n==> [$Label] docker run $($isolationArgs -join ' ') --memory ${MemoryGb}g $Image" -ForegroundColor Cyan
    & $Docker run --name $ContainerName @isolationArgs --memory "${MemoryGb}g" `
        --mount "type=bind,source=$ws,target=C:\ws-mnt" `
        --mount "type=bind,source=$scratch,target=C:\host-scratch" `
        $Image pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\host-scratch\$Script"
    $clientExit = $LASTEXITCODE
    # The docker CLI pipe can drop while the container keeps running -- trust
    # the container state, not the client exit code.
    while ($true) {
        $state = & $Docker inspect -f '{{.State.Status}}' $ContainerName 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $state -or $state -ne 'running') { break }
        Write-Host "[$Label] docker client detached (exit $clientExit) but container still running -- waiting..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
    }
    $exitCode = & $Docker inspect -f '{{.State.ExitCode}}' $ContainerName 2>$null
    [void](Remove-BuildContainerSafe -DockerExe $Docker -Name $ContainerName)
    if ("$exitCode" -ne '0') { throw "[$Label] container run failed (exit $exitCode) -- see $scratch logs" }
    Write-Host "[$Label] OK" -ForegroundColor Green
}

if (-not $TestOnly) {
    Invoke-ContainerScript -Script 'rust-build-all.ps1' -Label 'build'
    # With the repository mounted directly, the container already wrote into
    # target\container - copying it onto itself would be a /MIR of a directory
    # over itself, which robocopy refuses. Only the staged path needs this.
    if ($StageSources) {
        robocopy (Join-Path $ws 'target\container') (Join-Path $repoRoot 'target\container') /MIR /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw 'artifact copy-back failed' }
    }
    if (-not (Test-Path (Join-Path $repoRoot 'target\container'))) {
        throw "the build reported success but produced no target\container - nothing was delivered through the mount"
    }
    foreach ($p in 'debug', 'profile', 'release') {
        robocopy (Join-Path $repoRoot "target\container\$p") (Join-Path $repoRoot $p) /MIR /NFL /NDL /NJH /NJS | Out-Null
    }
    Write-Host "Artifacts: $repoRoot\target\container\{debug,profile,release} (+ root mirrors)" -ForegroundColor Green
}
if ($Test -or $TestOnly) {
    Invoke-ContainerScript -Script 'rust-test-all.ps1' -Label 'test'
}
Write-Host "`nDone. Logs: $scratch\in-container-*.log"

