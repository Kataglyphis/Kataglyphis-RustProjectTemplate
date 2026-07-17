# Runs INSIDE the Windows container (Windows PowerShell 5.1, VsDevCmd env via entrypoint).
# Builds the workspace in dev (debug), profile, and release, then copies artifacts
# back to the bind-mounted repo. All build writes go to fresh container-local dirs
# (C:\ct, C:\ch) to dodge the wcifs/bindFlt rename bugs on this host/base skew
# (see Kataglyphis-ContainerHub docs/windows-builds.md, run-side wcifs symptoms).
# NB: EAP stays 'Continue' -- PS 5.1 + native stderr under 'Stop' is a known trap.
$ProgressPreference = 'SilentlyContinue'

# Persist ALL output to the mounted scratch dir -- the docker CLI pipe drops
# intermittently on this host, so console logs alone can be lost.
$log = 'C:\host-scratch\in-container-build.log'
Remove-Item $log -Force -ErrorAction SilentlyContinue
function Say([string]$msg) { Write-Host $msg; Add-Content -Path $log -Value $msg }
function Run-Logged([string]$cmdline) {
    # cmd wrapper: merges native stderr without PS 5.1 ErrorRecord mangling
    cmd /c "$cmdline 2>&1" | ForEach-Object { Write-Host $_; Add-Content -Path $log -Value $_ }
    return $LASTEXITCODE
}

Say "=== Rust container build: debug / profile / release ==="
Say "cpus: $env:NUMBER_OF_PROCESSORS"
[void](Run-Logged 'rustc -vV')
if ((Run-Logged 'cargo --version') -ne 0) { Say 'FATAL: cargo not usable'; exit 1 }

New-Item -ItemType Directory -Force -Path C:\ct, C:\ch | Out-Null
$env:CARGO_TARGET_DIR = 'C:\ct'
$env:CARGO_HOME = 'C:\ch'

Set-Location C:\ws-mnt
if (-not (Test-Path .\Cargo.toml)) { Write-Host 'FATAL: workspace mount C:\ws-mnt has no Cargo.toml'; exit 1 }

$profiles = @(
    @{ Name = 'debug';   Args = @('build', '--workspace', '--locked') },
    @{ Name = 'profile'; Args = @('build', '--workspace', '--locked', '--profile', 'profile') },
    @{ Name = 'release'; Args = @('build', '--workspace', '--locked', '--release') }
)

foreach ($p in $profiles) {
    Write-Host "`n==> cargo $($p.Args -join ' ')"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & cargo $p.Args
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($p.Name) build (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    Write-Host ("<== {0} OK in {1:mm\:ss}" -f $p.Name, $sw.Elapsed)
}

# Copy artifacts to the mounted repo. Plain copies are verified to work on bind
# mounts on this host; renames/two-path ops are not -- so cmd copy, no Move-Item.
foreach ($p in $profiles) {
    $src = Join-Path 'C:\ct' $p.Name
    $dst = "C:\ws-mnt\target\container\$($p.Name)"
    cmd /c "if not exist $dst mkdir $dst" | Out-Null
    Get-ChildItem $src -File | Where-Object { $_.Extension -match '^\.(exe|dll|pdb|lib)$' } | ForEach-Object {
        cmd /c "copy /y ""$($_.FullName)"" ""$dst"" >nul"
        if ($LASTEXITCODE -ne 0) { Write-Host "WARN: copy failed for $($_.Name)" }
    }
    Write-Host "`n$($p.Name) artifacts -> target\container\$($p.Name):"
    Get-ChildItem $dst -File | ForEach-Object { Write-Host ("  {0,14:n0}  {1}" -f $_.Length, $_.Name) }
}

Write-Host "`nALL BUILDS SUCCEEDED"
exit 0
