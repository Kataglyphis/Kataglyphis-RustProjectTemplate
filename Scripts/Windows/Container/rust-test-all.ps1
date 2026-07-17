# Runs INSIDE the Windows container: full `cargo test` (debug profile) for the
# workspace -- unit tests, integration tests (tests/integration.rs), and the
# proptest fuzz suite (tests/fuzz_test.rs), plus doc tests.
# Same wcifs-safe layout as rust-build-all.ps1: writes only to C:\ct / C:\ch,
# everything logged to the mounted C:\host-scratch.
$ProgressPreference = 'SilentlyContinue'
$log = 'C:\host-scratch\in-container-test.log'
Remove-Item $log -Force -ErrorAction SilentlyContinue
function Say([string]$msg) { Write-Host $msg; Add-Content -Path $log -Value $msg }
function Run-Logged([string]$cmdline) {
    cmd /c "$cmdline 2>&1" | ForEach-Object { Write-Host $_; Add-Content -Path $log -Value $_ }
    return $LASTEXITCODE
}

Say "=== Rust container test run (debug): unit + integration + fuzz(proptest) + doc ==="
Say "cpus: $env:NUMBER_OF_PROCESSORS"
[void](Run-Logged 'rustc -vV')

New-Item -ItemType Directory -Force -Path C:\ct, C:\ch | Out-Null
$env:CARGO_TARGET_DIR = 'C:\ct'
$env:CARGO_HOME = 'C:\ch'
Set-Location C:\ws-mnt

$code = Run-Logged 'cargo test --workspace --locked'
if ($code -ne 0) { Say "TESTS FAILED (exit $code)"; exit $code }
Say 'ALL TESTS PASSED'
exit 0
