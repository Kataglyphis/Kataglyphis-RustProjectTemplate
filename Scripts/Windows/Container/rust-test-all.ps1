# Runs INSIDE the Windows container: full `cargo test` (debug profile) for the
# workspace -- unit tests, integration tests (tests/integration.rs), and the
# proptest fuzz suite (tests/fuzz_test.rs), plus doc tests.
# Same wcifs-safe layout as rust-build-all.ps1: writes only to C:\ct / C:\ch,
# everything logged to the mounted C:\host-scratch.
#requires -Version 7.0

$ProgressPreference = 'SilentlyContinue'
# See rust-build-all.ps1: the driver stages this module into the scratch mount
# because ExternalLib is excluded from the sources copied into the container.
Import-Module 'C:\host-scratch\WindowsContainerLog.Common.psm1' -Force
Start-ContainerLog -Path 'C:\host-scratch\in-container-test.log'

Write-ContainerLog "=== Rust container test run (debug): unit + integration + fuzz(proptest) + doc ==="
Write-ContainerLog "cpus: $env:NUMBER_OF_PROCESSORS"
[void](Invoke-ContainerLoggedCommand 'rustc -vV')

New-Item -ItemType Directory -Force -Path C:\ct, C:\ch | Out-Null
$env:CARGO_TARGET_DIR = 'C:\ct'
$env:CARGO_HOME = 'C:\ch'
Set-Location C:\ws-mnt

$code = Invoke-ContainerLoggedCommand 'cargo test --workspace --locked'
if ($code -ne 0) { Write-ContainerLog "TESTS FAILED (exit $code)"; exit $code }
Write-ContainerLog 'ALL TESTS PASSED'
exit 0

