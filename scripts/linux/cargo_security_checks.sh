#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../ExternalLib/Kataglyphis-ContainerHub/linux/scripts/01-core/logging.sh"

run_step() {
	local description="$1"
	shift

	info "Starting: ${description}"
	if "$@"; then
		info "Completed: ${description}"
	else
		local exit_code=$?
		err "Failed: ${description} (exit code: ${exit_code})"
		exit "${exit_code}"
	fi
}

info "Security checks started"

run_step "Install security tooling (cargo-audit, cargo-deny)" \
	cargo install --locked cargo-audit cargo-deny

run_step "Run vulnerability audit (cargo audit)" \
	cargo audit

run_step "Run policy checks (cargo deny: advisories, licenses, bans, sources)" \
	cargo deny check advisories licenses bans sources

info "Security checks completed successfully"
