#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

timestamp() {
	date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
	echo "[$(timestamp)] [INFO] [$SCRIPT_NAME] $*"
}

log_error() {
	echo "[$(timestamp)] [ERROR] [$SCRIPT_NAME] $*" >&2
}

run_step() {
	local description="$1"
	shift

	log_info "Starting: ${description}"
	if "$@"; then
		log_info "Completed: ${description}"
	else
		local exit_code=$?
		log_error "Failed: ${description} (exit code: ${exit_code})"
		exit "${exit_code}"
	fi
}

log_info "Security checks started"

run_step "Install security tooling (cargo-audit, cargo-deny)" \
	cargo install --locked cargo-audit cargo-deny

run_step "Run vulnerability audit (cargo audit)" \
	cargo audit

run_step "Run policy checks (cargo deny: advisories, licenses, bans, sources)" \
	cargo deny check advisories licenses bans sources

log_info "Security checks completed successfully"
