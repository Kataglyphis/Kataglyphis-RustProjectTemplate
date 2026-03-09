#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../ExternalLib/Kataglyphis-ContainerHub/linux/scripts/01-core/logging.sh"

AUTO_YES="${1:-}"

info "[1/4] Updating lockfile packages..."
cargo update

info "[2/4] Ensuring cargo-edit is installed..."
cargo install cargo-edit

info "[3/4] Running dry-run upgrade preview..."
cargo upgrade --dry-run --verbose

proceed_upgrade() {
	info "[4/4] Applying incompatible upgrades..."
	cargo upgrade --incompatible --pinned
}

if [[ "$AUTO_YES" == "--yes" ]]; then
	proceed_upgrade
	exit 0
fi

if [[ -t 0 ]]; then
	read -r -p "Proceed with real upgrade now? [y/N]: " answer
	case "$answer" in
		y|Y|yes|YES)
			proceed_upgrade
			;;
		*)
			warn "Upgrade cancelled by user after dry-run."
			;;
	esac
else
	warn "Non-interactive shell detected. Skipping real upgrade. Use --yes to proceed automatically."
fi
