#!/usr/bin/env bash
set -euo pipefail

AUTO_YES="${1:-}"

echo "[1/4] Updating lockfile packages..."
cargo update

echo "[2/4] Ensuring cargo-edit is installed..."
cargo install cargo-edit

echo "[3/4] Running dry-run upgrade preview..."
cargo upgrade --dry-run --verbose

proceed_upgrade() {
	echo "[4/4] Applying incompatible upgrades..."
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
			echo "Upgrade cancelled by user after dry-run."
			;;
	esac
else
	echo "Non-interactive shell detected. Skipping real upgrade. Use --yes to proceed automatically."
fi