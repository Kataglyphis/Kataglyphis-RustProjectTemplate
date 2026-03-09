#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../ExternalLib/Kataglyphis-ContainerHub/linux/scripts/01-core/logging.sh"

info "Adding rustfmt component and checking formatting..."
rustup component add rustfmt
cargo fmt --all -- --check

info "Adding clippy component and running clippy checks..."
rustup component add clippy
cargo clippy --all-targets --all-features -- -D warnings

info "Formatting and clippy checks completed successfully."
