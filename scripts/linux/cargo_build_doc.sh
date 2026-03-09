#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../ExternalLib/Kataglyphis-ContainerHub/linux/scripts/01-core/logging.sh"

# Extract crate name from Cargo.toml and format for rustdoc directory (replace dashes with underscores)
CRATE_NAME=$(grep -E '^name\s*=' Cargo.toml | head -n 1 | awk -F'"' '{print $2}')
if [ -z "$CRATE_NAME" ]; then
    err "Failed to extract crate name from Cargo.toml"
    exit 1
fi
CRATE_DIR_NAME="${CRATE_NAME//-/_}"
info "Detected crate name: $CRATE_NAME (doc dir: $CRATE_DIR_NAME)"

# Combine CSS files to create a custom rustdoc theme
if [ -f "./ExternalLib/Kataglyphis-ContainerHub/docs/_static/css/custom.css" ] && [ -f "./resources/web/rustdoc-mapping.css" ]; then
    info "Combining CSS files for custom rustdoc theme..."
    cat ./ExternalLib/Kataglyphis-ContainerHub/docs/_static/css/custom.css ./resources/web/rustdoc-mapping.css > ./combined-theme.css
    export RUSTDOCFLAGS="--extend-css ./combined-theme.css"
fi

# Generate documentation without dependencies
info "Generating rust documentation (cargo doc --no-deps)..."
cargo doc --no-deps

# Provide a redirect from the root doc directory to the crate's docs
info "Providing redirect page..."
cp ./resources/web/redirect/index.html ./target/doc/

# Copy logo and images into the doc directory
if [ -f "./images/logo.png" ]; then
    info "Copying logo..."
    cp ./images/logo.png ./target/doc/logo.png
fi

if [ -d "./images" ]; then
    info "Copying images to target/doc/$CRATE_DIR_NAME/ ..."
    cp -r ./images "./target/doc/$CRATE_DIR_NAME/"
fi

info "Documentation built successfully."
