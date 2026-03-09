#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../ExternalLib/Kataglyphis-ContainerHub/linux/scripts/01-core/logging.sh"

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
    info "Copying images..."
    cp -r ./images ./target/doc/kataglyphis_rustprojecttemplate/
fi

info "Documentation built successfully."
