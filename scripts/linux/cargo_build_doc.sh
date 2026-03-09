#!/bin/bash
set -e

# Combine CSS files to create a custom rustdoc theme
if [ -f "./ExternalLib/Kataglyphis-ContainerHub/docs/_static/css/custom.css" ] && [ -f "./resources/web/rustdoc-mapping.css" ]; then
    cat ./ExternalLib/Kataglyphis-ContainerHub/docs/_static/css/custom.css ./resources/web/rustdoc-mapping.css > ./combined-theme.css
    export RUSTDOCFLAGS="--extend-css ./combined-theme.css"
fi

# Generate documentation without dependencies
cargo doc --no-deps

# Provide a redirect from the root doc directory to the crate's docs
cp ./resources/web/redirect/index.html ./target/doc/

# Copy logo and images into the doc directory
if [ -f "./images/logo.png" ]; then
    cp ./images/logo.png ./target/doc/logo.png
fi
if [ -d "./images" ]; then
    cp -r ./images ./target/doc/kataglyphis_rustprojecttemplate/
fi
