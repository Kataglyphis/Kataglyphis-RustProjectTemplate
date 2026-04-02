#!/bin/bash
set -e

echo "Starte GUI mit Person Detection..."
cargo run --release -p kataglyphis_cli --features="gui_unix" --bin kataglyphis_rustprojecttemplate -- gui
