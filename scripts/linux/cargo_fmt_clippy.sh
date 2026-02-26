rustup component add rustfmt && cargo fmt --all -- --check
rustup component add clippy
cargo clippy --all-targets --all-features -- -D warnings