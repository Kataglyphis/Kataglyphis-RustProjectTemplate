#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn rusty_cxxbridge_integer() -> i32;
    }
}

// Implementierung der in `extern "Rust"` deklarierten Funktion.
pub fn rusty_cxxbridge_integer() -> i32 {
    42
}
