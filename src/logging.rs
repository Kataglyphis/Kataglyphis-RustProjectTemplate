// src/logging.rs — Logger initialisation and the `success!` macro.

use std::io::Write;

/// Initialise the `env_logger` with a loguru-style format (`HH:MM:SS | LEVEL | msg`).
///
/// Logging behaviour:
/// - If `RUST_LOG` is set, the user has full control.
/// - Otherwise default to INFO (override with `KATAGLYPHIS_LOG_LEVEL`).
/// - Noisy wgpu / naga modules are clamped to WARN.
pub fn init_logger() {
    let mut builder = env_logger::Builder::from_env(env_logger::Env::default());

    builder.format(|buf, record| {
        let ts = chrono::Local::now().format("%H:%M:%S");
        let level = if record.target() == "SUCCESS" {
            "SUCCESS"
        } else {
            match record.level() {
                log::Level::Error => "ERROR",
                log::Level::Warn => "WARN",
                log::Level::Info => "INFO",
                log::Level::Debug => "DEBUG",
                log::Level::Trace => "TRACE",
            }
        };
        writeln!(buf, "{ts} | {level:<8} | {}", record.args())
    });

    if std::env::var_os("RUST_LOG").is_none() {
        let level = std::env::var("KATAGLYPHIS_LOG_LEVEL")
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref()
            .and_then(|v| match v {
                "error" => Some(log::LevelFilter::Error),
                "warn" | "warning" => Some(log::LevelFilter::Warn),
                "info" => Some(log::LevelFilter::Info),
                "debug" => Some(log::LevelFilter::Debug),
                "trace" => Some(log::LevelFilter::Trace),
                _ => None,
            })
            .unwrap_or(log::LevelFilter::Info);

        builder.filter_level(level);

        // Suppress very chatty modules.
        for module in &["wgpu", "wgpu_core", "wgpu_hal", "naga"] {
            builder.filter_module(module, log::LevelFilter::Warn);
        }
    }

    builder.init();
}

/// Log a message at INFO level with the target set to `"SUCCESS"` so the
/// formatter renders it with the `SUCCESS` label.
#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        log::info!(target: "SUCCESS", $($arg)*);
    };
}
