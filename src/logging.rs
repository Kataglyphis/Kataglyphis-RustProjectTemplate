#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        log::info!(target: "SUCCESS", $($arg)*);
    };
}
