//! Graphical User Interface components.

#[cfg(all(feature = "gui_unix", not(windows)))]
pub mod gui;

#[cfg(feature = "gui_windows")]
pub mod gui_wgpu;
