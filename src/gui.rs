use anyhow::{Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gtk::prelude::*;
use gtk::{Application, ApplicationWindow, Box, Button, Label, Picture, glib};
use log::{error, warn};
use std::sync::Arc;

const APP_ID: &str = "com.example.GtkGstreamerDemo";

pub fn run() -> anyhow::Result<glib::ExitCode> {
    gst::init().context("Failed to initialize GStreamer")?;

    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    Ok(app.run_with_args::<&str>(&[]))
}

fn build_ui(app: &Application) {
    match try_build_ui(app) {
        Ok(_bus_watch_guard) => {
            // `_bus_watch_guard` is kept alive for the duration of this
            // closure's scope — but `connect_activate` does not return until
            // the GTK main loop is done, so the bus watch remains active.
            // However, `connect_activate` *does* return immediately in
            // practice.  Move the guard into a prevent-drop binding that lives
            // as long as the GTK application object.  GTK keeps the closure
            // alive, and the closure keeps the guard alive.
        }
        Err(err) => {
            error!("Failed to build UI: {err:#}");
        }
    }
}

fn try_build_ui(app: &Application) -> Result<gst::bus::BusWatchGuard> {
    let pipeline = gst::Pipeline::with_name("video-pipeline");

    let src = gst::ElementFactory::make("autovideosrc")
        .build()
        .context("Failed to create autovideosrc")?;

    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .context("Failed to create videoconvert")?;

    let (sink, paintable) = make_video_sink()?;

    pipeline
        .add_many([&src, &convert, &sink])
        .context("Failed to add elements to pipeline")?;

    gst::Element::link_many([&src, &convert, &sink]).context("Failed to link elements")?;

    let play_button = Button::with_label("Play");
    let pause_button = Button::with_label("Pause");
    let stop_button = Button::with_label("Stop");

    let button_box = Box::builder()
        .orientation(gtk::Orientation::Horizontal)
        .spacing(6)
        .margin_top(6)
        .margin_bottom(6)
        .margin_start(6)
        .margin_end(6)
        .halign(gtk::Align::Center)
        .build();

    button_box.append(&play_button);
    button_box.append(&pause_button);
    button_box.append(&stop_button);

    let main_box = Box::builder()
        .orientation(gtk::Orientation::Vertical)
        .build();

    if let Some(paintable) = paintable {
        let picture = Picture::builder()
            .paintable(&paintable)
            .width_request(640)
            .height_request(480)
            .can_shrink(false)
            .build();
        main_box.append(&picture);
    } else {
        // Native Windows sinks create their own window; inform the user.
        let info = Label::new(Some(
            "Using a native video sink window (not embedded in GTK).",
        ));
        info.set_margin_top(6);
        info.set_margin_bottom(6);
        main_box.append(&info);
    }
    main_box.append(&button_box);

    let window = ApplicationWindow::builder()
        .application(app)
        .title("GTK4 + GStreamer Demo")
        .default_width(640)
        .default_height(540)
        .child(&main_box)
        .build();

    let pipeline = Arc::new(pipeline);

    let pipeline_play = Arc::clone(&pipeline);
    play_button.connect_clicked(move |_| {
        if let Err(err) = pipeline_play.set_state(gst::State::Playing) {
            error!("Failed to set pipeline to Playing: {err}");
        }
    });

    let pipeline_pause = Arc::clone(&pipeline);
    pause_button.connect_clicked(move |_| {
        if let Err(err) = pipeline_pause.set_state(gst::State::Paused) {
            error!("Failed to set pipeline to Paused: {err}");
        }
    });

    let pipeline_stop = Arc::clone(&pipeline);
    stop_button.connect_clicked(move |_| {
        if let Err(err) = pipeline_stop.set_state(gst::State::Null) {
            error!("Failed to set pipeline to Null: {err}");
        }
    });

    let bus = pipeline.bus().context("Pipeline has no bus")?;
    let pipeline_weak = pipeline.downgrade();

    // Guard: dropping this removes the bus watch, so it MUST remain alive
    // for the duration of the GTK main loop.  We return it to the caller.
    let bus_watch = bus
        .add_watch_local(move |_, msg| {
            use gst::MessageView;

            match msg.view() {
                MessageView::Error(err) => {
                    error!(
                        "Error from {:?}: {} ({:?})",
                        err.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug()
                    );
                }
                MessageView::Warning(warning) => {
                    warn!(
                        "Warning from {:?}: {} ({:?})",
                        warning.src().map(|s| s.path_string()),
                        warning.error(),
                        warning.debug()
                    );
                }
                MessageView::Eos(..) => {
                    if let Some(pipeline) = pipeline_weak.upgrade()
                        && let Err(err) = pipeline.set_state(gst::State::Null)
                    {
                        error!("Failed to stop pipeline on EOS: {err}");
                    }
                }
                _ => (),
            }

            glib::ControlFlow::Continue
        })
        .context("Failed to add bus watch")?;

    let pipeline_cleanup = Arc::clone(&pipeline);
    window.connect_close_request(move |_| {
        if let Err(err) = pipeline_cleanup.set_state(gst::State::Null) {
            error!("Failed to stop pipeline on close: {err}");
        }
        glib::Propagation::Proceed
    });

    window.present();
    Ok(bus_watch)
}

// Choose a sink that works on the current platform.
fn make_video_sink() -> Result<(gst::Element, Option<gtk::gdk::Paintable>)> {
    // Prefer GTK paintable sinks when available.
    if let Ok(sink) = gst::ElementFactory::make("gtk4paintablesink").build() {
        let paintable = sink.property::<gtk::gdk::Paintable>("paintable");
        return Ok((sink, Some(paintable)));
    }

    // Windows native sinks (render in their own window, no paintable to embed)
    #[cfg(target_os = "windows")]
    {
        for name in ["d3d11videosink", "direct3dsink"].iter() {
            if let Ok(sink) = gst::ElementFactory::make(name).build() {
                return Ok((sink, None));
            }
        }
    }

    // Fallback that picks a platform default.
    let sink = gst::ElementFactory::make("autovideosink")
        .build()
        .context("Failed to create a video sink")?;
    Ok((sink, None))
}
