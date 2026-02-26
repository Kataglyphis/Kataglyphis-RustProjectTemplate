use std::path::{Path, PathBuf};

use anyhow::Context;
use plotters::coord::Shift;
use plotters::prelude::*;

pub fn plot_loss_curve(path: impl AsRef<Path>, losses: &[f32], title: &str) -> anyhow::Result<()> {
    if losses.is_empty() {
        anyhow::bail!("cannot plot empty loss series");
    }

    let path = path.as_ref();
    ensure_parent_dir(path)?;

    render_loss_curve(path, losses, title)
}

fn render_loss_curve(path: &Path, losses: &[f32], title: &str) -> anyhow::Result<()> {
    let (width, height) = (1280, 720);
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();

    fill_white(&root)?;
    let (x_max, y_min, y_max) = chart_bounds(losses);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0i32..x_max, y_min..y_max)
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("loss")
        .draw()
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))?;

    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(i, y)| (i as i32, *y)),
            &RED,
        ))
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))?
        .label("loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))?;

    present(&root)?;

    Ok(())
}

fn fill_white(area: &DrawingArea<BitMapBackend<'_>, Shift>) -> anyhow::Result<()> {
    area.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))
}

fn chart_bounds(losses: &[f32]) -> (i32, f32, f32) {
    let x_max = (losses.len() - 1) as i32;
    let (min_y, max_y) = min_max(losses);
    let y_pad = ((max_y - min_y).abs() * 0.1).max(1e-6);
    (x_max, min_y - y_pad, max_y + y_pad)
}

fn present(area: &DrawingArea<BitMapBackend<'_>, Shift>) -> anyhow::Result<()> {
    area.present()
        .map_err(|e| anyhow::anyhow!("plotters error: {e:?}"))
}

fn ensure_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

#[allow(dead_code)]
fn default_plot_path(stem: &str) -> PathBuf {
    PathBuf::from(format!("./{stem}.png"))
}
