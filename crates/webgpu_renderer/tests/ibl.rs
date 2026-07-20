//! Split-sum IBL, checked against the properties the maths guarantees.
//!
//! These are structural readback assertions, not golden images. The maps are
//! `Rgba16Float` cubes whose exact contents vary with driver filtering, but the
//! things that make them *correct* - a constant environment convolving to
//! itself, energy never being created, roughness monotonically blurring - are
//! exact statements that hold on any conforming GPU.
//!
//! The single most valuable one is `a_constant_environment_convolves_to_its_own_radiance`.
//! Dropping the sin(theta) solid-angle weight from the irradiance convolution
//! is the classic bug in this pipeline and is invisible on a rendered frame -
//! the map just comes out uniformly too bright, which reads as "the ambient
//! slider wants turning down". Deleting that one factor was measured against
//! this test: the map comes out exactly 2x too bright, a relative error of
//! 1.000, because the average of cos(theta) over theta uniform on [0, PI/2] is
//! 2/PI where the cosine-weighted solid angle average is 1/PI.
//!
//! `the_brdf_lut_reproduces_the_known_mirror_and_grazing_behaviour` earned its
//! keep the same way: it caught a NaN in the near-zero-roughness GGX sample
//! that silently discarded 44% of the samples.

use kataglyphis_webgpu_renderer::render::ibl::{
    BrdfLut, IblEnvironment, BRDF_LUT_SIZE, IRRADIANCE_SIZE, PREFILTER_MIPS, PREFILTER_SIZE,
};
use kataglyphis_webgpu_renderer::{
    decode_hdr, load_gltf, EquirectImage, ForwardRenderer, GpuContext, OrbitCamera,
};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

/// An environment split at the equator: bright above, dark below.
///
/// A two-level signal whose contrast lives at the lowest possible spatial
/// frequency. That matters for the roughness test: the variance of a two-level
/// image depends only on the proportion of bright to dark texels, not on the
/// resolution, so comparing variance across mips of different sizes measures
/// the GGX blur and not the downsampling.
fn split_environment(width: u32, height: u32, upper: f32, lower: f32) -> EquirectImage {
    let mut rgba32f = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let value = if y < height / 2 { upper } else { lower };
        for _ in 0..width {
            rgba32f.extend_from_slice(&[value, value, value, 1.0]);
        }
    }
    EquirectImage::new(width, height, rgba32f).expect("split environment is well formed")
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn variance(values: &[f32]) -> f32 {
    let m = mean(values);
    values.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32
}

#[test]
fn a_constant_environment_convolves_to_its_own_radiance() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The one environment whose convolution has a closed form. For constant
    // radiance L the irradiance is E = integral of L cos(theta) dw over the
    // hemisphere = PI * L, and the map stores E / PI (the value that
    // multiplies albedo directly), so every texel of every face must read back
    // as exactly L - independent of the normal, the face, everything.
    // 0.25 and 0.5 are exact in binary16; 0.7 is not, so the measured error is
    // a real number rather than an artefact of picking friendly constants.
    let radiance = [0.25f32, 0.5, 0.7];
    let environment = IblEnvironment::bake(&gpu, &EquirectImage::constant(64, 32, radiance));

    let mut worst = 0.0f32;
    for face in 0..6u32 {
        let texels = environment.read_irradiance_face(&gpu, face);
        assert_eq!(texels.len(), (IRRADIANCE_SIZE * IRRADIANCE_SIZE) as usize);
        for texel in &texels {
            for channel in 0..3 {
                let relative = (texel[channel] - radiance[channel]).abs() / radiance[channel];
                worst = worst.max(relative);
            }
        }
    }

    // Measured worst-case relative error: see the report in the commit
    // message. The budget covers the midpoint quadrature (~1e-4), the
    // Rgba16Float storage of both the environment and the result (2^-11 each),
    // and driver-dependent cube filtering across face seams.
    assert!(
        worst < 0.01,
        "constant environment did not convolve to itself: worst relative error {worst}"
    );
    eprintln!("uniform-environment irradiance: worst relative error {worst:.6}");
}

#[test]
fn a_constant_environment_prefilters_to_its_own_radiance_at_every_roughness() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The prefilter is a weighted average of environment samples normalised by
    // the sum of its own weights, so a constant environment must survive it
    // untouched at every roughness. This is what catches a missing or
    // mis-summed `total_weight` - which otherwise shows up only as an
    // environment that gets darker as it gets rougher, and looks plausible.
    let radiance = [0.4f32, 0.4, 0.4];
    let environment = IblEnvironment::bake(&gpu, &EquirectImage::constant(64, 32, radiance));

    for mip in 0..PREFILTER_MIPS {
        let mut worst = 0.0f32;
        for face in 0..6u32 {
            for texel in environment.read_prefiltered_face(&gpu, face, mip) {
                worst = worst.max((texel[0] - radiance[0]).abs() / radiance[0]);
            }
        }
        assert!(
            worst < 0.01,
            "prefilter mip {mip} (roughness {}) drifted from a constant environment: {worst}",
            mip as f32 / (PREFILTER_MIPS - 1) as f32
        );
        eprintln!("prefilter mip {mip}: worst relative error {worst:.5}");
    }
}

#[test]
fn irradiance_never_exceeds_the_brightest_radiance_in_the_environment() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The bound, derived rather than guessed:
    //
    //   stored(n) = (1/PI) * integral_hemisphere L(w) cos(theta) dw
    //            <= (1/PI) * L_max * integral_hemisphere cos(theta) dw
    //             = (1/PI) * L_max * PI
    //             = L_max
    //
    // i.e. the cosine-weighted average of a bounded environment cannot exceed
    // that bound. It is tight: a fully constant environment attains it, which
    // is exactly what the test above asserts. Any weighting error that
    // over-counts solid angle - the dropped sin(theta), a hemisphere
    // integrated as a full sphere - breaks this.
    let bright = 4.0f32;
    let image = split_environment(64, 32, bright, 0.0);
    assert_eq!(image.max_radiance(), bright);
    let environment = IblEnvironment::bake(&gpu, &image);

    let mut highest = 0.0f32;
    let mut lowest = f32::INFINITY;
    for face in 0..6u32 {
        for texel in environment.read_irradiance_face(&gpu, face) {
            highest = highest.max(texel[0]);
            lowest = lowest.min(texel[0]);
        }
    }

    assert!(
        highest <= bright * 1.01,
        "irradiance {highest} exceeds the environment's maximum radiance {bright}"
    );
    // The bound must not be satisfied trivially: a normal pointing straight up
    // sees only the bright half and a normal pointing down only the dark half,
    // so the map has to span most of the range. A map of all zeros would pass
    // the bound above and nothing else.
    assert!(
        highest > bright * 0.8,
        "the up-facing normal should see nearly the full bright hemisphere, got {highest}"
    );
    assert!(
        lowest < bright * 0.2,
        "the down-facing normal should see nearly none of it, got {lowest}"
    );
    eprintln!("bounded-environment irradiance: max {highest:.4}, min {lowest:.4}, bound {bright}");
}

#[test]
fn higher_roughness_prefilter_mips_are_strictly_blurrier() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let environment = IblEnvironment::bake(&gpu, &split_environment(128, 64, 1.0, 0.0));

    // Face 4 is +Z, which straddles the equator symmetrically, so roughly half
    // its texels start bright and half dark. Blurring can only move texels
    // toward the mean, so the variance must fall with every roughness step.
    let variances: Vec<f32> = (0..PREFILTER_MIPS)
        .map(|mip| {
            let face: Vec<f32> = environment
                .read_prefiltered_face(&gpu, 4, mip)
                .into_iter()
                .map(|texel| texel[0])
                .collect();
            assert_eq!(
                face.len(),
                ((PREFILTER_SIZE >> mip) * (PREFILTER_SIZE >> mip)) as usize
            );
            variance(&face)
        })
        .collect();

    eprintln!("prefilter variance by mip (roughness 0 -> 1): {variances:?}");
    for mip in 1..PREFILTER_MIPS as usize {
        assert!(
            variances[mip] < variances[mip - 1],
            "mip {mip} (variance {}) is not blurrier than mip {} (variance {})",
            variances[mip],
            mip - 1,
            variances[mip - 1]
        );
    }
    // Not merely monotone but substantially so: roughness 1.0 should have
    // smeared the hemisphere edge into something close to flat.
    assert!(
        variances[PREFILTER_MIPS as usize - 1] < variances[0] * 0.25,
        "roughness 1.0 barely blurred anything: {variances:?}"
    );
}

#[test]
fn the_brdf_lut_stays_in_range_and_conserves_energy() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let lut = BrdfLut::new(&gpu);
    let table = lut.read_back(&gpu);
    assert_eq!(table.len(), (BRDF_LUT_SIZE * BRDF_LUT_SIZE) as usize);

    let mut worst_sum = 0.0f32;
    for entry in &table {
        assert!(
            (0.0..=1.0).contains(&entry[0]) && (0.0..=1.0).contains(&entry[1]),
            "BRDF LUT entry out of [0,1]: {entry:?}"
        );
        // The two terms are the split of a single Fresnel-weighted integral of
        // a BRDF that cannot reflect more than it receives, so scale + bias
        // (the value at F0 = 1) is bounded by 1.
        worst_sum = worst_sum.max(entry[0] + entry[1]);
    }
    assert!(
        worst_sum <= 1.0 + 1e-2,
        "BRDF LUT creates energy: max scale + bias = {worst_sum}"
    );
    eprintln!("BRDF LUT: max scale + bias = {worst_sum:.4}");
}

#[test]
fn the_brdf_lut_reproduces_the_known_mirror_and_grazing_behaviour() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let table = BrdfLut::new(&gpu).read_back(&gpu);
    let at = |n_dot_v_index: u32, roughness_index: u32| {
        table[(roughness_index * BRDF_LUT_SIZE + n_dot_v_index) as usize]
    };

    let smooth = 0u32;
    let rough = BRDF_LUT_SIZE - 1;
    let grazing = 0u32;
    let normal_incidence = BRDF_LUT_SIZE - 1;
    // A perfect mirror loses nothing: shadowing-masking is 1, so scale + bias
    // integrates to exactly 1 at every angle, with Fresnel deciding the split.
    // This is the assertion that caught the near-zero-roughness NaN in
    // `importance_sample_ggx` - it summed to 0.563 instead of 1.0, uniformly,
    // which no in-range or monotonicity check would have noticed.

    for n_dot_v in [grazing, BRDF_LUT_SIZE / 2, normal_incidence] {
        let [scale, bias] = at(n_dot_v, smooth);
        assert!(
            (scale + bias - 1.0).abs() < 0.02,
            "a mirror must be lossless at N.V index {n_dot_v}: {scale} + {bias}"
        );
    }

    // At normal incidence F = F0, so the whole answer is the scale on F0.
    let [scale, bias] = at(normal_incidence, smooth);
    assert!(
        scale > 0.98 && bias < 0.02,
        "mirror at normal incidence should be (1, 0), got ({scale}, {bias})"
    );

    // At grazing incidence Fresnel drives reflectance to 1 regardless of F0,
    // so the F0-independent bias takes over from the scale.
    let [grazing_scale, grazing_bias] = at(grazing, smooth);
    assert!(
        grazing_bias > grazing_scale,
        "grazing Fresnel should be F0-independent, got scale {grazing_scale} bias {grazing_bias}"
    );

    // Roughness costs energy: the Smith term removes light to shadowing and
    // masking, and none of it comes back.
    let smooth_total = {
        let [s, b] = at(BRDF_LUT_SIZE / 2, smooth);
        s + b
    };
    let rough_total = {
        let [s, b] = at(BRDF_LUT_SIZE / 2, rough);
        s + b
    };
    assert!(
        rough_total < smooth_total,
        "roughness 1 ({rough_total}) must lose more energy than roughness 0 ({smooth_total})"
    );
    eprintln!(
        "BRDF LUT: mirror total {smooth_total:.4}, roughest total {rough_total:.4}, \
         grazing (scale {grazing_scale:.4}, bias {grazing_bias:.4})"
    );
}

#[test]
fn the_equirect_projection_puts_the_sky_on_the_right_faces() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // Bright above the equator, dark below. If a face's basis were mirrored or
    // the latitude mapping flipped, the +Y and -Y faces would swap - which a
    // smooth panorama hides completely and every derived map inherits.
    let environment = IblEnvironment::bake(&gpu, &split_environment(128, 64, 1.0, 0.0));
    let face_mean = |face: u32| {
        let values: Vec<f32> = environment
            .read_environment_face(&gpu, face, 0)
            .into_iter()
            .map(|texel| texel[0])
            .collect();
        mean(&values)
    };

    let up = face_mean(2);
    let down = face_mean(3);
    assert!(
        up > 0.95,
        "+Y face must be entirely the bright half, got {up}"
    );
    assert!(
        down < 0.05,
        "-Y face must be entirely the dark half, got {down}"
    );

    // The four side faces each straddle the equator, so each is about half
    // bright. A mirrored side face would still average 0.5, but a face taking
    // its latitude from the wrong axis would not.
    for face in [0u32, 1, 4, 5] {
        let side = face_mean(face);
        assert!(
            (side - 0.5).abs() < 0.1,
            "side face {face} should straddle the equator, got mean {side}"
        );
    }
}

/// Renders the bundled cube and returns the frame bytes.
fn render(renderer: &mut ForwardRenderer, gpu: &GpuContext) -> Vec<u8> {
    renderer
        .render_to_pixels(gpu, 128, 128, &OrbitCamera::default())
        .expect("headless render must succeed")
}

#[test]
fn with_no_environment_the_analytic_path_renders_exactly_what_it_always_did() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    assert!(
        !renderer.environment_enabled(),
        "IBL must be off until an environment is set"
    );
    let baseline = render(&mut renderer, &gpu);

    // Round-tripping through an environment and back must land on the same
    // pixels, byte for byte. `forward.wgsl` picks between the analytic and the
    // environment result with `select`, so the fallback is not "close to" the
    // old path, it IS the old path.
    let mut round_tripped = ForwardRenderer::new(&gpu, 128, 128);
    round_tripped.upload_scene(&gpu, &scene);
    round_tripped.set_environment(&gpu, &EquirectImage::sky(64, 32));
    assert!(round_tripped.environment_enabled());
    round_tripped.clear_environment(&gpu);
    assert!(!round_tripped.environment_enabled());

    assert_eq!(
        baseline,
        render(&mut round_tripped, &gpu),
        "clearing the environment did not restore the analytic path exactly"
    );
}

#[test]
fn setting_an_environment_actually_changes_the_rendered_frame() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The guard against a feature that bakes beautiful maps nothing samples.
    // Everything above tests the precompute in isolation; this is the only
    // test that fails if group 1 is never bound, if the uniform's enabled flag
    // never reaches the shader, or if `select` picks the wrong branch.
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    let analytic = render(&mut renderer, &gpu);

    // A bright white environment: far brighter ambient than the analytic
    // sky/ground, so the cube's shaded side must lift.
    renderer.set_environment(&gpu, &EquirectImage::constant(64, 32, [3.0, 3.0, 3.0]));
    let lit = render(&mut renderer, &gpu);
    assert_ne!(
        analytic, lit,
        "the baked environment never reached the frame"
    );

    // Directional, not merely different: sum over the cube's pixels only. The
    // sky fills the background and is drawn by its own pass, which IBL does
    // not touch, so comparing whole-frame means would dilute the signal.
    let cube_luma = |pixels: &[u8]| {
        let mut total = 0u64;
        let mut count = 0u64;
        for pixel in pixels.chunks_exact(4) {
            // The cube is red-dominant; the sky is blue-dominant.
            if pixel[0] > pixel[2] {
                total += pixel[0] as u64 + pixel[1] as u64 + pixel[2] as u64;
                count += 1;
            }
        }
        assert!(count > 100, "found only {count} cube pixels to compare");
        total as f64 / count as f64
    };

    let analytic_luma = cube_luma(&analytic);
    let lit_luma = cube_luma(&lit);
    eprintln!("cube mean luma: analytic {analytic_luma:.2}, environment-lit {lit_luma:.2}");
    assert!(
        lit_luma > analytic_luma + 5.0,
        "a 3.0-radiance environment should brighten the cube: {analytic_luma} -> {lit_luma}"
    );

    // A dark environment must push it the other way, so the test cannot be
    // passed by anything that merely adds a constant.
    renderer.set_environment(&gpu, &EquirectImage::constant(64, 32, [0.01, 0.01, 0.01]));
    let dim_luma = cube_luma(&render(&mut renderer, &gpu));
    eprintln!("cube mean luma: dark environment {dim_luma:.2}");
    assert!(
        dim_luma < analytic_luma,
        "a near-black environment should darken the cube: {analytic_luma} -> {dim_luma}"
    );
}

/// Radiance-encodes an [`EquirectImage`] as a flat (unRLE'd) `.hdr` file.
///
/// A test-local encoder rather than a crate API on purpose: the renderer only
/// ever *reads* `.hdr`, and keeping the writer beside the test that needs it
/// stops it from looking like a supported feature.
fn encode_hdr_flat(image: &EquirectImage) -> Vec<u8> {
    let mut out = format!(
        "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {} +X {}\n",
        image.height, image.width
    )
    .into_bytes();
    for texel in image.rgba32f.chunks_exact(4) {
        let max = texel[0].max(texel[1]).max(texel[2]);
        if max < 1e-32 {
            out.extend_from_slice(&[0; 4]);
            continue;
        }
        // frexp by hand: max = v * 2^e, v in [0.5, 1); mantissas are then
        // 8-bit fractions of 2^e, matching Radiance's setcolr.
        let mut e = 0i32;
        let mut v = max;
        while v >= 1.0 {
            v *= 0.5;
            e += 1;
        }
        while v < 0.5 {
            v *= 2.0;
            e -= 1;
        }
        let scale = f64::from(v) * 256.0 / f64::from(max);
        out.extend_from_slice(&[
            (f64::from(texel[0]) * scale) as u8,
            (f64::from(texel[1]) * scale) as u8,
            (f64::from(texel[2]) * scale) as u8,
            (e + 128) as u8,
        ]);
    }
    out
}

#[test]
fn hdr_bytes_decode_and_bake_into_the_same_environment_as_the_source_pixels() {
    // The whole pipeline the decoder exists for: sky -> .hdr bytes -> decode
    // -> bake, checked against baking the sky directly. The CPU half runs
    // everywhere; the bake comparison needs an adapter.
    let sky = EquirectImage::sky(64, 32);
    let bytes = encode_hdr_flat(&sky);
    let decoded = decode_hdr(&bytes).expect("the encoded sky must decode");
    assert_eq!((decoded.width, decoded.height), (sky.width, sky.height));

    let mut worst = 0.0f32;
    for (got, want) in decoded
        .rgba32f
        .chunks_exact(4)
        .zip(sky.rgba32f.chunks_exact(4))
    {
        let max = want[0].max(want[1]).max(want[2]);
        for channel in 0..3 {
            worst = worst.max((got[channel] - want[channel]).abs() / max);
        }
    }
    eprintln!("hdr round-trip of the sky: worst relative error {worst:.6}");
    assert!(
        worst < 1.0 / 128.0,
        "RGBE round-trip error {worst} exceeds its quantum"
    );

    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let direct = IblEnvironment::bake(&gpu, &sky);
    let via_hdr = IblEnvironment::bake_hdr(&gpu, &bytes).expect("bake_hdr composes decode + bake");

    // The irradiance convolution averages thousands of environment texels, so
    // the per-pixel RGBE quantisation (< 1/128) cannot grow on the way
    // through; 2% also covers the half-float storage of both maps.
    let mut worst = 0.0f32;
    for face in 0..6u32 {
        let a = direct.read_irradiance_face(&gpu, face);
        let b = via_hdr.read_irradiance_face(&gpu, face);
        for (x, y) in a.iter().zip(&b) {
            for channel in 0..3 {
                worst = worst.max((x[channel] - y[channel]).abs() / x[channel].max(1e-3));
            }
        }
    }
    eprintln!("irradiance from .hdr vs from source pixels: worst relative diff {worst:.6}");
    assert!(
        worst < 0.02,
        "baking the decoded .hdr diverged from the source: {worst}"
    );
}

#[test]
fn the_brdf_table_is_baked_once_and_shared_across_environments() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The LUT integrates the GGX BRDF against a white furnace and has no
    // environment term, so rebaking it per environment would be pure waste.
    // Pin the sharing so a later refactor cannot quietly reintroduce it.
    let mut renderer = ForwardRenderer::new(&gpu, 64, 64);
    assert!(
        renderer.brdf_lut().is_none(),
        "nothing should bake before use"
    );

    renderer.set_environment(&gpu, &EquirectImage::constant(32, 16, [1.0, 1.0, 1.0]));
    let first = renderer
        .brdf_lut()
        .expect("set_environment bakes the LUT")
        .read_back(&gpu);

    renderer.set_environment(&gpu, &EquirectImage::sky(32, 16));
    let second = renderer
        .brdf_lut()
        .expect("the LUT survives")
        .read_back(&gpu);

    assert_eq!(
        first, second,
        "the BRDF LUT must not depend on the environment"
    );
}
