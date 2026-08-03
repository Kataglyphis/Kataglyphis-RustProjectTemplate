@binding(0) @group(0) var hdrTex_0 : texture_2d<f32>;

@binding(1) @group(0) var hdrSampler_0 : sampler;

@binding(2) @group(0) var bloomTex_0 : texture_2d<f32>;

@binding(4) @group(0) var aoTex_0 : texture_2d<f32>;

struct TonemapUniforms_std140_0
{
    @align(16) params_0 : vec4<f32>,
};

@binding(3) @group(0) var<uniform> tonemapUniforms_0 : TonemapUniforms_std140_0;
@binding(5) @group(0) var<storage, read> exposureState_0 : array<f32>;

struct FullscreenVsOut_0
{
    @location(0) svPosition_0 : vec4<f32>,
    @location(0) uv_0 : vec2<f32>,
};

fn fullscreen_vs_0( vid_0 : u32) -> FullscreenVsOut_0
{
    var x_0 : f32 = f32(vid_0 / u32(2)) * 4.0f - 1.0f;
    var y_0 : f32 = f32(vid_0 % u32(2)) * 4.0f - 1.0f;
    var o_0 : FullscreenVsOut_0;
    o_0.svPosition_0 = vec4<f32>(x_0, y_0, 0.0f, 1.0f);
    o_0.uv_0 = vec2<f32>((x_0 + 1.0f) * 0.5f, 1.0f - (y_0 + 1.0f) * 0.5f);
    return o_0;
}

struct VsOut_0
{
    @builtin(position) svPosition_1 : vec4<f32>,
    @location(0) uv_1 : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid_1 : u32) -> VsOut_0
{
    var o_1 : FullscreenVsOut_0 = fullscreen_vs_0(vid_1);
    var r_0 : VsOut_0;
    r_0.svPosition_1 = o_1.svPosition_0;
    r_0.uv_1 = o_1.uv_0;
    return r_0;
}

fn aces_tonemap_0( x_1 : vec3<f32>) -> vec3<f32>
{
    return clamp(x_1 * (vec3<f32>(2.50999999046325684f) * x_1 + vec3<f32>(0.02999999932944775f)) / (x_1 * (vec3<f32>(2.43000006675720215f) * x_1 + vec3<f32>(0.5899999737739563f)) + vec3<f32>(0.14000000059604645f)), vec3<f32>(0.0f), vec3<f32>(1.0f));
}

fn linear_to_srgb_0( c_0 : vec3<f32>) -> vec3<f32>
{
    return select(vec3<f32>(1.0549999475479126f) * pow(max(c_0, vec3<f32>(0.0f)), vec3<f32>(0.4166666567325592f)) - vec3<f32>(0.05499999970197678f), c_0 * vec3<f32>(12.92000007629394531f), c_0 < vec3<f32>(0.00313080009073019f));
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) uv_2 : vec2<f32>,
};

@fragment
fn fs_main( _S1 : pixelInput_0, @builtin(position) svPosition_2 : vec4<f32>) -> pixelOutput_0
{
    var mapped_0 : vec3<f32> = aces_tonemap_0((textureSample((hdrTex_0), (hdrSampler_0), (_S1.uv_2))).xyz * vec3<f32>(mix(1.0f, (textureSample((aoTex_0), (hdrSampler_0), (_S1.uv_2))).x, tonemapUniforms_0.params_0.y)) * vec3<f32>(exp2(exposureState_0[i32(0)])) + (textureSample((bloomTex_0), (hdrSampler_0), (_S1.uv_2))).xyz * vec3<f32>(tonemapUniforms_0.params_0.x));
    if((tonemapUniforms_0.params_0.w) > 0.5f)
    {
        var _S2 : pixelOutput_0 = pixelOutput_0( vec4<f32>(linear_to_srgb_0(mapped_0), 1.0f) );
        return _S2;
    }
    var _S3 : pixelOutput_0 = pixelOutput_0( vec4<f32>(mapped_0, 1.0f) );
    return _S3;
}

