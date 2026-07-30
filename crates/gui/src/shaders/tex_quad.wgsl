@binding(0) @group(0) var tex_0 : texture_2d<f32>;

@binding(1) @group(0) var samp_0 : sampler;

struct TexQuadVsOut_0
{
    @builtin(position) svPosition_0 : vec4<f32>,
    @location(0) uv_0 : vec2<f32>,
};

struct vertexInput_0
{
    @location(0) position_0 : vec2<f32>,
    @location(1) uv_in_0 : vec2<f32>,
};

@vertex
fn vs_main( _S1 : vertexInput_0) -> TexQuadVsOut_0
{
    var o_0 : TexQuadVsOut_0;
    o_0.svPosition_0 = vec4<f32>(_S1.position_0, 0.0f, 1.0f);
    o_0.uv_0 = _S1.uv_in_0;
    return o_0;
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) uv_1 : vec2<f32>,
};

@fragment
fn fs_main( _S2 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var _S3 : pixelOutput_0 = pixelOutput_0( (textureSample((tex_0), (samp_0), (_S2.uv_1))) );
    return _S3;
}

