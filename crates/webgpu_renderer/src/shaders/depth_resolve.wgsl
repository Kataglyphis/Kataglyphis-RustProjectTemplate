@binding(0) @group(0) var msaaDepth_0 : texture_depth_multisampled_2d;

struct FullscreenVsOut_0
{
    @builtin(position) svPosition_0 : vec4<f32>,
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

@vertex
fn vs_main(@builtin(vertex_index) vid_1 : u32) -> FullscreenVsOut_0
{
    return fullscreen_vs_0(vid_1);
}

struct pixelOutput_0
{
    @builtin(frag_depth) output_0 : f32,
};

struct pixelInput_0
{
    @location(0) uv_1 : vec2<f32>,
};

@fragment
fn fs_main( _S1 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var w_0 : u32;
    var h_0 : u32;
    var samples_0 : u32;
    {var dim = textureDimensions((msaaDepth_0));((w_0)) = dim.x;((h_0)) = dim.y;((samples_0)) = textureNumSamples((msaaDepth_0));};
    var _S2 : vec2<i32> = vec2<i32>(_S1.uv_1 * vec2<f32>(f32(w_0), f32(h_0)));
    var minDepth_0 : f32 = 1.0f;
    var i_0 : u32 = u32(0);
    for(;;)
    {
        if(i_0 < samples_0)
        {
        }
        else
        {
            break;
        }
        var _S3 : f32 = min(minDepth_0, (textureLoad((msaaDepth_0), (_S2), (i32(i_0)))));
        var _S4 : u32 = i_0 + u32(1);
        minDepth_0 = _S3;
        i_0 = _S4;
    }
    var _S5 : pixelOutput_0 = pixelOutput_0( minDepth_0 );
    return _S5;
}

