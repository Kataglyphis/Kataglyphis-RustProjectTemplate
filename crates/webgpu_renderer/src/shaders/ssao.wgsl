@binding(0) @group(0) var depthTex_0 : texture_depth_2d;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct SsaoUniforms_std140_0
{
    @align(16) proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) inv_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) params_0 : vec4<f32>,
};

@binding(1) @group(0) var<uniform> u_0 : SsaoUniforms_std140_0;
@binding(2) @group(0) var aoTex_0 : texture_2d<f32>;

const KERNEL_0 : array<vec3<f32>, i32(12)> = array<vec3<f32>, i32(12)>( vec3<f32>(0.20399999618530273f, 0.10400000214576721f, 0.30099999904632568f), vec3<f32>(-0.32199999690055847f, 0.17000000178813934f, 0.22200000286102295f), vec3<f32>(0.10899999737739563f, -0.37200000882148743f, 0.28400000929832458f), vec3<f32>(-0.11400000005960464f, -0.18400000035762787f, 0.45100000500679016f), vec3<f32>(0.45100000500679016f, 0.32199999690055847f, 0.33899998664855957f), vec3<f32>(-0.53200000524520874f, -0.28099998831748962f, 0.23999999463558197f), vec3<f32>(0.31999999284744263f, -0.56400001049041748f, 0.3529999852180481f), vec3<f32>(-0.31000000238418579f, 0.61000001430511475f, 0.34599998593330383f), vec3<f32>(0.67100000381469727f, -0.16599999368190765f, 0.47400000691413879f), vec3<f32>(-0.14399999380111694f, -0.70200002193450928f, 0.48199999332427979f), vec3<f32>(-0.65799999237060547f, 0.3970000147819519f, 0.51399999856948853f), vec3<f32>(0.44200000166893005f, 0.67900002002716064f, 0.53100001811981201f) );
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

fn load_depth_0( uv_1 : vec2<f32>) -> f32
{
    var w_0 : u32;
    var h_0 : u32;
    {var dim = textureDimensions((depthTex_0));((w_0)) = dim.x;((h_0)) = dim.y;};
    var _S1 : vec3<i32> = vec3<i32>(clamp(vec2<i32>(uv_1 * vec2<f32>(f32(w_0), f32(h_0))), vec2<i32>(i32(0), i32(0)), vec2<i32>(i32(w_0), i32(h_0)) - vec2<i32>(i32(1))), i32(0));
    return (textureLoad((depthTex_0), ((_S1)).xy, ((_S1)).z));
}

fn view_pos_at_0( uv_2 : vec2<f32>,  depth_0 : f32) -> vec3<f32>
{
    var v_0 : vec4<f32> = (((vec4<f32>(uv_2.x * 2.0f - 1.0f, 1.0f - uv_2.y * 2.0f, depth_0, 1.0f)) * (mat4x4<f32>(u_0.inv_proj_0.data_0[i32(0)][i32(0)], u_0.inv_proj_0.data_0[i32(1)][i32(0)], u_0.inv_proj_0.data_0[i32(2)][i32(0)], u_0.inv_proj_0.data_0[i32(3)][i32(0)], u_0.inv_proj_0.data_0[i32(0)][i32(1)], u_0.inv_proj_0.data_0[i32(1)][i32(1)], u_0.inv_proj_0.data_0[i32(2)][i32(1)], u_0.inv_proj_0.data_0[i32(3)][i32(1)], u_0.inv_proj_0.data_0[i32(0)][i32(2)], u_0.inv_proj_0.data_0[i32(1)][i32(2)], u_0.inv_proj_0.data_0[i32(2)][i32(2)], u_0.inv_proj_0.data_0[i32(3)][i32(2)], u_0.inv_proj_0.data_0[i32(0)][i32(3)], u_0.inv_proj_0.data_0[i32(1)][i32(3)], u_0.inv_proj_0.data_0[i32(2)][i32(3)], u_0.inv_proj_0.data_0[i32(3)][i32(3)]))));
    return v_0.xyz / vec3<f32>(v_0.w);
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) uv_3 : vec2<f32>,
};

@fragment
fn fs_ssao( _S2 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var depth_1 : f32 = load_depth_0(_S2.uv_3);
    if(depth_1 >= 1.0f)
    {
        var _S3 : pixelOutput_0 = pixelOutput_0( vec4<f32>(1.0f) );
        return _S3;
    }
    var p_0 : vec3<f32> = view_pos_at_0(_S2.uv_3, depth_1);
    var w_1 : u32;
    var h_1 : u32;
    {var dim = textureDimensions((depthTex_0));((w_1)) = dim.x;((h_1)) = dim.y;};
    var _S4 : vec2<f32> = _S2.uv_3 + vec2<f32>(1.0f / f32(w_1), 0.0f);
    var _S5 : vec2<f32> = _S2.uv_3 + vec2<f32>(0.0f, 1.0f / f32(h_1));
    var n_0 : vec3<f32> = normalize(cross(view_pos_at_0(_S4, load_depth_0(_S4)) - p_0, view_pos_at_0(_S5, load_depth_0(_S5)) - p_0));
    const _S6 : vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    var up_0 : vec3<f32>;
    if((abs(n_0.y)) > 0.98000001907348633f)
    {
        up_0 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    else
    {
        up_0 = _S6;
    }
    var t_0 : vec3<f32> = normalize(cross(up_0, n_0));
    var _S7 : vec3<f32> = cross(n_0, t_0);
    var _S8 : f32 = u_0.params_0.x;
    var _S9 : f32 = u_0.params_0.y;
    var i_0 : i32 = i32(0);
    var occlusion_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_0 < i32(12))
        {
        }
        else
        {
            break;
        }
        var samplePos_0 : vec3<f32> = p_0 + (t_0 * vec3<f32>(KERNEL_0[i_0].x) + _S7 * vec3<f32>(KERNEL_0[i_0].y) + n_0 * vec3<f32>(KERNEL_0[i_0].z)) * vec3<f32>(_S8);
        var clip_0 : vec4<f32> = (((vec4<f32>(samplePos_0, 1.0f)) * (mat4x4<f32>(u_0.proj_0.data_0[i32(0)][i32(0)], u_0.proj_0.data_0[i32(1)][i32(0)], u_0.proj_0.data_0[i32(2)][i32(0)], u_0.proj_0.data_0[i32(3)][i32(0)], u_0.proj_0.data_0[i32(0)][i32(1)], u_0.proj_0.data_0[i32(1)][i32(1)], u_0.proj_0.data_0[i32(2)][i32(1)], u_0.proj_0.data_0[i32(3)][i32(1)], u_0.proj_0.data_0[i32(0)][i32(2)], u_0.proj_0.data_0[i32(1)][i32(2)], u_0.proj_0.data_0[i32(2)][i32(2)], u_0.proj_0.data_0[i32(3)][i32(2)], u_0.proj_0.data_0[i32(0)][i32(3)], u_0.proj_0.data_0[i32(1)][i32(3)], u_0.proj_0.data_0[i32(2)][i32(3)], u_0.proj_0.data_0[i32(3)][i32(3)]))));
        var _S10 : f32 = clip_0.w;
        if(_S10 <= 0.0f)
        {
            i_0 = i_0 + i32(1);
            continue;
        }
        var _S11 : f32 = clip_0.x / _S10 * 0.5f + 0.5f;
        var _S12 : f32 = 0.5f - clip_0.y / _S10 * 0.5f;
        var sampleUV_0 : vec2<f32> = vec2<f32>(_S11, _S12);
        var _S13 : bool;
        if(_S11 < 0.0f)
        {
            _S13 = true;
        }
        else
        {
            _S13 = _S11 > 1.0f;
        }
        var _S14 : bool;
        if(_S13)
        {
            _S14 = true;
        }
        else
        {
            _S14 = _S12 < 0.0f;
        }
        var _S15 : bool;
        if(_S14)
        {
            _S15 = true;
        }
        else
        {
            _S15 = _S12 > 1.0f;
        }
        if(_S15)
        {
            i_0 = i_0 + i32(1);
            continue;
        }
        var sceneZ_0 : f32 = view_pos_at_0(sampleUV_0, load_depth_0(sampleUV_0)).z;
        var rangeCheck_0 : f32 = smoothstep(0.0f, 1.0f, _S8 / max(abs(p_0.z - sceneZ_0), 0.00009999999747379f));
        var occlusion_1 : f32;
        if(sceneZ_0 >= (samplePos_0.z + _S9))
        {
            occlusion_1 = occlusion_0 + rangeCheck_0;
        }
        else
        {
            occlusion_1 = occlusion_0;
        }
        occlusion_0 = occlusion_1;
        i_0 = i_0 + i32(1);
    }
    var _S16 : pixelOutput_0 = pixelOutput_0( vec4<f32>(clamp(1.0f - u_0.params_0.z * (occlusion_0 / 12.0f), 0.0f, 1.0f)) );
    return _S16;
}

struct pixelOutput_1
{
    @location(0) output_1 : vec4<f32>,
};

struct pixelInput_1
{
    @location(0) uv_4 : vec2<f32>,
};

@fragment
fn fs_blur( _S17 : pixelInput_1, @builtin(position) svPosition_2 : vec4<f32>) -> pixelOutput_1
{
    var w_2 : u32;
    var h_2 : u32;
    {var dim = textureDimensions((aoTex_0));((w_2)) = dim.x;((h_2)) = dim.y;};
    var _S18 : vec2<i32> = vec2<i32>(_S17.uv_4 * vec2<f32>(f32(w_2), f32(h_2)));
    var _S19 : vec2<i32> = vec2<i32>(i32(w_2), i32(h_2)) - vec2<i32>(i32(1));
    var y_1 : i32 = i32(-1);
    var total_0 : f32 = 0.0f;
    for(;;)
    {
        if(y_1 <= i32(1))
        {
        }
        else
        {
            break;
        }
        var x_1 : i32 = i32(-1);
        for(;;)
        {
            if(x_1 <= i32(1))
            {
            }
            else
            {
                break;
            }
            var _S20 : vec3<i32> = vec3<i32>(clamp(_S18 + vec2<i32>(x_1, y_1), vec2<i32>(i32(0), i32(0)), _S19), i32(0));
            var total_1 : f32 = total_0 + (textureLoad((aoTex_0), ((_S20)).xy, ((_S20)).z)).x;
            x_1 = x_1 + i32(1);
            total_0 = total_1;
        }
        y_1 = y_1 + i32(1);
    }
    var _S21 : pixelOutput_1 = pixelOutput_1( vec4<f32>((total_0 / 9.0f)) );
    return _S21;
}

