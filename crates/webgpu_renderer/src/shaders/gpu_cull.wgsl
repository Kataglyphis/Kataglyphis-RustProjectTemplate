struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct CullParams_std140_0
{
    @align(16) view_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) inv_view_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) width_0 : f32,
    @align(4) height_0 : f32,
    @align(8) primitive_count_0 : u32,
};

@binding(1) @group(0) var<uniform> params_0 : CullParams_std140_0;
struct Aabb_std430_0
{
    @align(16) min_0 : vec4<f32>,
    @align(16) max_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> aabbData_0 : array<Aabb_std430_0>;

@binding(3) @group(0) var<storage, read_write> visibility_0 : array<u32>;

@binding(0) @group(0) var depthTex_0 : texture_depth_2d;

const UNIT_CORNERS_0 : array<vec3<f32>, i32(8)> = array<vec3<f32>, i32(8)>( vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(1.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f), vec3<f32>(1.0f, 0.0f, 1.0f), vec3<f32>(0.0f, 1.0f, 1.0f), vec3<f32>(1.0f, 1.0f, 1.0f) );
@compute
@workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (params_0.primitive_count_0))
    {
        return;
    }
    var _S1 : Aabb_std430_0 = aabbData_0[idx_0];
    const _S2 : vec2<f32> = vec2<f32>(1.00000001504746622e+30f, 1.00000001504746622e+30f);
    const _S3 : vec2<f32> = vec2<f32>(-1.00000001504746622e+30f, -1.00000001504746622e+30f);
    var aabbNear_0 : f32 = 1.00000001504746622e+30f;
    var uvMin_0 : vec2<f32> = _S2;
    var uvMax_0 : vec2<f32> = _S3;
    var i_0 : u32 = u32(0);
    for(;;)
    {
        if(i_0 < u32(8))
        {
        }
        else
        {
            break;
        }
        var clip_0 : vec4<f32> = (((vec4<f32>(mix(_S1.min_0.xyz, _S1.max_0.xyz, UNIT_CORNERS_0[i_0]), 1.0f)) * (mat4x4<f32>(params_0.view_proj_0.data_0[i32(0)][i32(0)], params_0.view_proj_0.data_0[i32(1)][i32(0)], params_0.view_proj_0.data_0[i32(2)][i32(0)], params_0.view_proj_0.data_0[i32(3)][i32(0)], params_0.view_proj_0.data_0[i32(0)][i32(1)], params_0.view_proj_0.data_0[i32(1)][i32(1)], params_0.view_proj_0.data_0[i32(2)][i32(1)], params_0.view_proj_0.data_0[i32(3)][i32(1)], params_0.view_proj_0.data_0[i32(0)][i32(2)], params_0.view_proj_0.data_0[i32(1)][i32(2)], params_0.view_proj_0.data_0[i32(2)][i32(2)], params_0.view_proj_0.data_0[i32(3)][i32(2)], params_0.view_proj_0.data_0[i32(0)][i32(3)], params_0.view_proj_0.data_0[i32(1)][i32(3)], params_0.view_proj_0.data_0[i32(2)][i32(3)], params_0.view_proj_0.data_0[i32(3)][i32(3)]))));
        var _S4 : f32 = clip_0.w;
        if(_S4 <= 0.0f)
        {
            visibility_0[idx_0] = u32(1);
            return;
        }
        var ndc_0 : vec2<f32> = clip_0.xy / vec2<f32>(_S4);
        var uv_0 : vec2<f32> = vec2<f32>(ndc_0.x * 0.5f + 0.5f, 0.5f - ndc_0.y * 0.5f);
        var _S5 : f32 = min(aabbNear_0, clip_0.z / _S4);
        var _S6 : vec2<f32> = min(uvMin_0, uv_0);
        var _S7 : vec2<f32> = max(uvMax_0, uv_0);
        var i_1 : u32 = i_0 + u32(1);
        aabbNear_0 = _S5;
        uvMin_0 = _S6;
        uvMax_0 = _S7;
        i_0 = i_1;
    }
    const _S8 : vec2<f32> = vec2<f32>(0.0f, 0.0f);
    const _S9 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var uvMin_1 : vec2<f32> = clamp(uvMin_0, _S8, _S9);
    var uvMax_1 : vec2<f32> = clamp(uvMax_0, _S8, _S9);
    var _S10 : bool;
    if((uvMax_1.x) <= (uvMin_1.x))
    {
        _S10 = true;
    }
    else
    {
        _S10 = (uvMax_1.y) <= (uvMin_1.y);
    }
    if(_S10)
    {
        visibility_0[idx_0] = u32(0);
        return;
    }
    var w_0 : u32;
    var h_0 : u32;
    {var dim = textureDimensions((depthTex_0));((w_0)) = dim.x;((h_0)) = dim.y;};
    var _S11 : vec2<i32> = vec2<i32>(i32(w_0) - i32(1), i32(h_0) - i32(1));
    var maxSampled_0 : f32 = 0.0f;
    var ty_0 : u32 = u32(0);
    for(;;)
    {
        if(ty_0 < u32(8))
        {
        }
        else
        {
            break;
        }
        var tx_0 : u32 = u32(0);
        for(;;)
        {
            if(tx_0 < u32(8))
            {
            }
            else
            {
                break;
            }
            var _S12 : vec3<i32> = vec3<i32>(clamp(vec2<i32>(mix(uvMin_1, uvMax_1, (vec2<f32>(f32(tx_0), f32(ty_0)) + vec2<f32>(0.5f)) / vec2<f32>(8.0f)) * vec2<f32>(f32(w_0), f32(h_0))), vec2<i32>(i32(0), i32(0)), _S11), i32(0));
            var _S13 : f32 = max(maxSampled_0, (textureLoad((depthTex_0), ((_S12)).xy, ((_S12)).z)));
            var tx_1 : u32 = tx_0 + u32(1);
            maxSampled_0 = _S13;
            tx_0 = tx_1;
        }
        ty_0 = ty_0 + u32(1);
    }
    if(aabbNear_0 > maxSampled_0)
    {
        i_0 = u32(0);
    }
    else
    {
        i_0 = u32(1);
    }
    visibility_0[idx_0] = i_0;
    return;
}

