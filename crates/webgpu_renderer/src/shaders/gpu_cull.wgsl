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

@compute
@workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (params_0.primitive_count_0))
    {
        return;
    }
    var aabb_0 : Aabb_std430_0 = aabbData_0[idx_0];
    var clip_0 : vec4<f32> = (((vec4<f32>((aabb_0.min_0.xyz + aabb_0.max_0.xyz) * vec3<f32>(0.5f), 1.0f)) * (mat4x4<f32>(params_0.view_proj_0.data_0[i32(0)][i32(0)], params_0.view_proj_0.data_0[i32(1)][i32(0)], params_0.view_proj_0.data_0[i32(2)][i32(0)], params_0.view_proj_0.data_0[i32(3)][i32(0)], params_0.view_proj_0.data_0[i32(0)][i32(1)], params_0.view_proj_0.data_0[i32(1)][i32(1)], params_0.view_proj_0.data_0[i32(2)][i32(1)], params_0.view_proj_0.data_0[i32(3)][i32(1)], params_0.view_proj_0.data_0[i32(0)][i32(2)], params_0.view_proj_0.data_0[i32(1)][i32(2)], params_0.view_proj_0.data_0[i32(2)][i32(2)], params_0.view_proj_0.data_0[i32(3)][i32(2)], params_0.view_proj_0.data_0[i32(0)][i32(3)], params_0.view_proj_0.data_0[i32(1)][i32(3)], params_0.view_proj_0.data_0[i32(2)][i32(3)], params_0.view_proj_0.data_0[i32(3)][i32(3)]))));
    var _S1 : f32 = clip_0.w;
    var depth_0 : f32 = clip_0.z / _S1;
    var _S2 : vec2<f32> = vec2<f32>(0.5f);
    var uv_0 : vec2<f32> = clip_0.xy / vec2<f32>(_S1) * _S2 + _S2;
    var _S3 : bool;
    if(depth_0 < 0.0f)
    {
        _S3 = true;
    }
    else
    {
        _S3 = depth_0 > 1.0f;
    }
    if(_S3)
    {
        visibility_0[idx_0] = u32(0);
        return;
    }
    var _S4 : f32 = uv_0.x;
    if(_S4 < 0.0f)
    {
        _S3 = true;
    }
    else
    {
        _S3 = _S4 > 1.0f;
    }
    if(_S3)
    {
        _S3 = true;
    }
    else
    {
        _S3 = (uv_0.y) < 0.0f;
    }
    if(_S3)
    {
        _S3 = true;
    }
    else
    {
        _S3 = (uv_0.y) > 1.0f;
    }
    if(_S3)
    {
        visibility_0[idx_0] = u32(0);
        return;
    }
    var w_0 : u32;
    var h_0 : u32;
    {var dim = textureDimensions((depthTex_0));((w_0)) = dim.x;((h_0)) = dim.y;};
    var _S5 : vec3<i32> = vec3<i32>(vec2<i32>(uv_0 * vec2<f32>(f32(w_0), f32(h_0))), i32(0));
    var _S6 : u32;
    if(depth_0 <= (textureLoad((depthTex_0), ((_S5)).xy, ((_S5)).z)))
    {
        _S6 = u32(1);
    }
    else
    {
        _S6 = u32(0);
    }
    visibility_0[idx_0] = _S6;
    return;
}

