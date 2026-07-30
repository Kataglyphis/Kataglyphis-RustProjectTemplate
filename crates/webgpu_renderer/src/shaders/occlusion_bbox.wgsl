struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct OcclusionUniforms_std140_0
{
    @align(16) view_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
};

@binding(0) @group(0) var<uniform> uniforms_0 : OcclusionUniforms_std140_0;
const corners_0 : array<u32, i32(36)> = array<u32, i32(36)>( u32(0), u32(2), u32(1), u32(1), u32(2), u32(3), u32(4), u32(5), u32(6), u32(5), u32(7), u32(6), u32(0), u32(1), u32(4), u32(1), u32(5), u32(4), u32(2), u32(6), u32(3), u32(3), u32(6), u32(7), u32(0), u32(4), u32(2), u32(2), u32(4), u32(6), u32(1), u32(3), u32(5), u32(3), u32(7), u32(5) );
struct vertexOutput_0
{
    @builtin(position) output_0 : vec4<f32>,
};

struct vertexInput_0
{
    @location(0) aabb_min_0 : vec3<f32>,
    @location(1) aabb_max_0 : vec3<f32>,
};

@vertex
fn vs_main( _S1 : vertexInput_0, @builtin(vertex_index) vertexIndex_0 : u32) -> vertexOutput_0
{
    var _S2 : vertexOutput_0 = vertexOutput_0( (((vec4<f32>(mix(_S1.aabb_min_0, _S1.aabb_max_0, vec3<f32>(f32(((corners_0[vertexIndex_0]) & (u32(1)))), f32(((((corners_0[vertexIndex_0]) >> (u32(1)))) & (u32(1)))), f32(((((corners_0[vertexIndex_0]) >> (u32(2)))) & (u32(1)))))), 1.0f)) * (mat4x4<f32>(uniforms_0.view_proj_0.data_0[i32(0)][i32(0)], uniforms_0.view_proj_0.data_0[i32(1)][i32(0)], uniforms_0.view_proj_0.data_0[i32(2)][i32(0)], uniforms_0.view_proj_0.data_0[i32(3)][i32(0)], uniforms_0.view_proj_0.data_0[i32(0)][i32(1)], uniforms_0.view_proj_0.data_0[i32(1)][i32(1)], uniforms_0.view_proj_0.data_0[i32(2)][i32(1)], uniforms_0.view_proj_0.data_0[i32(3)][i32(1)], uniforms_0.view_proj_0.data_0[i32(0)][i32(2)], uniforms_0.view_proj_0.data_0[i32(1)][i32(2)], uniforms_0.view_proj_0.data_0[i32(2)][i32(2)], uniforms_0.view_proj_0.data_0[i32(3)][i32(2)], uniforms_0.view_proj_0.data_0[i32(0)][i32(3)], uniforms_0.view_proj_0.data_0[i32(1)][i32(3)], uniforms_0.view_proj_0.data_0[i32(2)][i32(3)], uniforms_0.view_proj_0.data_0[i32(3)][i32(3)])))) );
    return _S2;
}

