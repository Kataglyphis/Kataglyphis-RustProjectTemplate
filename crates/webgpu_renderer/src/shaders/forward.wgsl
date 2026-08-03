struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct PrimUniforms_std140_0
{
    @align(16) model_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) normal_matrix_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) base_color_0 : vec4<f32>,
    @align(16) material_factors_0 : vec4<f32>,
    @align(16) emissive_factor_0 : vec4<f32>,
    @align(16) base_uv_row0_0 : vec4<f32>,
    @align(16) base_uv_row1_0 : vec4<f32>,
    @align(16) material_flags_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> prim_0 : PrimUniforms_std140_0;
struct _MatrixStorage_float4x4_ColMajorstd430_0
{
    @align(16) data_1 : array<vec4<f32>, i32(4)>,
};

@binding(13) @group(0) var<storage, read> jointMatrices_0 : array<_MatrixStorage_float4x4_ColMajorstd430_0>;

struct FrameUniforms_std140_0
{
    @align(16) view_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) light_space_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) light_space_1_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) light_space_2_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) light_dir_ambient_0 : vec4<f32>,
    @align(16) light_color_intensity_0 : vec4<f32>,
    @align(16) camera_position_0 : vec4<f32>,
    @align(16) cascade_splits_0 : vec4<f32>,
};

@binding(0) @group(2) var<uniform> frame_0 : FrameUniforms_std140_0;
@binding(0) @group(1) var<uniform> shadowCascadeIndex_0 : vec4<u32>;
@binding(3) @group(0) var baseColorTex_0 : texture_2d<f32>;

@binding(4) @group(0) var baseColorSampler_0 : sampler;

@binding(5) @group(0) var metalRoughTex_0 : texture_2d<f32>;

@binding(6) @group(0) var metalRoughSampler_0 : sampler;

@binding(7) @group(0) var normalTex_0 : texture_2d<f32>;

@binding(8) @group(0) var normalSampler_0 : sampler;

@binding(9) @group(0) var emissiveTex_0 : texture_2d<f32>;

@binding(10) @group(0) var emissiveSampler_0 : sampler;

@binding(11) @group(0) var occlusionTex_0 : texture_2d<f32>;

@binding(12) @group(0) var occlusionSampler_0 : sampler;

@binding(1) @group(0) var shadowMap_0 : texture_depth_2d_array;

@binding(2) @group(0) var shadowSampler_0 : sampler_comparison;

@binding(0) @group(4) var<storage, read> tileLightGrid_0 : array<vec2<u32>>;

@binding(1) @group(4) var<storage, read> tileLightIndices_0 : array<u32>;

@binding(0) @group(3) var<storage, read> punctualLights_0 : array<vec4<f32>>;

struct IblParams_std140_0
{
    @align(16) enabled_maxmip_intensity_0 : vec4<f32>,
};

@binding(0) @group(1) var<uniform> iblParams_0 : IblParams_std140_0;
@binding(1) @group(1) var irradianceMap_0 : texture_cube<f32>;

@binding(4) @group(1) var iblSampler_0 : sampler;

@binding(2) @group(1) var prefilteredMap_0 : texture_cube<f32>;

@binding(3) @group(1) var brdfLut_0 : texture_2d<f32>;

struct VsIn_0
{
     position_0 : vec3<f32>,
     normal_0 : vec3<f32>,
     uv_0 : vec2<f32>,
     tangent_0 : vec4<f32>,
     joints_0 : vec4<f32>,
     weights_0 : vec4<f32>,
     color_0 : vec4<f32>,
     uv1_0 : vec2<f32>,
     instance0_0 : vec4<f32>,
     instance1_0 : vec4<f32>,
     instance2_0 : vec4<f32>,
     instance3_0 : vec4<f32>,
};

fn instance_matrix_0( In_0 : VsIn_0) -> mat4x4<f32>
{
    return mat4x4<f32>(In_0.instance0_0, In_0.instance1_0, In_0.instance2_0, In_0.instance3_0);
}

fn skin_matrix_0( In_1 : VsIn_0) -> mat4x4<f32>
{
    var _S1 : f32 = In_1.weights_0.x;
    var _S2 : f32 = In_1.weights_0.y;
    var _S3 : f32 = In_1.weights_0.z;
    var _S4 : f32 = In_1.weights_0.w;
    var total_0 : f32 = _S1 + _S2 + _S3 + _S4;
    if(total_0 <= 0.00009999999747379f)
    {
        return mat4x4<f32>(prim_0.model_0.data_0[i32(0)][i32(0)], prim_0.model_0.data_0[i32(1)][i32(0)], prim_0.model_0.data_0[i32(2)][i32(0)], prim_0.model_0.data_0[i32(3)][i32(0)], prim_0.model_0.data_0[i32(0)][i32(1)], prim_0.model_0.data_0[i32(1)][i32(1)], prim_0.model_0.data_0[i32(2)][i32(1)], prim_0.model_0.data_0[i32(3)][i32(1)], prim_0.model_0.data_0[i32(0)][i32(2)], prim_0.model_0.data_0[i32(1)][i32(2)], prim_0.model_0.data_0[i32(2)][i32(2)], prim_0.model_0.data_0[i32(3)][i32(2)], prim_0.model_0.data_0[i32(0)][i32(3)], prim_0.model_0.data_0[i32(1)][i32(3)], prim_0.model_0.data_0[i32(2)][i32(3)], prim_0.model_0.data_0[i32(3)][i32(3)]);
    }
    var _S5 : mat4x4<f32> = mat4x4<f32>(jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(0)][i32(0)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(1)][i32(0)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(2)][i32(0)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(3)][i32(0)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(0)][i32(1)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(1)][i32(1)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(2)][i32(1)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(3)][i32(1)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(0)][i32(2)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(1)][i32(2)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(2)][i32(2)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(3)][i32(2)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(0)][i32(3)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(1)][i32(3)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(2)][i32(3)], jointMatrices_0[u32(In_1.joints_0.x)].data_1[i32(3)][i32(3)]);
    var _S6 : mat4x4<f32> = mat4x4<f32>(_S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1, _S1);
    var _S7 : mat4x4<f32> = mat4x4<f32>(jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(0)][i32(0)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(1)][i32(0)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(2)][i32(0)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(3)][i32(0)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(0)][i32(1)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(1)][i32(1)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(2)][i32(1)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(3)][i32(1)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(0)][i32(2)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(1)][i32(2)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(2)][i32(2)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(3)][i32(2)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(0)][i32(3)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(1)][i32(3)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(2)][i32(3)], jointMatrices_0[u32(In_1.joints_0.y)].data_1[i32(3)][i32(3)]);
    var _S8 : mat4x4<f32> = mat4x4<f32>(_S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2, _S2);
    var _S9 : mat4x4<f32> = mat4x4<f32>(jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(0)][i32(0)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(1)][i32(0)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(2)][i32(0)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(3)][i32(0)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(0)][i32(1)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(1)][i32(1)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(2)][i32(1)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(3)][i32(1)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(0)][i32(2)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(1)][i32(2)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(2)][i32(2)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(3)][i32(2)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(0)][i32(3)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(1)][i32(3)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(2)][i32(3)], jointMatrices_0[u32(In_1.joints_0.z)].data_1[i32(3)][i32(3)]);
    var _S10 : mat4x4<f32> = mat4x4<f32>(_S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3, _S3);
    var _S11 : mat4x4<f32> = mat4x4<f32>(jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(0)][i32(0)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(1)][i32(0)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(2)][i32(0)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(3)][i32(0)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(0)][i32(1)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(1)][i32(1)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(2)][i32(1)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(3)][i32(1)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(0)][i32(2)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(1)][i32(2)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(2)][i32(2)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(3)][i32(2)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(0)][i32(3)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(1)][i32(3)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(2)][i32(3)], jointMatrices_0[u32(In_1.joints_0.w)].data_1[i32(3)][i32(3)]);
    var _S12 : mat4x4<f32> = mat4x4<f32>(_S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4, _S4);
    var _S13 : mat4x4<f32> = mat4x4<f32>(_S5[0] * _S6[0], _S5[1] * _S6[1], _S5[2] * _S6[2], _S5[3] * _S6[3]) + mat4x4<f32>(_S7[0] * _S8[0], _S7[1] * _S8[1], _S7[2] * _S8[2], _S7[3] * _S8[3]) + mat4x4<f32>(_S9[0] * _S10[0], _S9[1] * _S10[1], _S9[2] * _S10[2], _S9[3] * _S10[3]) + mat4x4<f32>(_S11[0] * _S12[0], _S11[1] * _S12[1], _S11[2] * _S12[2], _S11[3] * _S12[3]);
    var _S14 : mat4x4<f32> = mat4x4<f32>(1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0, 1.0f / total_0);
    return mat4x4<f32>(_S13[0] * _S14[0], _S13[1] * _S14[1], _S13[2] * _S14[2], _S13[3] * _S14[3]);
}

fn instance_cofactor_0( In_2 : VsIn_0) -> mat3x3<f32>
{
    var m_0 : mat4x4<f32> = instance_matrix_0(In_2);
    var c0_0 : vec3<f32> = m_0[i32(0)].xyz;
    var c1_0 : vec3<f32> = m_0[i32(1)].xyz;
    var c2_0 : vec3<f32> = m_0[i32(2)].xyz;
    return mat3x3<f32>(cross(c1_0, c2_0), cross(c2_0, c0_0), cross(c0_0, c1_0));
}

struct VsOut_0
{
    @builtin(position) svPosition_0 : vec4<f32>,
    @location(0) worldNormal_0 : vec3<f32>,
    @location(1) uv_1 : vec2<f32>,
    @location(2) lightSpacePos_0 : vec4<f32>,
    @location(3) worldTangent_0 : vec4<f32>,
    @location(4) worldPosition_0 : vec3<f32>,
    @location(5) viewDepth_0 : f32,
    @location(6) vertexColor_0 : vec4<f32>,
    @location(7) uv1_1 : vec2<f32>,
};

struct vertexInput_0
{
    @location(0) position_1 : vec3<f32>,
    @location(4) normal_1 : vec3<f32>,
    @location(5) uv_2 : vec2<f32>,
    @location(6) tangent_1 : vec4<f32>,
    @location(7) joints_1 : vec4<f32>,
    @location(8) weights_1 : vec4<f32>,
    @location(9) color_1 : vec4<f32>,
    @location(1) uv1_2 : vec2<f32>,
    @location(10) instance0_1 : vec4<f32>,
    @location(11) instance1_1 : vec4<f32>,
    @location(2) instance2_1 : vec4<f32>,
    @location(3) instance3_1 : vec4<f32>,
};

@vertex
fn vs_main( _S15 : vertexInput_0) -> VsOut_0
{
    var _S16 : VsIn_0 = VsIn_0( _S15.position_1, _S15.normal_1, _S15.uv_2, _S15.tangent_1, _S15.joints_1, _S15.weights_1, _S15.color_1, _S15.uv1_2, _S15.instance0_1, _S15.instance1_1, _S15.instance2_1, _S15.instance3_1 );
    var model_1 : mat4x4<f32> = (((skin_matrix_0(_S16)) * (instance_matrix_0(_S16))));
    var worldPos_0 : vec4<f32> = (((vec4<f32>(_S15.position_1, 1.0f)) * (model_1)));
    var o_0 : VsOut_0;
    o_0.svPosition_0 = (((worldPos_0) * (mat4x4<f32>(frame_0.view_proj_0.data_0[i32(0)][i32(0)], frame_0.view_proj_0.data_0[i32(1)][i32(0)], frame_0.view_proj_0.data_0[i32(2)][i32(0)], frame_0.view_proj_0.data_0[i32(3)][i32(0)], frame_0.view_proj_0.data_0[i32(0)][i32(1)], frame_0.view_proj_0.data_0[i32(1)][i32(1)], frame_0.view_proj_0.data_0[i32(2)][i32(1)], frame_0.view_proj_0.data_0[i32(3)][i32(1)], frame_0.view_proj_0.data_0[i32(0)][i32(2)], frame_0.view_proj_0.data_0[i32(1)][i32(2)], frame_0.view_proj_0.data_0[i32(2)][i32(2)], frame_0.view_proj_0.data_0[i32(3)][i32(2)], frame_0.view_proj_0.data_0[i32(0)][i32(3)], frame_0.view_proj_0.data_0[i32(1)][i32(3)], frame_0.view_proj_0.data_0[i32(2)][i32(3)], frame_0.view_proj_0.data_0[i32(3)][i32(3)]))));
    if((_S15.weights_1.x + _S15.weights_1.y + _S15.weights_1.z + _S15.weights_1.w) > 0.00009999999747379f)
    {
        o_0.worldNormal_0 = normalize((((vec4<f32>(_S15.normal_1, 0.0f)) * (model_1))).xyz);
    }
    else
    {
        o_0.worldNormal_0 = normalize(((((((vec4<f32>(_S15.normal_1, 0.0f)) * (mat4x4<f32>(prim_0.normal_matrix_0.data_0[i32(0)][i32(0)], prim_0.normal_matrix_0.data_0[i32(1)][i32(0)], prim_0.normal_matrix_0.data_0[i32(2)][i32(0)], prim_0.normal_matrix_0.data_0[i32(3)][i32(0)], prim_0.normal_matrix_0.data_0[i32(0)][i32(1)], prim_0.normal_matrix_0.data_0[i32(1)][i32(1)], prim_0.normal_matrix_0.data_0[i32(2)][i32(1)], prim_0.normal_matrix_0.data_0[i32(3)][i32(1)], prim_0.normal_matrix_0.data_0[i32(0)][i32(2)], prim_0.normal_matrix_0.data_0[i32(1)][i32(2)], prim_0.normal_matrix_0.data_0[i32(2)][i32(2)], prim_0.normal_matrix_0.data_0[i32(3)][i32(2)], prim_0.normal_matrix_0.data_0[i32(0)][i32(3)], prim_0.normal_matrix_0.data_0[i32(1)][i32(3)], prim_0.normal_matrix_0.data_0[i32(2)][i32(3)], prim_0.normal_matrix_0.data_0[i32(3)][i32(3)])))).xyz) * (instance_cofactor_0(_S16)))));
    }
    o_0.worldTangent_0 = vec4<f32>(normalize((((vec4<f32>(_S15.tangent_1.xyz, 0.0f)) * (model_1))).xyz), _S15.tangent_1.w);
    o_0.uv_1 = _S15.uv_2;
    o_0.lightSpacePos_0 = (((worldPos_0) * (mat4x4<f32>(frame_0.light_space_0.data_0[i32(0)][i32(0)], frame_0.light_space_0.data_0[i32(1)][i32(0)], frame_0.light_space_0.data_0[i32(2)][i32(0)], frame_0.light_space_0.data_0[i32(3)][i32(0)], frame_0.light_space_0.data_0[i32(0)][i32(1)], frame_0.light_space_0.data_0[i32(1)][i32(1)], frame_0.light_space_0.data_0[i32(2)][i32(1)], frame_0.light_space_0.data_0[i32(3)][i32(1)], frame_0.light_space_0.data_0[i32(0)][i32(2)], frame_0.light_space_0.data_0[i32(1)][i32(2)], frame_0.light_space_0.data_0[i32(2)][i32(2)], frame_0.light_space_0.data_0[i32(3)][i32(2)], frame_0.light_space_0.data_0[i32(0)][i32(3)], frame_0.light_space_0.data_0[i32(1)][i32(3)], frame_0.light_space_0.data_0[i32(2)][i32(3)], frame_0.light_space_0.data_0[i32(3)][i32(3)]))));
    var _S17 : vec3<f32> = worldPos_0.xyz;
    o_0.worldPosition_0 = _S17;
    o_0.viewDepth_0 = distance(_S17, frame_0.camera_position_0.xyz);
    o_0.vertexColor_0 = _S15.color_1;
    o_0.uv1_1 = _S15.uv1_2;
    return o_0;
}

struct vertexOutput_0
{
    @builtin(position) output_0 : vec4<f32>,
};

struct vertexInput_1
{
    @location(0) position_2 : vec3<f32>,
    @location(4) normal_2 : vec3<f32>,
    @location(5) uv_3 : vec2<f32>,
    @location(6) tangent_2 : vec4<f32>,
    @location(7) joints_2 : vec4<f32>,
    @location(8) weights_2 : vec4<f32>,
    @location(9) color_2 : vec4<f32>,
    @location(1) uv1_3 : vec2<f32>,
    @location(10) instance0_2 : vec4<f32>,
    @location(11) instance1_2 : vec4<f32>,
    @location(2) instance2_2 : vec4<f32>,
    @location(3) instance3_2 : vec4<f32>,
};

@vertex
fn vs_shadow( _S18 : vertexInput_1) -> vertexOutput_0
{
    var _S19 : VsIn_0 = VsIn_0( _S18.position_2, _S18.normal_2, _S18.uv_3, _S18.tangent_2, _S18.joints_2, _S18.weights_2, _S18.color_2, _S18.uv1_3, _S18.instance0_2, _S18.instance1_2, _S18.instance2_2, _S18.instance3_2 );
    var world_0 : vec4<f32> = (((vec4<f32>(_S18.position_2, 1.0f)) * ((((skin_matrix_0(_S19)) * (instance_matrix_0(_S19)))))));
    var cascade_0 : u32 = shadowCascadeIndex_0.x;
    if(cascade_0 == u32(1))
    {
        var _S20 : vertexOutput_0 = vertexOutput_0( (((world_0) * (mat4x4<f32>(frame_0.light_space_1_0.data_0[i32(0)][i32(0)], frame_0.light_space_1_0.data_0[i32(1)][i32(0)], frame_0.light_space_1_0.data_0[i32(2)][i32(0)], frame_0.light_space_1_0.data_0[i32(3)][i32(0)], frame_0.light_space_1_0.data_0[i32(0)][i32(1)], frame_0.light_space_1_0.data_0[i32(1)][i32(1)], frame_0.light_space_1_0.data_0[i32(2)][i32(1)], frame_0.light_space_1_0.data_0[i32(3)][i32(1)], frame_0.light_space_1_0.data_0[i32(0)][i32(2)], frame_0.light_space_1_0.data_0[i32(1)][i32(2)], frame_0.light_space_1_0.data_0[i32(2)][i32(2)], frame_0.light_space_1_0.data_0[i32(3)][i32(2)], frame_0.light_space_1_0.data_0[i32(0)][i32(3)], frame_0.light_space_1_0.data_0[i32(1)][i32(3)], frame_0.light_space_1_0.data_0[i32(2)][i32(3)], frame_0.light_space_1_0.data_0[i32(3)][i32(3)])))) );
        return _S20;
    }
    if(cascade_0 == u32(2))
    {
        var _S21 : vertexOutput_0 = vertexOutput_0( (((world_0) * (mat4x4<f32>(frame_0.light_space_2_0.data_0[i32(0)][i32(0)], frame_0.light_space_2_0.data_0[i32(1)][i32(0)], frame_0.light_space_2_0.data_0[i32(2)][i32(0)], frame_0.light_space_2_0.data_0[i32(3)][i32(0)], frame_0.light_space_2_0.data_0[i32(0)][i32(1)], frame_0.light_space_2_0.data_0[i32(1)][i32(1)], frame_0.light_space_2_0.data_0[i32(2)][i32(1)], frame_0.light_space_2_0.data_0[i32(3)][i32(1)], frame_0.light_space_2_0.data_0[i32(0)][i32(2)], frame_0.light_space_2_0.data_0[i32(1)][i32(2)], frame_0.light_space_2_0.data_0[i32(2)][i32(2)], frame_0.light_space_2_0.data_0[i32(3)][i32(2)], frame_0.light_space_2_0.data_0[i32(0)][i32(3)], frame_0.light_space_2_0.data_0[i32(1)][i32(3)], frame_0.light_space_2_0.data_0[i32(2)][i32(3)], frame_0.light_space_2_0.data_0[i32(3)][i32(3)])))) );
        return _S21;
    }
    var _S22 : vertexOutput_0 = vertexOutput_0( (((world_0) * (mat4x4<f32>(frame_0.light_space_0.data_0[i32(0)][i32(0)], frame_0.light_space_0.data_0[i32(1)][i32(0)], frame_0.light_space_0.data_0[i32(2)][i32(0)], frame_0.light_space_0.data_0[i32(3)][i32(0)], frame_0.light_space_0.data_0[i32(0)][i32(1)], frame_0.light_space_0.data_0[i32(1)][i32(1)], frame_0.light_space_0.data_0[i32(2)][i32(1)], frame_0.light_space_0.data_0[i32(3)][i32(1)], frame_0.light_space_0.data_0[i32(0)][i32(2)], frame_0.light_space_0.data_0[i32(1)][i32(2)], frame_0.light_space_0.data_0[i32(2)][i32(2)], frame_0.light_space_0.data_0[i32(3)][i32(2)], frame_0.light_space_0.data_0[i32(0)][i32(3)], frame_0.light_space_0.data_0[i32(1)][i32(3)], frame_0.light_space_0.data_0[i32(2)][i32(3)], frame_0.light_space_0.data_0[i32(3)][i32(3)])))) );
    return _S22;
}

struct VsShadowMaskedOut_0
{
    @builtin(position) pos_0 : vec4<f32>,
    @location(0) uv_4 : vec2<f32>,
};

struct vertexInput_2
{
    @location(0) position_3 : vec3<f32>,
    @location(4) normal_3 : vec3<f32>,
    @location(5) uv_5 : vec2<f32>,
    @location(6) tangent_3 : vec4<f32>,
    @location(7) joints_3 : vec4<f32>,
    @location(8) weights_3 : vec4<f32>,
    @location(9) color_3 : vec4<f32>,
    @location(1) uv1_4 : vec2<f32>,
    @location(10) instance0_3 : vec4<f32>,
    @location(11) instance1_3 : vec4<f32>,
    @location(2) instance2_3 : vec4<f32>,
    @location(3) instance3_3 : vec4<f32>,
};

@vertex
fn vs_shadow_masked( _S23 : vertexInput_2) -> VsShadowMaskedOut_0
{
    var _S24 : VsIn_0 = VsIn_0( _S23.position_3, _S23.normal_3, _S23.uv_5, _S23.tangent_3, _S23.joints_3, _S23.weights_3, _S23.color_3, _S23.uv1_4, _S23.instance0_3, _S23.instance1_3, _S23.instance2_3, _S23.instance3_3 );
    var o_1 : VsShadowMaskedOut_0;
    var world_1 : vec4<f32> = (((vec4<f32>(_S23.position_3, 1.0f)) * ((((skin_matrix_0(_S24)) * (instance_matrix_0(_S24)))))));
    var cascade_1 : u32 = shadowCascadeIndex_0.x;
    if(cascade_1 == u32(1))
    {
        o_1.pos_0 = (((world_1) * (mat4x4<f32>(frame_0.light_space_1_0.data_0[i32(0)][i32(0)], frame_0.light_space_1_0.data_0[i32(1)][i32(0)], frame_0.light_space_1_0.data_0[i32(2)][i32(0)], frame_0.light_space_1_0.data_0[i32(3)][i32(0)], frame_0.light_space_1_0.data_0[i32(0)][i32(1)], frame_0.light_space_1_0.data_0[i32(1)][i32(1)], frame_0.light_space_1_0.data_0[i32(2)][i32(1)], frame_0.light_space_1_0.data_0[i32(3)][i32(1)], frame_0.light_space_1_0.data_0[i32(0)][i32(2)], frame_0.light_space_1_0.data_0[i32(1)][i32(2)], frame_0.light_space_1_0.data_0[i32(2)][i32(2)], frame_0.light_space_1_0.data_0[i32(3)][i32(2)], frame_0.light_space_1_0.data_0[i32(0)][i32(3)], frame_0.light_space_1_0.data_0[i32(1)][i32(3)], frame_0.light_space_1_0.data_0[i32(2)][i32(3)], frame_0.light_space_1_0.data_0[i32(3)][i32(3)]))));
    }
    else
    {
        if(cascade_1 == u32(2))
        {
            o_1.pos_0 = (((world_1) * (mat4x4<f32>(frame_0.light_space_2_0.data_0[i32(0)][i32(0)], frame_0.light_space_2_0.data_0[i32(1)][i32(0)], frame_0.light_space_2_0.data_0[i32(2)][i32(0)], frame_0.light_space_2_0.data_0[i32(3)][i32(0)], frame_0.light_space_2_0.data_0[i32(0)][i32(1)], frame_0.light_space_2_0.data_0[i32(1)][i32(1)], frame_0.light_space_2_0.data_0[i32(2)][i32(1)], frame_0.light_space_2_0.data_0[i32(3)][i32(1)], frame_0.light_space_2_0.data_0[i32(0)][i32(2)], frame_0.light_space_2_0.data_0[i32(1)][i32(2)], frame_0.light_space_2_0.data_0[i32(2)][i32(2)], frame_0.light_space_2_0.data_0[i32(3)][i32(2)], frame_0.light_space_2_0.data_0[i32(0)][i32(3)], frame_0.light_space_2_0.data_0[i32(1)][i32(3)], frame_0.light_space_2_0.data_0[i32(2)][i32(3)], frame_0.light_space_2_0.data_0[i32(3)][i32(3)]))));
        }
        else
        {
            o_1.pos_0 = (((world_1) * (mat4x4<f32>(frame_0.light_space_0.data_0[i32(0)][i32(0)], frame_0.light_space_0.data_0[i32(1)][i32(0)], frame_0.light_space_0.data_0[i32(2)][i32(0)], frame_0.light_space_0.data_0[i32(3)][i32(0)], frame_0.light_space_0.data_0[i32(0)][i32(1)], frame_0.light_space_0.data_0[i32(1)][i32(1)], frame_0.light_space_0.data_0[i32(2)][i32(1)], frame_0.light_space_0.data_0[i32(3)][i32(1)], frame_0.light_space_0.data_0[i32(0)][i32(2)], frame_0.light_space_0.data_0[i32(1)][i32(2)], frame_0.light_space_0.data_0[i32(2)][i32(2)], frame_0.light_space_0.data_0[i32(3)][i32(2)], frame_0.light_space_0.data_0[i32(0)][i32(3)], frame_0.light_space_0.data_0[i32(1)][i32(3)], frame_0.light_space_0.data_0[i32(2)][i32(3)], frame_0.light_space_0.data_0[i32(3)][i32(3)]))));
        }
    }
    o_1.uv_4 = _S23.uv_5;
    return o_1;
}

struct pixelInput_0
{
    @location(0) uv_6 : vec2<f32>,
};

@fragment
fn fs_shadow_masked( _S25 : pixelInput_0, @builtin(position) pos_1 : vec4<f32>)
{
    var _S26 : f32 = _S25.uv_6.x;
    var _S27 : f32 = _S25.uv_6.y;
    if((prim_0.base_color_0.w * (textureSample((baseColorTex_0), (baseColorSampler_0), (vec2<f32>(prim_0.base_uv_row0_0.x * _S26 + prim_0.base_uv_row0_0.y * _S27 + prim_0.base_uv_row0_0.z, prim_0.base_uv_row1_0.x * _S26 + prim_0.base_uv_row1_0.y * _S27 + prim_0.base_uv_row1_0.z)), (vec2<i32>(i32(0))))).w) < (prim_0.emissive_factor_0.w))
    {
        discard;
    }
    return;
}

fn distribution_ggx_0( n_dot_h_0 : f32,  roughness_0 : f32) -> f32
{
    var a_0 : f32 = roughness_0 * roughness_0;
    var a2_0 : f32 = a_0 * a_0;
    var denom_0 : f32 = n_dot_h_0 * n_dot_h_0 * (a2_0 - 1.0f) + 1.0f;
    return a2_0 / max(3.14159274101257324f * denom_0 * denom_0, 9.99999997475242708e-07f);
}

fn geometry_smith_0( n_dot_v_0 : f32,  n_dot_l_0 : f32,  roughness_1 : f32) -> f32
{
    var r_0 : f32 = roughness_1 + 1.0f;
    var k_0 : f32 = r_0 * r_0 / 8.0f;
    var _S28 : f32 = 1.0f - k_0;
    return n_dot_v_0 / max(n_dot_v_0 * _S28 + k_0, 9.99999997475242708e-07f) * (n_dot_l_0 / max(n_dot_l_0 * _S28 + k_0, 9.99999997475242708e-07f));
}

fn fresnel_schlick_0( cos_theta_0 : f32,  f0_0 : vec3<f32>) -> vec3<f32>
{
    return f0_0 + (vec3<f32>(1.0f) - f0_0) * vec3<f32>(pow(clamp(1.0f - cos_theta_0, 0.0f, 1.0f), 5.0f));
}

fn lambert_diffuse_0( albedo_0 : vec3<f32>) -> vec3<f32>
{
    return albedo_0 / vec3<f32>(3.14159274101257324f);
}

fn brdf_direct_0( n_0 : vec3<f32>,  v_0 : vec3<f32>,  l_0 : vec3<f32>,  albedo_1 : vec3<f32>,  metallic_0 : f32,  roughness_2 : f32,  f0_1 : vec3<f32>,  radiance_0 : vec3<f32>) -> vec3<f32>
{
    var h_0 : vec3<f32> = normalize(l_0 + v_0);
    var _S29 : f32 = max(dot(n_0, l_0), 0.0f);
    var _S30 : f32 = max(dot(n_0, v_0), 0.00009999999747379f);
    var f_0 : vec3<f32> = fresnel_schlick_0(max(dot(h_0, v_0), 0.0f), f0_1);
    return ((vec3<f32>(1.0f) - f_0) * vec3<f32>((1.0f - metallic_0)) * lambert_diffuse_0(albedo_1) + vec3<f32>((distribution_ggx_0(max(dot(n_0, h_0), 0.0f), roughness_2) * geometry_smith_0(_S30, _S29, roughness_2))) * f_0 / vec3<f32>(max(4.0f * _S30 * _S29, 9.99999997475242708e-07f))) * radiance_0 * vec3<f32>(_S29);
}

fn light_space_for_0( cascade_2 : i32) -> mat4x4<f32>
{
    if(cascade_2 == i32(1))
    {
        return mat4x4<f32>(frame_0.light_space_1_0.data_0[i32(0)][i32(0)], frame_0.light_space_1_0.data_0[i32(1)][i32(0)], frame_0.light_space_1_0.data_0[i32(2)][i32(0)], frame_0.light_space_1_0.data_0[i32(3)][i32(0)], frame_0.light_space_1_0.data_0[i32(0)][i32(1)], frame_0.light_space_1_0.data_0[i32(1)][i32(1)], frame_0.light_space_1_0.data_0[i32(2)][i32(1)], frame_0.light_space_1_0.data_0[i32(3)][i32(1)], frame_0.light_space_1_0.data_0[i32(0)][i32(2)], frame_0.light_space_1_0.data_0[i32(1)][i32(2)], frame_0.light_space_1_0.data_0[i32(2)][i32(2)], frame_0.light_space_1_0.data_0[i32(3)][i32(2)], frame_0.light_space_1_0.data_0[i32(0)][i32(3)], frame_0.light_space_1_0.data_0[i32(1)][i32(3)], frame_0.light_space_1_0.data_0[i32(2)][i32(3)], frame_0.light_space_1_0.data_0[i32(3)][i32(3)]);
    }
    if(cascade_2 == i32(2))
    {
        return mat4x4<f32>(frame_0.light_space_2_0.data_0[i32(0)][i32(0)], frame_0.light_space_2_0.data_0[i32(1)][i32(0)], frame_0.light_space_2_0.data_0[i32(2)][i32(0)], frame_0.light_space_2_0.data_0[i32(3)][i32(0)], frame_0.light_space_2_0.data_0[i32(0)][i32(1)], frame_0.light_space_2_0.data_0[i32(1)][i32(1)], frame_0.light_space_2_0.data_0[i32(2)][i32(1)], frame_0.light_space_2_0.data_0[i32(3)][i32(1)], frame_0.light_space_2_0.data_0[i32(0)][i32(2)], frame_0.light_space_2_0.data_0[i32(1)][i32(2)], frame_0.light_space_2_0.data_0[i32(2)][i32(2)], frame_0.light_space_2_0.data_0[i32(3)][i32(2)], frame_0.light_space_2_0.data_0[i32(0)][i32(3)], frame_0.light_space_2_0.data_0[i32(1)][i32(3)], frame_0.light_space_2_0.data_0[i32(2)][i32(3)], frame_0.light_space_2_0.data_0[i32(3)][i32(3)]);
    }
    return mat4x4<f32>(frame_0.light_space_0.data_0[i32(0)][i32(0)], frame_0.light_space_0.data_0[i32(1)][i32(0)], frame_0.light_space_0.data_0[i32(2)][i32(0)], frame_0.light_space_0.data_0[i32(3)][i32(0)], frame_0.light_space_0.data_0[i32(0)][i32(1)], frame_0.light_space_0.data_0[i32(1)][i32(1)], frame_0.light_space_0.data_0[i32(2)][i32(1)], frame_0.light_space_0.data_0[i32(3)][i32(1)], frame_0.light_space_0.data_0[i32(0)][i32(2)], frame_0.light_space_0.data_0[i32(1)][i32(2)], frame_0.light_space_0.data_0[i32(2)][i32(2)], frame_0.light_space_0.data_0[i32(3)][i32(2)], frame_0.light_space_0.data_0[i32(0)][i32(3)], frame_0.light_space_0.data_0[i32(1)][i32(3)], frame_0.light_space_0.data_0[i32(2)][i32(3)], frame_0.light_space_0.data_0[i32(3)][i32(3)]);
}

fn shadow_visibility_0( viewDepth_1 : f32,  worldPos_1 : vec3<f32>,  nDotL_0 : f32) -> f32
{
    var selected_0 : i32;
    if(viewDepth_1 > (frame_0.cascade_splits_0.x))
    {
        selected_0 = i32(1);
    }
    else
    {
        selected_0 = i32(0);
    }
    if(viewDepth_1 > (frame_0.cascade_splits_0.y))
    {
        selected_0 = i32(2);
    }
    if(selected_0 > i32(2))
    {
        selected_0 = i32(2);
    }
    var inBounds_0 : bool;
    const _S31 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var uv_7 : vec2<f32> = vec2<f32>(0.0f, 0.0f);
    var proj_0 : vec3<f32> = _S31;
    var cascade_3 : i32 = selected_0;
    for(;;)
    {
        if(cascade_3 < i32(3))
        {
        }
        else
        {
            inBounds_0 = false;
            break;
        }
        var lightSpacePos_1 : vec4<f32> = (((vec4<f32>(worldPos_1, 1.0f)) * (light_space_for_0(cascade_3))));
        var proj_1 : vec3<f32> = lightSpacePos_1.xyz / vec3<f32>(lightSpacePos_1.w);
        var _S32 : f32 = proj_1.x * 0.5f + 0.5f;
        var _S33 : f32 = 0.5f - proj_1.y * 0.5f;
        var uv_8 : vec2<f32> = vec2<f32>(_S32, _S33);
        if(_S32 >= 0.0f)
        {
            inBounds_0 = _S32 <= 1.0f;
        }
        else
        {
            inBounds_0 = false;
        }
        var _S34 : bool;
        if(inBounds_0)
        {
            _S34 = _S33 >= 0.0f;
        }
        else
        {
            _S34 = false;
        }
        var _S35 : bool;
        if(_S34)
        {
            _S35 = _S33 <= 1.0f;
        }
        else
        {
            _S35 = false;
        }
        var _S36 : bool;
        if(_S35)
        {
            _S36 = (proj_1.z) <= 1.0f;
        }
        else
        {
            _S36 = false;
        }
        if(_S36)
        {
            inBounds_0 = true;
            uv_7 = uv_8;
            proj_0 = proj_1;
            break;
        }
        var _S37 : i32 = cascade_3 + i32(1);
        uv_7 = uv_8;
        proj_0 = proj_1;
        cascade_3 = _S37;
    }
    if(!inBounds_0)
    {
        return 1.0f;
    }
    var _S38 : f32 = clamp(0.0020000000949949f * (1.0f - nDotL_0) + 0.00050000002374873f, 0.00050000002374873f, 0.00400000018998981f);
    var shadowWidth_0 : u32;
    var shadowHeight_0 : u32;
    var shadowLayers_0 : u32;
    {var dim = textureDimensions((shadowMap_0));((shadowWidth_0)) = dim.x;((shadowHeight_0)) = dim.y;((shadowLayers_0)) = textureNumLayers((shadowMap_0));};
    var _S39 : f32 = 1.0f / f32(shadowWidth_0);
    var y_0 : i32 = i32(-1);
    var sum_0 : f32 = 0.0f;
    for(;;)
    {
        if(y_0 <= i32(1))
        {
        }
        else
        {
            break;
        }
        var x_0 : i32 = i32(-1);
        for(;;)
        {
            if(x_0 <= i32(1))
            {
            }
            else
            {
                break;
            }
            var _S40 : vec3<f32> = vec3<f32>(uv_7 + vec2<f32>(f32(x_0), f32(y_0)) * vec2<f32>(_S39), f32(cascade_3));
            var sum_1 : f32 = sum_0 + (textureSampleCompareLevel((shadowMap_0), (shadowSampler_0), ((_S40)).xy, i32(((_S40)).z), (proj_0.z - _S38)));
            x_0 = x_0 + i32(1);
            sum_0 = sum_1;
        }
        y_0 = y_0 + i32(1);
    }
    return sum_0 / 9.0f;
}

fn punctual_lighting_0( worldPos_2 : vec3<f32>,  n_1 : vec3<f32>,  v_1 : vec3<f32>,  albedo_2 : vec3<f32>,  metallic_1 : f32,  roughness_3 : f32,  f0_2 : vec3<f32>,  fragCoord_0 : vec4<f32>) -> vec3<f32>
{
    var _S41 : vec3<f32> = vec3<f32>(0.0f);
    var tileX_0 : u32 = u32(fragCoord_0.x) / u32(16);
    var tileY_0 : u32 = u32(fragCoord_0.y) / u32(16);
    var tileW_0 : u32 = u32(frame_0.cascade_splits_0.z);
    var tileH_0 : u32 = u32(frame_0.cascade_splits_0.w);
    var totalLights_0 : i32 = i32(frame_0.camera_position_0.w);
    var tileCount_0 : i32;
    var count_0 : i32;
    var tileOffset_0 : u32;
    if(tileW_0 != u32(0))
    {
        var gridEntry_0 : vec2<u32> = tileLightGrid_0[min(tileY_0, max(tileH_0, u32(1)) - u32(1)) * tileW_0 + min(tileX_0, max(tileW_0, u32(1)) - u32(1))];
        var count_1 : i32 = i32(gridEntry_0.x);
        var _S42 : u32 = gridEntry_0.y;
        if(count_1 > i32(0))
        {
            tileCount_0 = count_1;
        }
        else
        {
            tileCount_0 = totalLights_0;
        }
        count_0 = count_1;
        tileOffset_0 = _S42;
    }
    else
    {
        tileCount_0 = totalLights_0;
        count_0 = i32(0);
        tileOffset_0 = u32(0);
    }
    var j_0 : u32 = u32(0);
    var total_1 : vec3<f32> = _S41;
    for(;;)
    {
        if(j_0 < u32(tileCount_0))
        {
        }
        else
        {
            break;
        }
        var lightIdx_0 : u32;
        if(count_0 > i32(0))
        {
            lightIdx_0 = tileLightIndices_0[tileOffset_0 + j_0];
        }
        else
        {
            lightIdx_0 = j_0;
        }
        var _S43 : u32 = lightIdx_0 * u32(4);
        var a_1 : vec4<f32> = punctualLights_0[_S43];
        var b_0 : vec4<f32> = punctualLights_0[_S43 + u32(1)];
        var cvec_0 : vec4<f32> = punctualLights_0[_S43 + u32(2)];
        var dvec_0 : vec4<f32> = punctualLights_0[_S43 + u32(3)];
        var kind_0 : f32 = a_1.w;
        var l_1 : vec3<f32>;
        var attenuation_0 : f32;
        if(kind_0 > 2.5f)
        {
            l_1 = normalize((vec3<f32>(0) - cvec_0.xyz));
            attenuation_0 = 1.0f;
        }
        else
        {
            var toLight_0 : vec3<f32> = a_1.xyz - worldPos_2;
            var _S44 : f32 = max(length(toLight_0), 0.00009999999747379f);
            var l_2 : vec3<f32> = toLight_0 / vec3<f32>(_S44);
            var attenuation_1 : f32 = 1.0f / (_S44 * _S44);
            var range_0 : f32 = b_0.w;
            if(range_0 > 0.0f)
            {
                var k_1 : f32 = clamp(1.0f - pow(_S44 / range_0, 4.0f), 0.0f, 1.0f);
                attenuation_0 = attenuation_1 * (k_1 * k_1);
            }
            else
            {
                attenuation_0 = attenuation_1;
            }
            var attenuation_2 : f32;
            if(kind_0 > 1.5f)
            {
                attenuation_2 = attenuation_0 * smoothstep(dvec_0.x, cvec_0.w, dot(normalize(cvec_0.xyz), (vec3<f32>(0) - l_2)));
            }
            else
            {
                attenuation_2 = attenuation_0;
            }
            l_1 = l_2;
            attenuation_0 = attenuation_2;
        }
        if(attenuation_0 <= 0.0f)
        {
            j_0 = j_0 + u32(1);
            continue;
        }
        total_1 = total_1 + brdf_direct_0(n_1, v_1, l_1, albedo_2, metallic_1, roughness_3, f0_2, b_0.xyz * vec3<f32>(attenuation_0));
        j_0 = j_0 + u32(1);
    }
    return total_1;
}

fn hemisphere_irradiance_0( n_2 : vec3<f32>) -> vec3<f32>
{
    var _S45 : f32 = n_2.y;
    return mix(vec3<f32>(0.18000000715255737f, 0.15999999642372131f, 0.15000000596046448f) * vec3<f32>(0.69999998807907104f), mix(vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 0.72000002861022949f), vec3<f32>(0.09000000357627869f, 0.15999999642372131f, 0.34999999403953552f), vec3<f32>(pow(clamp(_S45, 0.0f, 1.0f), 0.69999998807907104f))), vec3<f32>(clamp(_S45 * 0.5f + 0.5f, 0.0f, 1.0f)));
}

fn sky_radiance_0( dir_0 : vec3<f32>,  withSun_0 : bool) -> vec3<f32>
{
    var _S46 : f32 = dir_0.y;
    var color_4 : vec3<f32>;
    if(_S46 >= 0.0f)
    {
        color_4 = mix(vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 0.72000002861022949f), vec3<f32>(0.09000000357627869f, 0.15999999642372131f, 0.34999999403953552f), vec3<f32>(pow(clamp(_S46, 0.0f, 1.0f), 0.69999998807907104f)));
    }
    else
    {
        color_4 = mix(vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 0.72000002861022949f), vec3<f32>(0.18000000715255737f, 0.15999999642372131f, 0.15000000596046448f), vec3<f32>(clamp(- _S46 * 3.0f, 0.0f, 1.0f)));
    }
    if(withSun_0)
    {
        var _S47 : f32 = max(dot(dir_0, normalize(frame_0.light_dir_ambient_0.xyz)), 0.0f);
        color_4 = color_4 + vec3<f32>(1.0f, 0.94999998807907104f, 0.85000002384185791f) * vec3<f32>((pow(_S47, 1200.0f) * 24.0f + pow(_S47, 48.0f) * 0.5f)) * vec3<f32>((frame_0.light_color_intensity_0.w * 0.40000000596046448f));
    }
    return color_4;
}

fn env_brdf_approx_0( f0_3 : vec3<f32>,  roughness_4 : f32,  nDotV_0 : f32) -> vec3<f32>
{
    var r_1 : vec4<f32> = vec4<f32>(roughness_4) * vec4<f32>(-1.0f, -0.02749999985098839f, -0.57200002670288086f, 0.02199999988079071f) + vec4<f32>(1.0f, 0.04250000044703484f, 1.03999996185302734f, -0.03999999910593033f);
    var _S48 : f32 = r_1.x;
    var ab_0 : vec2<f32> = vec2<f32>(-1.03999996185302734f, 1.03999996185302734f) * vec2<f32>((min(_S48 * _S48, exp2(-9.27999973297119141f * nDotV_0)) * _S48 + r_1.y)) + r_1.zw;
    return f0_3 * vec3<f32>(ab_0.x) + vec3<f32>(ab_0.y);
}

struct pixelOutput_0
{
    @location(0) output_1 : vec4<f32>,
};

struct pixelInput_1
{
    @location(0) worldNormal_1 : vec3<f32>,
    @location(1) uv_9 : vec2<f32>,
    @location(2) lightSpacePos_2 : vec4<f32>,
    @location(3) worldTangent_1 : vec4<f32>,
    @location(4) worldPosition_1 : vec3<f32>,
    @location(5) viewDepth_2 : f32,
    @location(6) vertexColor_1 : vec4<f32>,
    @location(7) uv1_5 : vec2<f32>,
};

@fragment
fn fs_main( _S49 : pixelInput_1, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var uvMask_0 : u32 = u32(prim_0.material_flags_0.y);
    var baseIn_0 : vec2<f32>;
    if(((uvMask_0 & (u32(1)))) != u32(0))
    {
        baseIn_0 = _S49.uv1_5;
    }
    else
    {
        baseIn_0 = _S49.uv_9;
    }
    var mrIn_0 : vec2<f32>;
    if(((uvMask_0 & (u32(2)))) != u32(0))
    {
        mrIn_0 = _S49.uv1_5;
    }
    else
    {
        mrIn_0 = _S49.uv_9;
    }
    var normalIn_0 : vec2<f32>;
    if(((uvMask_0 & (u32(4)))) != u32(0))
    {
        normalIn_0 = _S49.uv1_5;
    }
    else
    {
        normalIn_0 = _S49.uv_9;
    }
    var emissiveIn_0 : vec2<f32>;
    if(((uvMask_0 & (u32(8)))) != u32(0))
    {
        emissiveIn_0 = _S49.uv1_5;
    }
    else
    {
        emissiveIn_0 = _S49.uv_9;
    }
    var occlusionIn_0 : vec2<f32>;
    if(((uvMask_0 & (u32(16)))) != u32(0))
    {
        occlusionIn_0 = _S49.uv1_5;
    }
    else
    {
        occlusionIn_0 = _S49.uv_9;
    }
    var _S50 : f32 = baseIn_0.x;
    var _S51 : f32 = baseIn_0.y;
    var mrSample_0 : vec4<f32> = (textureSample((metalRoughTex_0), (metalRoughSampler_0), (mrIn_0)));
    var normalSample_0 : vec4<f32> = (textureSample((normalTex_0), (normalSampler_0), (normalIn_0)));
    var emissiveSample_0 : vec4<f32> = (textureSample((emissiveTex_0), (emissiveSampler_0), (emissiveIn_0)));
    var occlusionSample_0 : vec4<f32> = (textureSample((occlusionTex_0), (occlusionSampler_0), (occlusionIn_0)));
    var albedo_3 : vec4<f32> = prim_0.base_color_0 * (textureSample((baseColorTex_0), (baseColorSampler_0), (vec2<f32>(prim_0.base_uv_row0_0.x * _S50 + prim_0.base_uv_row0_0.y * _S51 + prim_0.base_uv_row0_0.z, prim_0.base_uv_row1_0.x * _S50 + prim_0.base_uv_row1_0.y * _S51 + prim_0.base_uv_row1_0.z)))) * _S49.vertexColor_1;
    var _S52 : f32 = albedo_3.w;
    if(_S52 < (prim_0.emissive_factor_0.w))
    {
        discard;
    }
    if((prim_0.material_flags_0.x) > 0.5f)
    {
        var _S53 : pixelOutput_0 = pixelOutput_0( albedo_3 );
        return _S53;
    }
    var metallic_2 : f32 = clamp(prim_0.material_factors_0.x * mrSample_0.z, 0.0f, 1.0f);
    var roughness_5 : f32 = clamp(prim_0.material_factors_0.y * mrSample_0.y, 0.04500000178813934f, 1.0f);
    var occlusion_0 : f32 = mix(1.0f, occlusionSample_0.x, prim_0.material_factors_0.z);
    var emissive_0 : vec3<f32> = prim_0.emissive_factor_0.xyz * emissiveSample_0.xyz;
    var nGeom_0 : vec3<f32> = normalize(_S49.worldNormal_1);
    var _S54 : vec3<f32> = _S49.worldTangent_1.xyz;
    var t_0 : vec3<f32> = normalize(_S54 - nGeom_0 * vec3<f32>(dot(nGeom_0, _S54)));
    var nTs_0 : vec3<f32> = normalSample_0.xyz * vec3<f32>(2.0f) - vec3<f32>(1.0f);
    var n_3 : vec3<f32> = normalize((((vec3<f32>(nTs_0.xy * vec2<f32>(prim_0.material_factors_0.w), nTs_0.z)) * (mat3x3<f32>(t_0, cross(nGeom_0, t_0) * vec3<f32>(_S49.worldTangent_1.w), nGeom_0)))));
    var l_3 : vec3<f32> = normalize(frame_0.light_dir_ambient_0.xyz);
    var v_2 : vec3<f32> = normalize(frame_0.camera_position_0.xyz - _S49.worldPosition_1);
    var _S55 : vec3<f32> = albedo_3.xyz;
    var f0_4 : vec3<f32> = mix(vec3<f32>(0.03999999910593033f), _S55, vec3<f32>(metallic_2));
    var directLight_0 : vec3<f32> = brdf_direct_0(n_3, v_2, l_3, _S55, metallic_2, roughness_5, f0_4, frame_0.light_color_intensity_0.xyz * vec3<f32>(frame_0.light_color_intensity_0.w)) * vec3<f32>(shadow_visibility_0(_S49.viewDepth_2, _S49.worldPosition_1, max(dot(n_3, l_3), 0.0f)));
    var punctual_0 : vec3<f32> = punctual_lighting_0(_S49.worldPosition_1, n_3, v_2, _S55, metallic_2, roughness_5, f0_4, svPosition_1);
    var reflected_0 : vec3<f32> = reflect((vec3<f32>(0) - v_2), n_3);
    var _S56 : f32 = dot(n_3, v_2);
    var _S57 : f32 = max(_S56, 0.00009999999747379f);
    var kSIbl_0 : vec3<f32> = fresnel_schlick_0(_S57, f0_4);
    var iblStrength_0 : f32 = frame_0.light_dir_ambient_0.w;
    var envEnabled_0 : bool = (iblParams_0.enabled_maxmip_intensity_0.x) > 0.5f;
    var irradianceAnalytic_0 : vec3<f32> = hemisphere_irradiance_0(n_3);
    var _S58 : vec3<f32> = vec3<f32>(iblParams_0.enabled_maxmip_intensity_0.z);
    var irradianceEnv_0 : vec3<f32> = (textureSampleLevel((irradianceMap_0), (iblSampler_0), (n_3), (0.0f))).xyz * _S58;
    var irradiance_0 : vec3<f32>;
    if(envEnabled_0)
    {
        irradiance_0 = irradianceEnv_0;
    }
    else
    {
        irradiance_0 = irradianceAnalytic_0;
    }
    var diffuseIbl_0 : vec3<f32> = (vec3<f32>(1.0f) - kSIbl_0) * vec3<f32>((1.0f - metallic_2)) * _S55 * irradiance_0;
    var brdfLutSample_0 : vec2<f32> = (textureSample((brdfLut_0), (iblSampler_0), (vec2<f32>(max(_S56, 0.0f), roughness_5)))).xy;
    var specularSplitSum_0 : vec3<f32> = (textureSampleLevel((prefilteredMap_0), (iblSampler_0), (reflected_0), (roughness_5 * iblParams_0.enabled_maxmip_intensity_0.y))).xyz * _S58 * (f0_4 * vec3<f32>(brdfLutSample_0.x) + vec3<f32>(brdfLutSample_0.y));
    var specularAnalytic_0 : vec3<f32> = mix(sky_radiance_0(reflected_0, true), irradianceAnalytic_0, vec3<f32>((roughness_5 * roughness_5))) * env_brdf_approx_0(f0_4, roughness_5, _S57);
    var specularIbl_0 : vec3<f32>;
    if(envEnabled_0)
    {
        specularIbl_0 = specularSplitSum_0;
    }
    else
    {
        specularIbl_0 = specularAnalytic_0;
    }
    var _S59 : pixelOutput_0 = pixelOutput_0( vec4<f32>(directLight_0 + punctual_0 + vec3<f32>((iblStrength_0 * occlusion_0)) * (diffuseIbl_0 + specularIbl_0) + emissive_0, _S52) );
    return _S59;
}

