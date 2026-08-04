struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct SkyUniforms_std140_0
{
    @align(16) inv_view_proj_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) light_dir_intensity_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> sky_0 : SkyUniforms_std140_0;
struct SkyVsOut_0
{
    @builtin(position) svPosition_0 : vec4<f32>,
    @location(0) ndc_0 : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid_0 : u32) -> SkyVsOut_0
{
    var x_0 : f32 = f32(vid_0 / u32(2)) * 4.0f - 1.0f;
    var y_0 : f32 = f32(vid_0 % u32(2)) * 4.0f - 1.0f;
    var o_0 : SkyVsOut_0;
    o_0.svPosition_0 = vec4<f32>(x_0, y_0, 1.0f, 1.0f);
    o_0.ndc_0 = vec2<f32>(x_0, y_0);
    return o_0;
}

fn sky_gradient_0( dir_0 : vec3<f32>) -> vec3<f32>
{
    var _S1 : f32 = dir_0.y;
    if(_S1 >= 0.0f)
    {
        return mix(vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 0.72000002861022949f), vec3<f32>(0.09000000357627869f, 0.15999999642372131f, 0.34999999403953552f), vec3<f32>(pow(clamp(_S1, 0.0f, 1.0f), 0.69999998807907104f)));
    }
    else
    {
        return mix(vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 0.72000002861022949f), vec3<f32>(0.18000000715255737f, 0.15999999642372131f, 0.15000000596046448f), vec3<f32>(clamp(- _S1 * 3.0f, 0.0f, 1.0f)));
    }
}

fn sun_disk_0( dir_1 : vec3<f32>,  lightDir_0 : vec3<f32>,  intensity_0 : f32) -> vec3<f32>
{
    var _S2 : f32 = max(dot(dir_1, normalize(lightDir_0)), 0.0f);
    return vec3<f32>(1.0f, 0.94999998807907104f, 0.85000002384185791f) * vec3<f32>((pow(_S2, 1200.0f) * 24.0f + pow(_S2, 48.0f) * 0.5f)) * vec3<f32>((intensity_0 * 0.40000000596046448f));
}

struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) ndc_1 : vec2<f32>,
};

@fragment
fn fs_main( _S3 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var nearP_0 : vec4<f32> = (((vec4<f32>(_S3.ndc_1, 0.0f, 1.0f)) * (mat4x4<f32>(sky_0.inv_view_proj_0.data_0[i32(0)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(3)]))));
    var farP_0 : vec4<f32> = (((vec4<f32>(_S3.ndc_1, 1.0f, 1.0f)) * (mat4x4<f32>(sky_0.inv_view_proj_0.data_0[i32(0)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(0)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(1)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(2)], sky_0.inv_view_proj_0.data_0[i32(0)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(1)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(2)][i32(3)], sky_0.inv_view_proj_0.data_0[i32(3)][i32(3)]))));
    var dir_2 : vec3<f32> = normalize(farP_0.xyz / vec3<f32>(farP_0.w) - nearP_0.xyz / vec3<f32>(nearP_0.w));
    var _S4 : pixelOutput_0 = pixelOutput_0( vec4<f32>(sky_gradient_0(dir_2) + sun_disk_0(dir_2, sky_0.light_dir_intensity_0.xyz, sky_0.light_dir_intensity_0.w), 1.0f) );
    return _S4;
}

