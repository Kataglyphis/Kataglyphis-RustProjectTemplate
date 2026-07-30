struct IblParams_std140_0
{
    @align(16) face_roughness_samples_mip_0 : vec4<f32>,
    @align(16) source_resolution_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> params_0 : IblParams_std140_0;
@binding(1) @group(0) var srcEquirect_0 : texture_2d<f32>;

@binding(3) @group(0) var srcSampler_0 : sampler;

@binding(2) @group(0) var srcCube_0 : texture_cube<f32>;

struct FullscreenVsOut_0
{
    @builtin(position) svPosition_0 : vec4<f32>,
    @location(0) uv_0 : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid_0 : u32) -> FullscreenVsOut_0
{
    var x_0 : f32 = f32((((vid_0 << (u32(1)))) & (u32(2))));
    var y_0 : f32 = f32((vid_0 & (u32(2))));
    var o_0 : FullscreenVsOut_0;
    o_0.svPosition_0 = vec4<f32>(x_0 * 2.0f - 1.0f, 1.0f - y_0 * 2.0f, 0.0f, 1.0f);
    o_0.uv_0 = vec2<f32>(x_0, y_0);
    return o_0;
}

fn cube_direction_0( face_0 : u32,  uv_1 : vec2<f32>) -> vec3<f32>
{
    var a_0 : f32 = 2.0f * uv_1.x - 1.0f;
    var b_0 : f32 = 1.0f - 2.0f * uv_1.y;
    var dir_0 : vec3<f32>;
    switch(face_0)
    {
    case u32(0):
        {
            dir_0 = vec3<f32>(1.0f, b_0, - a_0);
            break;
        }
    case u32(1):
        {
            dir_0 = vec3<f32>(-1.0f, b_0, a_0);
            break;
        }
    case u32(2):
        {
            dir_0 = vec3<f32>(a_0, 1.0f, - b_0);
            break;
        }
    case u32(3):
        {
            dir_0 = vec3<f32>(a_0, -1.0f, b_0);
            break;
        }
    case u32(4):
        {
            dir_0 = vec3<f32>(a_0, b_0, 1.0f);
            break;
        }
    default :
        {
            dir_0 = vec3<f32>(- a_0, b_0, -1.0f);
            break;
        }
    }
    return normalize(dir_0);
}

fn equirect_uv_0( dir_1 : vec3<f32>) -> vec2<f32>
{
    return vec2<f32>(atan2(dir_1.z, dir_1.x) / 6.28318548202514648f + 0.5f, 0.5f - asin(clamp(dir_1.y, -1.0f, 1.0f)) / 3.14159274101257324f);
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
fn fs_equirect_to_cube( _S1 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var _S2 : pixelOutput_0 = pixelOutput_0( (textureSample((srcEquirect_0), (srcSampler_0), (equirect_uv_0(cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S1.uv_2))))) );
    return _S2;
}

struct pixelOutput_1
{
    @location(0) output_1 : vec4<f32>,
};

struct pixelInput_1
{
    @location(0) uv_3 : vec2<f32>,
};

@fragment
fn fs_irradiance( _S3 : pixelInput_1, @builtin(position) svPosition_2 : vec4<f32>) -> pixelOutput_1
{
    var dir_2 : vec3<f32> = cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S3.uv_3);
    var up_0 : vec3<f32>;
    if((abs(dir_2.y)) < 0.99900001287460327f)
    {
        up_0 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        up_0 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    var right_0 : vec3<f32> = normalize(cross(up_0, dir_2));
    var _S4 : vec3<f32> = cross(dir_2, right_0);
    var _S5 : vec3<f32> = vec3<f32>(0.0f);
    var phi_i_0 : u32 = u32(0);
    var irradiance_0 : vec3<f32> = _S5;
    var sampleCount_0 : f32 = 0.0f;
    for(;;)
    {
        if(phi_i_0 < u32(64))
        {
        }
        else
        {
            break;
        }
        var theta_i_0 : u32 = u32(0);
        for(;;)
        {
            if(theta_i_0 < u32(32))
            {
            }
            else
            {
                break;
            }
            var phi_0 : f32 = f32(phi_i_0) * 0.09817477315664291f;
            var theta_0 : f32 = f32(theta_i_0) * 0.04908738657832146f;
            var _S6 : f32 = sin(theta_0);
            var _S7 : vec3<f32> = vec3<f32>(cos(theta_0));
            var irradiance_1 : vec3<f32> = irradiance_0 + (textureSample((srcCube_0), (srcSampler_0), (vec3<f32>((_S6 * cos(phi_0))) * right_0 + vec3<f32>((_S6 * sin(phi_0))) * _S4 + _S7 * dir_2))).xyz * _S7 * vec3<f32>(_S6);
            var sampleCount_1 : f32 = sampleCount_0 + 1.0f;
            theta_i_0 = theta_i_0 + u32(1);
            irradiance_0 = irradiance_1;
            sampleCount_0 = sampleCount_1;
        }
        phi_i_0 = phi_i_0 + u32(1);
    }
    var _S8 : pixelOutput_1 = pixelOutput_1( vec4<f32>(vec3<f32>(3.14159274101257324f) * irradiance_0 / vec3<f32>(sampleCount_0), 1.0f) );
    return _S8;
}

struct pixelOutput_2
{
    @location(0) output_2 : vec4<f32>,
};

struct pixelInput_2
{
    @location(0) uv_4 : vec2<f32>,
};

@fragment
fn fs_prefilter( _S9 : pixelInput_2, @builtin(position) svPosition_3 : vec4<f32>) -> pixelOutput_2
{
    var dir_3 : vec3<f32> = cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S9.uv_4);
    var _S10 : u32 = u32(params_0.face_roughness_samples_mip_0.z);
    var _S11 : vec3<f32> = vec3<f32>(0.0f);
    var i_0 : u32 = u32(0);
    var prefiltered_0 : vec3<f32> = _S11;
    var totalWeight_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_0 < _S10)
        {
        }
        else
        {
            break;
        }
        var _S12 : f32 = f32(i_0) / f32(_S10);
        var h_0 : vec3<f32> = vec3<f32>(_S12 * cos(0.0f), _S12 * sin(0.0f), sqrt(1.0f - _S12 * _S12));
        var sampleDir_0 : vec3<f32> = normalize(vec3<f32>((2.0f * dot(dir_3, h_0))) * h_0 - dir_3);
        var _S13 : f32 = max(dot(dir_3, sampleDir_0), 0.0f);
        if(_S13 > 0.0f)
        {
            var totalWeight_1 : f32 = totalWeight_0 + _S13;
            prefiltered_0 = prefiltered_0 + (textureSample((srcCube_0), (srcSampler_0), (sampleDir_0))).xyz * vec3<f32>(_S13);
            totalWeight_0 = totalWeight_1;
        }
        i_0 = i_0 + u32(1);
    }
    var _S14 : pixelOutput_2 = pixelOutput_2( vec4<f32>(prefiltered_0 / vec3<f32>(max(totalWeight_0, 0.00009999999747379f)), 1.0f) );
    return _S14;
}

fn geometry_smith_ibl_0( nDotV_0 : f32,  nDotL_0 : f32,  roughness_0 : f32) -> f32
{
    var k_0 : f32 = roughness_0 * roughness_0 / 2.0f;
    var _S15 : f32 = 1.0f - k_0;
    return nDotV_0 / max(nDotV_0 * _S15 + k_0, 0.00009999999747379f) * (nDotL_0 / max(nDotL_0 * _S15 + k_0, 0.00009999999747379f));
}

struct pixelOutput_3
{
    @location(0) output_3 : vec4<f32>,
};

struct pixelInput_3
{
    @location(0) uv_5 : vec2<f32>,
};

@fragment
fn fs_brdf_lut( _S16 : pixelInput_3, @builtin(position) svPosition_4 : vec4<f32>) -> pixelOutput_3
{
    var nDotV_1 : f32 = _S16.uv_5.x;
    var _S17 : f32 = _S16.uv_5.y;
    var _S18 : vec3<f32> = vec3<f32>(sqrt(1.0f - nDotV_1 * nDotV_1), 0.0f, nDotV_1);
    var _S19 : vec2<f32> = vec2<f32>(0.0f);
    var i_1 : u32 = u32(0);
    var result_0 : vec2<f32> = _S19;
    for(;;)
    {
        if(i_1 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var _S20 : f32 = f32(i_1) / 1024.0f;
        var cosTheta_0 : f32 = sqrt(1.0f - _S20 * _S20);
        var h_1 : vec3<f32> = vec3<f32>(_S20 * cos(0.0f), _S20 * sin(0.0f), cosTheta_0);
        var _S21 : f32 = dot(_S18, h_1);
        var _S22 : f32 = max(normalize(vec3<f32>((2.0f * _S21)) * h_1 - _S18).z, 0.0f);
        var _S23 : f32 = max(cosTheta_0, 0.0f);
        var _S24 : f32 = max(_S21, 0.0f);
        if(_S22 > 0.0f)
        {
            var _S25 : f32 = geometry_smith_ibl_0(nDotV_1, _S22, _S17) * _S24;
            result_0 = result_0 + vec2<f32>(_S25 / max(_S23 * _S22, 0.00009999999747379f), _S25 / max(_S23 * nDotV_1, 0.00009999999747379f));
        }
        i_1 = i_1 + u32(1);
    }
    var _S26 : pixelOutput_3 = pixelOutput_3( vec4<f32>(result_0 / vec2<f32>(1024.0f), 0.0f, 1.0f) );
    return _S26;
}

