struct IblParams_std140_0
{
    @align(16) face_roughness_samples_mip_0 : vec4<f32>,
    @align(16) source_resolution_0 : vec4<f32>,
};

@binding(0) @group(0) var<uniform> params_0 : IblParams_std140_0;
@binding(1) @group(0) var srcEquirect_0 : texture_2d<f32>;

@binding(2) @group(0) var srcCube_0 : texture_cube<f32>;

@binding(3) @group(0) var srcSampler_0 : sampler;

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

fn sample_equirect_0( dir_2 : vec3<f32>) -> vec3<f32>
{
    var dimsX_0 : u32;
    var dimsY_0 : u32;
    {var dim = textureDimensions((srcEquirect_0));((dimsX_0)) = dim.x;((dimsY_0)) = dim.y;};
    var _S1 : i32 = i32(dimsX_0);
    var _S2 : i32 = i32(dimsY_0);
    var texel_0 : vec2<f32> = equirect_uv_0(dir_2) * vec2<f32>(vec2<i32>(_S1, _S2)) - vec2<f32>(0.5f);
    var base_0 : vec2<f32> = floor(texel_0);
    var frac_0 : vec2<f32> = texel_0 - base_0;
    var x0_0 : i32 = i32(base_0.x);
    var y0_0 : i32 = i32(base_0.y);
    var _S3 : i32 = x0_0 % _S1;
    var xa_0 : i32 = (_S3 + _S1) % _S1;
    var _S4 : i32 = (x0_0 + i32(1)) % _S1;
    var xb_0 : i32 = (_S4 + _S1) % _S1;
    var _S5 : i32 = _S2 - i32(1);
    var ya_0 : i32 = clamp(y0_0, i32(0), _S5);
    var yb_0 : i32 = clamp(y0_0 + i32(1), i32(0), _S5);
    var _S6 : vec3<i32> = vec3<i32>(xa_0, ya_0, i32(0));
    var _S7 : vec3<i32> = vec3<i32>(xb_0, ya_0, i32(0));
    var _S8 : vec3<i32> = vec3<i32>(xa_0, yb_0, i32(0));
    var _S9 : vec3<i32> = vec3<i32>(xb_0, yb_0, i32(0));
    var _S10 : vec3<f32> = vec3<f32>(frac_0.x);
    return mix(mix((textureLoad((srcEquirect_0), ((_S6)).xy, ((_S6)).z)).xyz, (textureLoad((srcEquirect_0), ((_S7)).xy, ((_S7)).z)).xyz, _S10), mix((textureLoad((srcEquirect_0), ((_S8)).xy, ((_S8)).z)).xyz, (textureLoad((srcEquirect_0), ((_S9)).xy, ((_S9)).z)).xyz, _S10), vec3<f32>(frac_0.y));
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
fn fs_equirect_to_cube( _S11 : pixelInput_0, @builtin(position) svPosition_1 : vec4<f32>) -> pixelOutput_0
{
    var _S12 : vec3<f32> = sample_equirect_0(cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S11.uv_2));
    var _S13 : pixelOutput_0 = pixelOutput_0( vec4<f32>(_S12, 1.0f) );
    return _S13;
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
fn fs_downsample_cube( _S14 : pixelInput_1, @builtin(position) svPosition_2 : vec4<f32>) -> pixelOutput_1
{
    var _S15 : pixelOutput_1 = pixelOutput_1( (textureSampleLevel((srcCube_0), (srcSampler_0), (cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S14.uv_3)), (params_0.face_roughness_samples_mip_0.w))) );
    return _S15;
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
fn fs_irradiance( _S16 : pixelInput_2, @builtin(position) svPosition_3 : vec4<f32>) -> pixelOutput_2
{
    var dir_3 : vec3<f32> = cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S16.uv_4);
    var up_0 : vec3<f32>;
    if((abs(dir_3.y)) < 0.99900001287460327f)
    {
        up_0 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    else
    {
        up_0 = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    var right_0 : vec3<f32> = normalize(cross(up_0, dir_3));
    var _S17 : vec3<f32> = cross(dir_3, right_0);
    var _S18 : vec3<f32> = vec3<f32>(0.0f);
    var phi_i_0 : u32 = u32(0);
    var irradiance_0 : vec3<f32> = _S18;
    for(;;)
    {
        if(phi_i_0 < u32(128))
        {
        }
        else
        {
            break;
        }
        var _S19 : f32 = 6.28318548202514648f * (f32(phi_i_0) + 0.5f) / 128.0f;
        var theta_i_0 : u32 = u32(0);
        for(;;)
        {
            if(theta_i_0 < u32(64))
            {
            }
            else
            {
                break;
            }
            var theta_0 : f32 = 1.57079637050628662f * (f32(theta_i_0) + 0.5f) / 64.0f;
            var _S20 : f32 = sin(theta_0);
            var _S21 : vec3<f32> = vec3<f32>(cos(theta_0));
            var irradiance_1 : vec3<f32> = irradiance_0 + (textureSampleLevel((srcCube_0), (srcSampler_0), (vec3<f32>((_S20 * cos(_S19))) * right_0 + vec3<f32>((_S20 * sin(_S19))) * _S17 + _S21 * dir_3), (0.0f))).xyz * _S21 * vec3<f32>(_S20);
            theta_i_0 = theta_i_0 + u32(1);
            irradiance_0 = irradiance_1;
        }
        phi_i_0 = phi_i_0 + u32(1);
    }
    var _S22 : pixelOutput_2 = pixelOutput_2( vec4<f32>(vec3<f32>(3.14159274101257324f) * irradiance_0 / vec3<f32>(8192.0f), 1.0f) );
    return _S22;
}

fn radical_inverse_vdc_0( bits_in_0 : u32) -> f32
{
    var bits_0 : u32 = (((bits_in_0 << (u32(16)))) | (((bits_in_0 >> (u32(16))))));
    var bits_1 : u32 = (((((bits_0 & (u32(1431655765)))) << (u32(1)))) | (((((bits_0 & (u32(2863311530)))) >> (u32(1))))));
    var bits_2 : u32 = (((((bits_1 & (u32(858993459)))) << (u32(2)))) | (((((bits_1 & (u32(3435973836)))) >> (u32(2))))));
    var bits_3 : u32 = (((((bits_2 & (u32(252645135)))) << (u32(4)))) | (((((bits_2 & (u32(4042322160)))) >> (u32(4))))));
    return f32((((((bits_3 & (u32(16711935)))) << (u32(8)))) | (((((bits_3 & (u32(4278255360)))) >> (u32(8))))))) * 2.32830643653869629e-10f;
}

fn hammersley_0( i_0 : u32,  count_0 : u32) -> vec2<f32>
{
    return vec2<f32>(f32(i_0) / f32(count_0), radical_inverse_vdc_0(i_0));
}

fn importance_sample_ggx_0( xi_0 : vec2<f32>,  normal_0 : vec3<f32>,  roughness_0 : f32) -> vec3<f32>
{
    var a_1 : f32 = roughness_0 * roughness_0;
    var phi_0 : f32 = 6.28318548202514648f * xi_0.x;
    var _S23 : f32 = xi_0.y;
    var cosTheta_0 : f32 = sqrt(clamp((1.0f - _S23) / (1.0f + (a_1 * a_1 - 1.0f) * _S23), 0.0f, 1.0f));
    var sinTheta_0 : f32 = sqrt(max(1.0f - cosTheta_0 * cosTheta_0, 0.0f));
    var _S24 : f32 = sinTheta_0 * cos(phi_0);
    var _S25 : f32 = sinTheta_0 * sin(phi_0);
    var up_1 : vec3<f32>;
    if((abs(normal_0.z)) < 0.99900001287460327f)
    {
        up_1 = vec3<f32>(0.0f, 0.0f, 1.0f);
    }
    else
    {
        up_1 = vec3<f32>(0.0f, 1.0f, 0.0f);
    }
    var tangent_0 : vec3<f32> = normalize(cross(up_1, normal_0));
    return normalize(vec3<f32>(_S24) * tangent_0 + vec3<f32>(_S25) * cross(normal_0, tangent_0) + vec3<f32>(cosTheta_0) * normal_0);
}

fn distribution_ggx_0( nDotH_0 : f32,  roughness_1 : f32) -> f32
{
    var a_2 : f32 = roughness_1 * roughness_1;
    var a2_0 : f32 = a_2 * a_2;
    var denom_0 : f32 = nDotH_0 * nDotH_0 * (a2_0 - 1.0f) + 1.0f;
    return a2_0 / max(3.14159274101257324f * denom_0 * denom_0, 9.99999997475242708e-07f);
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
fn fs_prefilter( _S26 : pixelInput_3, @builtin(position) svPosition_4 : vec4<f32>) -> pixelOutput_3
{
    var normal_1 : vec3<f32> = cube_direction_0(u32(params_0.face_roughness_samples_mip_0.x), _S26.uv_5);
    var roughness_2 : f32 = params_0.face_roughness_samples_mip_0.y;
    var _S27 : u32 = u32(params_0.face_roughness_samples_mip_0.z);
    var _S28 : f32 = params_0.source_resolution_0.x;
    if(roughness_2 <= 0.0f)
    {
        var _S29 : pixelOutput_3 = pixelOutput_3( (textureSampleLevel((srcCube_0), (srcSampler_0), (normal_1), (0.0f))) );
        return _S29;
    }
    var _S30 : vec3<f32> = vec3<f32>(0.0f);
    var i_1 : u32 = u32(0);
    var color_0 : vec3<f32> = _S30;
    var totalWeight_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_1 < _S27)
        {
        }
        else
        {
            break;
        }
        var halfVector_0 : vec3<f32> = importance_sample_ggx_0(hammersley_0(i_1, _S27), normal_1, roughness_2);
        var _S31 : f32 = dot(normal_1, halfVector_0);
        var sampleDir_0 : vec3<f32> = normalize(vec3<f32>((2.0f * _S31)) * halfVector_0 - normal_1);
        var nDotL_0 : f32 = dot(normal_1, sampleDir_0);
        if(nDotL_0 > 0.0f)
        {
            var totalWeight_1 : f32 = totalWeight_0 + nDotL_0;
            color_0 = color_0 + (textureSampleLevel((srcCube_0), (srcSampler_0), (sampleDir_0), (max(0.5f * log2(1.0f / (f32(_S27) * (distribution_ggx_0(max(_S31, 0.0f), roughness_2) * 0.25f + 0.00009999999747379f)) / (12.56637096405029297f / (6.0f * _S28 * _S28))), 0.0f)))).xyz * vec3<f32>(nDotL_0);
            totalWeight_0 = totalWeight_1;
        }
        i_1 = i_1 + u32(1);
    }
    var _S32 : pixelOutput_3 = pixelOutput_3( vec4<f32>(color_0 / vec3<f32>(max(totalWeight_0, 0.00009999999747379f)), 1.0f) );
    return _S32;
}

fn geometry_smith_ibl_0( nDotV_0 : f32,  nDotL_1 : f32,  roughness_3 : f32) -> f32
{
    var k_0 : f32 = roughness_3 * roughness_3 / 2.0f;
    var _S33 : f32 = 1.0f - k_0;
    return nDotV_0 / max(nDotV_0 * _S33 + k_0, 0.00009999999747379f) * (nDotL_1 / max(nDotL_1 * _S33 + k_0, 0.00009999999747379f));
}

struct pixelOutput_4
{
    @location(0) output_4 : vec4<f32>,
};

struct pixelInput_4
{
    @location(0) uv_6 : vec2<f32>,
};

@fragment
fn fs_brdf_lut( _S34 : pixelInput_4, @builtin(position) svPosition_5 : vec4<f32>) -> pixelOutput_4
{
    var nDotV_1 : f32 = _S34.uv_6.x;
    var _S35 : f32 = _S34.uv_6.y;
    var _S36 : vec3<f32> = vec3<f32>(sqrt(1.0f - nDotV_1 * nDotV_1), 0.0f, nDotV_1);
    const _S37 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 1.0f);
    var _S38 : vec2<f32> = vec2<f32>(0.0f);
    var i_2 : u32 = u32(0);
    var result_0 : vec2<f32> = _S38;
    for(;;)
    {
        if(i_2 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var halfVector_1 : vec3<f32> = importance_sample_ggx_0(hammersley_0(i_2, u32(1024)), _S37, _S35);
        var _S39 : f32 = dot(_S36, halfVector_1);
        var _S40 : f32 = max(normalize(vec3<f32>((2.0f * _S39)) * halfVector_1 - _S36).z, 0.0f);
        if(_S40 > 0.0f)
        {
            var _S41 : f32 = max(_S39, 0.0f);
            var _S42 : f32 = max(nDotV_1, 0.00009999999747379f);
            var gVis_0 : f32 = geometry_smith_ibl_0(_S42, _S40, _S35) * _S41 / max(max(halfVector_1.z, 0.0f) * _S42, 9.99999997475242708e-07f);
            var fc_0 : f32 = pow(1.0f - _S41, 5.0f);
            result_0 = result_0 + vec2<f32>((1.0f - fc_0) * gVis_0, fc_0 * gVis_0);
        }
        i_2 = i_2 + u32(1);
    }
    var _S43 : pixelOutput_4 = pixelOutput_4( vec4<f32>(result_0 / vec2<f32>(1024.0f), 0.0f, 1.0f) );
    return _S43;
}

