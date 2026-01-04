#version 460
#if defined(GL_EXT_control_flow_attributes)
#extension GL_EXT_control_flow_attributes : require
#define SPIRV_CROSS_FLATTEN [[flatten]]
#define SPIRV_CROSS_BRANCH [[dont_flatten]]
#define SPIRV_CROSS_UNROLL [[unroll]]
#define SPIRV_CROSS_LOOP [[dont_unroll]]
#else
#define SPIRV_CROSS_FLATTEN
#define SPIRV_CROSS_BRANCH
#define SPIRV_CROSS_UNROLL
#define SPIRV_CROSS_LOOP
#endif
#extension GL_EXT_spirv_intrinsics : require

float _162;

layout(set = 1, binding = 1, std140) uniform type_View
{
    layout(offset = 448) mat4 View_ViewToClip;
    layout(offset = 832) mat4 View_ScreenToTranslatedWorld;
    layout(offset = 1168) vec3 View_ViewForward;
    layout(offset = 1248) vec4 View_InvDeviceZToWorldZTransform;
    layout(offset = 1296) vec3 View_TranslatedWorldCameraOrigin;
    layout(offset = 1344) vec3 View_PreViewTranslationHigh;
    layout(offset = 1360) vec3 View_PreViewTranslationLow;
    layout(offset = 2368) vec4 View_ViewRectMin;
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;
    layout(offset = 2504) float View_PreExposure;
    layout(offset = 2512) vec4 View_DiffuseOverrideParameter;
    layout(offset = 2528) vec4 View_SpecularOverrideParameter;
    layout(offset = 2784) vec4 View_TemporalAAParams;
    layout(offset = 2912) vec3 View_PrecomputedIndirectSpecularColorScale;
    layout(offset = 3344) vec4 View_SkyLightColor;
    layout(offset = 3520) uint View_DistanceFieldAOSpecularOcclusionMode;
    layout(offset = 4032) float View_bCheckerboardSubsurfaceProfileRendering;
    layout(offset = 4364) float View_bSubsurfacePostprocessEnabled;
    layout(offset = 5696) uint View_bShadingEnergyConservation;
} View;

layout(set = 1, binding = 5, std430) readonly buffer type_StructuredBuffer_v4float
{
    vec4 _m0[];
} View_SkyIrradianceEnvironmentMap;

layout(set = 1, binding = 2, std140) uniform type_ReflectionCaptureSM5
{
    vec4 ReflectionCaptureSM5_PositionHighAndRadius[341];
    vec4 ReflectionCaptureSM5_PositionLow[341];
} ReflectionCaptureSM5;

layout(set = 1, binding = 3, std140) uniform type_ReflectionStruct
{
    vec4 ReflectionStruct_SkyLightParameters;
} ReflectionStruct;

layout(set = 1, binding = 4, std140) uniform type_ForwardLightStruct
{
    layout(offset = 8) uint ForwardLightStruct_NumReflectionCaptures;
    layout(offset = 16) uint ForwardLightStruct_NumGridCells;
    layout(offset = 32) ivec3 ForwardLightStruct_CulledGridSize;
    layout(offset = 48) uint ForwardLightStruct_LightGridPixelSizeShift;
    layout(offset = 64) vec3 ForwardLightStruct_LightGridZParams;
} ForwardLightStruct;

layout(set = 1, binding = 6, std430) readonly buffer type_StructuredBuffer_uint
{
    uint _m0[];
} ForwardLightStruct_NumCulledLightsGrid;

layout(set = 1, binding = 0, std140) uniform type_Globals
{
    float ApplyBentNormalAO;
    float InvSkySpecularOcclusionStrength;
    vec4 OcclusionTintAndMinOcclusion;
    vec3 ContrastAndNormalizeMulAdd;
    float OcclusionExponent;
    float OcclusionCombineMode;
    float AOMaxViewDistance;
    float DistanceFadeScale;
    vec2 AOBufferBilinearUVMax;
} _Globals;

layout(set = 1, binding = 18) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 1, binding = 19) uniform sampler View_SharedBilinearClampedSampler;
layout(set = 1, binding = 8) uniform texture2D View_ShadingEnergyGGXSpecTexture;
layout(set = 1, binding = 9) uniform textureCube ReflectionStruct_SkyLightCubemap;
layout(set = 1, binding = 20) uniform sampler ReflectionStruct_SkyLightCubemapSampler;
layout(set = 1, binding = 10) uniform textureCube ReflectionStruct_SkyLightBlendDestinationCubemap;
layout(set = 1, binding = 21) uniform sampler ReflectionStruct_SkyLightBlendDestinationCubemapSampler;
layout(set = 1, binding = 7) uniform usamplerBuffer ForwardLightStruct_CulledLightDataGrid16Bit;
layout(set = 1, binding = 11) uniform texture2D SceneDepthTexture;
layout(set = 1, binding = 12) uniform texture2D GBufferATexture;
layout(set = 1, binding = 13) uniform texture2D GBufferBTexture;
layout(set = 1, binding = 14) uniform texture2D GBufferCTexture;
layout(set = 1, binding = 15) uniform texture2D BentNormalAOTexture;
layout(set = 1, binding = 22) uniform sampler BentNormalAOSampler;
layout(set = 1, binding = 16) uniform texture2D ReflectionTexture;
layout(set = 1, binding = 23) uniform sampler ReflectionTextureSampler;
layout(set = 1, binding = 17) uniform texture2D AmbientOcclusionTexture;
layout(set = 1, binding = 24) uniform sampler AmbientOcclusionSampler;

layout(location = 0) out vec4 out_var_SV_Target0;

spirv_instruction(set = "GLSL.std.450", id = 79) float spvNMin(float, float);
spirv_instruction(set = "GLSL.std.450", id = 79) vec2 spvNMin(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 spvNMin(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 79) vec4 spvNMin(vec4, vec4);
spirv_instruction(set = "GLSL.std.450", id = 80) float spvNMax(float, float);
spirv_instruction(set = "GLSL.std.450", id = 80) vec2 spvNMax(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3 spvNMax(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 80) vec4 spvNMax(vec4, vec4);

mediump float spvNMaxRelaxed(mediump float a, mediump float b)
{
    mediump float res = spvNMax(a, b);
    return res;
}

mediump vec2 spvNMaxRelaxed(mediump vec2 a, mediump vec2 b)
{
    mediump vec2 res = spvNMax(a, b);
    return res;
}

mediump vec3 spvNMaxRelaxed(mediump vec3 a, mediump vec3 b)
{
    mediump vec3 res = spvNMax(a, b);
    return res;
}

mediump vec4 spvNMaxRelaxed(mediump vec4 a, mediump vec4 b)
{
    mediump vec4 res = spvNMax(a, b);
    return res;
}

void main()
{
    vec2 _185 = gl_FragCoord.xy * View.View_BufferSizeAndInvSize.zw;
    vec2 _189 = gl_FragCoord.xy - View.View_ViewRectMin.xy;
    vec4 _199 = vec4(((_189 * View.View_ViewSizeAndInvSize.zw) - vec2(0.5)) * vec2(2.0, -2.0), _162, 1.0) * (1.0 / gl_FragCoord.w);
    vec4 _208 = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), _185, 0.0);
    vec4 _212 = textureLod(sampler2D(GBufferCTexture, VulkanGlobalPointClampedSampler), _185, 0.0);
    uvec2 _215 = uvec2(_185 * View.View_BufferSizeAndInvSize.xy);
    bool _224 = (((_215.x + _215.y) + uint(View.View_TemporalAAParams.x)) % 2u) != 0u;
    uint _236 = uint((_208.w * 255.0) + 0.5) & 15u;
    vec3 _240 = vec3(_212.xyz);
    mediump vec3 mp_copy_240 = _240;
    float _241 = _212.w;
    vec3 _243 = (vec3(textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), _185, 0.0).xyz) * 2.0) - vec3(1.0);
    mediump vec3 mp_copy_243 = _243;
    uint _245 = (_236 != 0u) ? 1u : _236;
    mediump vec3 _46 = normalize(mp_copy_243);
    mediump float _71 = (_245 == 9u) ? 0.0 : _208.x;
    mediump vec3 _51 = mix(vec3(0.07999999821186065673828125 * _208.y), mp_copy_240, vec3(_71));
    int _248 = int(_245);
    bool _254;
    if (!(_248 == 5))
    {
        _254 = _248 == 9;
    }
    else
    {
        _254 = true;
    }
    mediump vec3 _72;
    mediump vec3 _73;
    if (_254)
    {
        bool _265;
        if (View.View_bSubsurfacePostprocessEnabled > 0.0)
        {
            _265 = View.View_bCheckerboardSubsurfaceProfileRendering > 0.0;
        }
        else
        {
            _265 = false;
        }
        vec3 _276;
        vec3 _277;
        if (_265)
        {
            _276 = _51 * float(!_224);
            _277 = vec3(float(_224));
        }
        else
        {
            _276 = _51;
            _277 = mix(_240, vec3(1.0), bvec3(View.View_bSubsurfacePostprocessEnabled != 0.0));
        }
        _73 = _277;
        _72 = _276;
    }
    else
    {
        _73 = _240;
        _72 = _51;
    }
    vec3 _290 = (_72 * View.View_SpecularOverrideParameter.w) + View.View_SpecularOverrideParameter.xyz;
    vec4 _834;
    SPIRV_CROSS_BRANCH
    if (_245 != 0u)
    {
        vec4 _297 = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), _185, 0.0);
        float _298 = _297.x;
        float _312 = ((_298 * View.View_InvDeviceZToWorldZTransform.x) + View.View_InvDeviceZToWorldZTransform.y) + (1.0 / ((_298 * View.View_InvDeviceZToWorldZTransform.z) - View.View_InvDeviceZToWorldZTransform.w));
        bool _316 = View.View_ViewToClip[3].w >= 1.0;
        vec2 _317 = _199.xy * _312;
        float _323;
        if (_316)
        {
            _323 = _199.x;
        }
        else
        {
            _323 = _317.x;
        }
        float _329;
        if (_316)
        {
            _329 = _199.y;
        }
        else
        {
            _329 = _317.y;
        }
        vec3 _334 = (View.View_ScreenToTranslatedWorld * vec4(_323, _329, _312, 1.0)).xyz;
        vec3 _340 = normalize(_334 - View.View_TranslatedWorldCameraOrigin);
        float _346;
        if (_316)
        {
            _346 = View.View_ViewForward.x;
        }
        else
        {
            _346 = _340.x;
        }
        float _352;
        if (_316)
        {
            _352 = View.View_ViewForward.y;
        }
        else
        {
            _352 = _340.y;
        }
        float _358;
        if (_316)
        {
            _358 = View.View_ViewForward.z;
        }
        else
        {
            _358 = _340.z;
        }
        vec3 _360 = -vec3(_346, _352, _358);
        vec4 _364 = textureLod(sampler2D(AmbientOcclusionTexture, AmbientOcclusionSampler), _185, 0.0);
        float _365 = _364.x;
        vec4 _369 = texture(sampler2D(ReflectionTexture, ReflectionTextureSampler), _185);
        float _371 = 1.0 - _369.w;
        vec2 _375 = spvNMin(_189 * View.View_BufferSizeAndInvSize.zw, _Globals.AOBufferBilinearUVMax);
        vec2 _377 = floor(View.View_BufferSizeAndInvSize.xy * vec2(0.5));
        vec2 _378 = vec2(1.0) / _377;
        vec2 _384 = (floor((_375 * _377) - vec2(0.5)) / _377) + (_378 * 0.5);
        vec2 _386 = (_375 - _384) * _377;
        vec4 _390 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), _384, 0.0);
        vec4 _397 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), _384 + vec2(_378.x, 0.0), 0.0);
        vec4 _404 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), _384 + vec2(0.0, _378.y), 0.0);
        vec4 _409 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), _384 + _378, 0.0);
        float _410 = _386.y;
        float _411 = 1.0 - _410;
        float _412 = _386.x;
        float _413 = 1.0 - _412;
        vec4 _429 = vec4(_411 * _413, _411 * _412, _410 * _413, _410 * _412) * (vec4(1.0) / (abs(vec4(_390.w, _397.w, _404.w, _409.w) - vec4(_312)) + vec4(9.9999997473787516355514526367188e-05)));
        uvec2 _481 = uvec2(uint(_189.x), uint(_189.y)) >> (uvec2(ForwardLightStruct.ForwardLightStruct_LightGridPixelSizeShift) & uvec2(31u));
        uint _495 = (ForwardLightStruct.ForwardLightStruct_NumGridCells + ((((min(uint(spvNMax(0.0, log2((_312 * ForwardLightStruct.ForwardLightStruct_LightGridZParams.x) + ForwardLightStruct.ForwardLightStruct_LightGridZParams.y) * ForwardLightStruct.ForwardLightStruct_LightGridZParams.z)), uint(ForwardLightStruct.ForwardLightStruct_CulledGridSize.z - 1)) * uint(ForwardLightStruct.ForwardLightStruct_CulledGridSize.y)) + _481.y) * uint(ForwardLightStruct.ForwardLightStruct_CulledGridSize.x)) + _481.x)) * 2u;
        uint _500 = min(ForwardLightStruct_NumCulledLightsGrid._m0[_495], ForwardLightStruct.ForwardLightStruct_NumReflectionCaptures);
        uint _501 = _495 + 1u;
        bool _507 = ReflectionStruct.ReflectionStruct_SkyLightParameters.y > 0.0;
        bool _512;
        if (!_507)
        {
            _512 = _500 > 0u;
        }
        else
        {
            _512 = true;
        }
        mediump float _53 = dot(_46, _360);
        vec3 _521 = _290.xyz;
        float _530 = clamp(spvNMax(_208.z, 0.0), 0.001000000047497451305389404296875, 1.0);
        vec4 _535 = textureLod(sampler2D(View_ShadingEnergyGGXSpecTexture, View_SharedBilinearClampedSampler), vec2(_53, _530), 0.0);
        vec3 _547;
        if (View.View_bShadingEnergyConservation != 0u)
        {
            float _542 = _535.x;
            _547 = vec3(1.0) + (_521 * ((1.0 - _542) / _542));
        }
        else
        {
            _547 = vec3(1.0);
        }
        mediump float mp_copy_566;
        vec3 _564;
        vec3 _565;
        float _566;
        SPIRV_CROSS_BRANCH
        if ((!_512) ? (_371 < 1.0) : true)
        {
            float _557 = _530 * _530;
            float _558 = 1.0 - _557;
            _564 = mix(_46, (_46 * (2.0 * dot(_360, _46))) - _360, vec3(_558 * (sqrt(_558) + _557)));
            _565 = _547 * ((_521 * _535.x) + (((vec3(1.0) * clamp(50.0 * spvNMax(_290.x, spvNMax(_290.y, _290.z)), 0.0, 1.0)) - _521) * _535.y));
            _566 = _530;
        }
        else
        {
            _564 = vec3(0.0);
            _565 = vec3(0.0);
            _566 = 0.0;
        }
        mp_copy_566 = _566;
        vec3 _567 = (((_73 - (_73 * _71)) * View.View_DiffuseOverrideParameter.www) + View.View_DiffuseOverrideParameter.xyz).xyz * 1.0;
        vec3 _569 = mix(_46, ((((_390.xyz * _429.x) + (_397.xyz * _429.y)) + (_404.xyz * _429.z)) + (_409.xyz * _429.w)) * (1.0 / dot(_429, vec4(1.0))), vec3(clamp((_Globals.AOMaxViewDistance - _312) * _Globals.DistanceFadeScale, 0.0, 1.0)));
        bool _571 = any(greaterThan(_567, vec3(0.0)));
        vec3 _671;
        if ((!_571) ? any(bvec3(false)) : true)
        {
            float _577 = length(_569);
            vec3 _580 = _569 / vec3(spvNMax(_577, 9.9999997473787516355514526367188e-06));
            vec3 _582 = mix(_580, _46, vec3(_577));
            float _607 = mix(pow(clamp(((1.0 / (1.0 + exp((-_Globals.ContrastAndNormalizeMulAdd.x) * ((_577 * 10.0) - 5.0)))) * _Globals.ContrastAndNormalizeMulAdd.y) + _Globals.ContrastAndNormalizeMulAdd.z, 0.0, 1.0), _Globals.OcclusionExponent), 1.0, _Globals.OcclusionTintAndMinOcclusion.w);
            float _618;
            if (_Globals.OcclusionCombineMode == 0.0)
            {
                _618 = spvNMin(_607, spvNMin(_241, _365));
            }
            else
            {
                _618 = _607 * spvNMin(_241, _365);
            }
            vec3 _670;
            if (_571)
            {
                float _626 = _582.x;
                float _627 = _582.y;
                vec4 _629 = vec4(_626, _627, _582.z, 1.0);
                vec4 _642 = _629.xyzz * _629.yzzx;
                _670 = (((spvNMax(vec3(0.0), (vec3(dot(View_SkyIrradianceEnvironmentMap._m0[0u], _629), dot(View_SkyIrradianceEnvironmentMap._m0[1u], _629), dot(View_SkyIrradianceEnvironmentMap._m0[2u], _629)) + vec3(dot(View_SkyIrradianceEnvironmentMap._m0[3u], _642), dot(View_SkyIrradianceEnvironmentMap._m0[4u], _642), dot(View_SkyIrradianceEnvironmentMap._m0[5u], _642))) + (View_SkyIrradianceEnvironmentMap._m0[6u].xyz * ((_626 * _626) - (_627 * _627)))) * View.View_SkyLightColor.xyz) * (_618 * mix(dot(_580, _46), 1.0, _577))) + (_Globals.OcclusionTintAndMinOcclusion.xyz * (1.0 - _618))) * _567;
            }
            else
            {
                _670 = vec3(0.0);
            }
            _671 = _670;
        }
        else
        {
            _671 = vec3(0.0);
        }
        vec3 _800;
        if (any(greaterThan(_565, vec3(0.0))))
        {
            vec3 _720;
            float _721;
            SPIRV_CROSS_BRANCH
            if (_Globals.ApplyBentNormalAO > 0.0)
            {
                float _681 = length(_569);
                float _711;
                SPIRV_CROSS_BRANCH
                if (View.View_DistanceFieldAOSpecularOcclusionMode == 0u)
                {
                    _711 = _681;
                }
                else
                {
                    float _689 = spvNMax(_566, 0.100000001490116119384765625) * 3.140625;
                    float _693 = (_681 * 3.140625) * _Globals.InvSkySpecularOcclusionStrength;
                    float _699 = abs(_689 - _693);
                    _711 = mix(0.0, smoothstep(0.0, 1.0, 1.0 - clamp((acos(dot(_569, _564) / spvNMax(_681, 0.001000000047497451305389404296875)) - _699) / ((_689 + _693) - _699), 0.0, 1.0)), clamp((_693 - 0.100000001490116119384765625) * 5.0, 0.0, 1.0));
                }
                float _715 = mix(_711, 1.0, _Globals.OcclusionTintAndMinOcclusion.w);
                _720 = _Globals.OcclusionTintAndMinOcclusion.xyz * (1.0 - _715);
                _721 = _715;
            }
            else
            {
                _720 = vec3(0.0);
                _721 = 1.0;
            }
            float _722 = _566 * _566;
            mediump float mp_copy_722 = _722;
            vec4 _726;
            _726 = vec4(0.0, 0.0, 0.0, _371 * clamp(pow(clamp(_53, 0.0, 1.0) + 1.0, mp_copy_722), 0.0, 1.0));
            vec4 _727;
            SPIRV_CROSS_LOOP
            for (uint _729 = 0u; _729 < _500; _726 = _727, _729++)
            {
                SPIRV_CROSS_BRANCH
                if (_726.w < 0.001000000047497451305389404296875)
                {
                    break;
                }
                uvec4 _740 = texelFetch(ForwardLightStruct_CulledLightDataGrid16Bit, int(ForwardLightStruct_NumCulledLightsGrid._m0[_501] + _729));
                uint _741 = _740.x;
                precise vec3 _57 = ReflectionCaptureSM5.ReflectionCaptureSM5_PositionHighAndRadius[_741].xyz + View.View_PreViewTranslationHigh;
                precise vec3 _58 = _57 - ReflectionCaptureSM5.ReflectionCaptureSM5_PositionHighAndRadius[_741].xyz;
                precise vec3 _59 = _57 - _58;
                precise vec3 _60 = ReflectionCaptureSM5.ReflectionCaptureSM5_PositionHighAndRadius[_741].xyz - _59;
                precise vec3 _61 = View.View_PreViewTranslationHigh - _58;
                precise vec3 _62 = _60 + _61;
                vec3 _753 = _334 - (_57 + (_62 + (ReflectionCaptureSM5.ReflectionCaptureSM5_PositionLow[_741].xyz + View.View_PreViewTranslationLow)));
                SPIRV_CROSS_BRANCH
                if (sqrt(dot(_753, _753)) < ReflectionCaptureSM5.ReflectionCaptureSM5_PositionHighAndRadius[_741].w)
                {
                    vec4 _759 = vec4(_726.x, _726.y, _726.z, _726.w);
                    _759.w = _726.w;
                    _727 = _759;
                }
                else
                {
                    _727 = _726;
                }
            }
            vec3 _793;
            SPIRV_CROSS_BRANCH
            if (_507 ? true : false)
            {
                float _769 = ReflectionStruct.ReflectionStruct_SkyLightParameters.x;
                mediump float mp_copy_769 = _769;
                mediump float _70 = (mp_copy_769 - 1.0) - (1.0 - (1.2001953125 * log2(spvNMaxRelaxed(mp_copy_566, 0.00100040435791015625))));
                vec3 _777 = textureLod(samplerCube(ReflectionStruct_SkyLightCubemap, ReflectionStruct_SkyLightCubemapSampler), _564, _70).xyz * View.View_SkyLightColor.xyz;
                vec3 _790;
                SPIRV_CROSS_BRANCH
                if (ReflectionStruct.ReflectionStruct_SkyLightParameters.w > 0.0)
                {
                    _790 = mix(_777, textureLod(samplerCube(ReflectionStruct_SkyLightBlendDestinationCubemap, ReflectionStruct_SkyLightBlendDestinationCubemapSampler), _564, _70).xyz * View.View_SkyLightColor.xyz, vec3(ReflectionStruct.ReflectionStruct_SkyLightParameters.w));
                }
                else
                {
                    _790 = _777;
                }
                _793 = _720 + (_790 * _721);
            }
            else
            {
                _793 = _720;
            }
            _800 = _565 * ((_726.xyz * View.View_PrecomputedIndirectSpecularColorScale).xyz + (_793 * _726.w)).xyz;
        }
        else
        {
            _800 = vec3(0.0);
        }
        vec4 _804 = vec4(_671, 0.0);
        bool _817;
        if (View.View_bCheckerboardSubsurfaceProfileRendering == 0.0)
        {
            _817 = View.View_bSubsurfacePostprocessEnabled != 0.0;
        }
        else
        {
            _817 = false;
        }
        vec4 _821;
        if (_817)
        {
            vec4 _820 = _804;
            _820.w = 0.0;
            _821 = _820;
        }
        else
        {
            _821 = _804;
        }
        _834 = ((_821 + vec4(_800, 0.0)) * View.View_PreExposure) + vec4((_369.xyz * _565) * 1.0, 0.0);
    }
    else
    {
        _834 = vec4(0.0);
    }
    out_var_SV_Target0 = _834;
}

