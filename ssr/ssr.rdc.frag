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

vec4 _105;
bvec4 _106;

layout(set = 1, binding = 1, std140) uniform type_View
{
    layout(offset = 0) mat4 View_TranslatedWorldToClip;
    layout(offset = 192) mat4 View_TranslatedWorldToView;
    layout(offset = 448) mat4 View_ViewToClip;
    layout(offset = 832) mat4 View_ScreenToTranslatedWorld;
    layout(offset = 1168) vec3 View_ViewForward;
    layout(offset = 1248) vec4 View_InvDeviceZToWorldZTransform;
    layout(offset = 1264) vec4 View_ScreenPositionScaleBias;
    layout(offset = 1296) vec3 View_TranslatedWorldCameraOrigin;
    layout(offset = 2176) mat4 View_ClipToPrevClip;
    layout(offset = 2368) vec4 View_ViewRectMin;
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;
    layout(offset = 2648) uint View_StateFrameIndexMod8;
} View;

layout(set = 1, binding = 0, std140) uniform type_Globals
{
    vec4 HZBUvFactorAndInvFactor;
    vec4 SSRParams;
    float PrevSceneColorPreExposureCorrection;
    vec4 PrevScreenPositionScaleBias;
} _Globals;

layout(set = 1, binding = 10) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 1, binding = 2) uniform texture2D SceneDepthTexture;
layout(set = 1, binding = 3) uniform texture2D GBufferATexture;
layout(set = 1, binding = 4) uniform texture2D GBufferBTexture;
layout(set = 1, binding = 5) uniform texture2D GBufferDTexture;
layout(set = 1, binding = 6) uniform texture2D GBufferVelocityTexture;
layout(set = 1, binding = 7) uniform texture2D GBufferFTexture;
layout(set = 1, binding = 8) uniform texture2D HZBTexture;
layout(set = 1, binding = 9) uniform texture2D SceneColor;
layout(set = 1, binding = 11) uniform sampler SceneColorSampler;

layout(location = 0) out vec4 out_var_SV_Target0;

spirv_instruction(set = "GLSL.std.450", id = 79) float spvNMin(float, float);
spirv_instruction(set = "GLSL.std.450", id = 79) vec2 spvNMin(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 spvNMin(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 79) vec4 spvNMin(vec4, vec4);
spirv_instruction(set = "GLSL.std.450", id = 80) float spvNMax(float, float);
spirv_instruction(set = "GLSL.std.450", id = 80) vec2 spvNMax(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3 spvNMax(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 80) vec4 spvNMax(vec4, vec4);

void main()
{
    vec4 _662;
    do
    {
        vec2 _122 = gl_FragCoord.xy * View.View_BufferSizeAndInvSize.zw;
        vec2 _130 = (gl_FragCoord.xy - View.View_ViewRectMin.xy) * View.View_ViewSizeAndInvSize.zw;
        float _133 = (2.0 * _130.x) - 1.0;
        float _136 = 1.0 - (2.0 * _130.y);
        vec4 _141 = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), _122, 0.0);
        float _142 = _141.x;
        float _156 = ((_142 * View.View_InvDeviceZToWorldZTransform.x) + View.View_InvDeviceZToWorldZTransform.y) + (1.0 / ((_142 * View.View_InvDeviceZToWorldZTransform.z) - View.View_InvDeviceZToWorldZTransform.w));
        vec4 _160 = textureLod(sampler2D(GBufferFTexture, VulkanGlobalPointClampedSampler), _122, 0.0);
        vec4 _168 = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), _122, 0.0);
        float _177 = _168.z;
        uint _181 = uint((_168.w * 255.0) + 0.5);
        uint _182 = _181 & 15u;
        vec3 _186 = (vec3(textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), _122, 0.0).xyz) * 2.0) - vec3(1.0);
        mediump vec3 mp_copy_186 = _186;
        int _189 = int(_182);
        bool _195;
        if (!(_189 == 2))
        {
            _195 = _189 == 3;
        }
        else
        {
            _195 = true;
        }
        bool _200;
        if (!_195)
        {
            _200 = _189 == 4;
        }
        else
        {
            _200 = true;
        }
        bool _205;
        if (!_200)
        {
            _205 = _189 == 5;
        }
        else
        {
            _205 = true;
        }
        bool _210;
        if (!_205)
        {
            _210 = _189 == 6;
        }
        else
        {
            _210 = true;
        }
        bool _215;
        if (!_210)
        {
            _215 = _189 == 7;
        }
        else
        {
            _215 = true;
        }
        bool _220;
        if (!_215)
        {
            _220 = _189 == 8;
        }
        else
        {
            _220 = true;
        }
        bool _225;
        if (!_220)
        {
            _225 = _189 == 9;
        }
        else
        {
            _225 = true;
        }
        vec3 hp_copy_25;
        vec4 _227 = mix(vec4(0.0), textureLod(sampler2D(GBufferDTexture, VulkanGlobalPointClampedSampler), _122, 0.0), bvec4(_225));
        mediump vec3 _22 = normalize(mp_copy_186);
        vec3 hp_copy_22 = _22;
        mediump float _24;
        mediump vec3 _25;
        if ((((_181 >> 4u) & 15u) & 1u) != 0u)
        {
            vec3 _234 = (_160.xyz * 2.0) - vec3(1.0);
            mediump vec3 mp_copy_234 = _234;
            _25 = normalize(mp_copy_234);
            _24 = (_160.w * 2.0) - 1.0;
        }
        else
        {
            _25 = vec3(0.0);
            _24 = 0.0;
        }
        hp_copy_25 = _25;
        bool _240 = View.View_ViewToClip[3].w >= 1.0;
        vec2 _241 = vec2(_133, _136) * _156;
        float _246;
        if (_240)
        {
            _246 = _133;
        }
        else
        {
            _246 = _241.x;
        }
        float _251;
        if (_240)
        {
            _251 = _136;
        }
        else
        {
            _251 = _241.y;
        }
        vec4 _255 = View.View_ScreenToTranslatedWorld * vec4(_246, _251, _156, 1.0);
        vec3 _256 = _255.xyz;
        vec3 _262 = normalize(_256 - View.View_TranslatedWorldCameraOrigin);
        float _268;
        if (_240)
        {
            _268 = View.View_ViewForward.x;
        }
        else
        {
            _268 = _262.x;
        }
        float _274;
        if (_240)
        {
            _274 = View.View_ViewForward.y;
        }
        else
        {
            _274 = _262.y;
        }
        float _280;
        if (_240)
        {
            _280 = View.View_ViewForward.z;
        }
        else
        {
            _280 = _262.z;
        }
        vec3 _281 = vec3(_268, _274, _280);
        float _283 = _227.x;
        bool _284 = _182 == 4u;
        float _286 = mix(_24, 0.0, _284 ? _283 : 0.0);
        float _287 = abs(_286);
        vec3 _304;
        if (_287 > 0.0)
        {
            vec3 _295 = mix(_25, normalize(cross(hp_copy_22, hp_copy_25)), bvec3(_286 >= 0.0));
            _304 = normalize(mix(_22, cross(cross(_295, -_281), _295), vec3(_287 * clamp(5.0 * _177, 0.0, 1.0))));
        }
        else
        {
            _304 = _22;
        }
        float _309;
        SPIRV_CROSS_BRANCH
        if (_284)
        {
            _309 = mix(_177, _227.y, _283);
        }
        else
        {
            _309 = _177;
        }
        float _315 = spvNMin((_309 * _Globals.SSRParams.y) + 2.0, 1.0);
        SPIRV_CROSS_BRANCH
        if ((!(_315 <= 0.0)) ? (_182 == 0u) : true)
        {
            _662 = vec4(0.0);
            break;
        }
        vec3 _331 = reflect(_281, _304);
        float _336 = spvNMin(_156, 1000000.0);
        vec4 _343 = View.View_TranslatedWorldToView * vec4(_331, 0.0);
        float _344 = _343.z;
        float _352;
        if (_344 < 0.0)
        {
            _352 = spvNMin(((-0.949999988079071044921875) * _156) / _344, _336);
        }
        else
        {
            _352 = _336;
        }
        vec4 _361 = View.View_TranslatedWorldToClip * vec4(_255.xyz, 1.0);
        vec4 _366 = View.View_TranslatedWorldToClip * vec4(_256 + (_331 * _352), 1.0);
        vec3 _370 = _361.xyz * (1.0 / _361.w);
        vec4 _377 = _361 + (View.View_ViewToClip * vec4(0.0, 0.0, _352, 0.0));
        vec3 _381 = _377.xyz * (1.0 / _377.w);
        vec3 _382 = (_366.xyz * (1.0 / _366.w)) - _370;
        vec2 _383 = _370.xy;
        vec2 _384 = _382.xy;
        float _386 = 0.5 * length(_384);
        vec2 _395 = vec2(1.0) - (spvNMax(abs(_384 + (_383 * _386)) - vec2(_386), vec2(0.0)) / abs(_384));
        vec3 _400 = _382 * (spvNMin(_395.x, _395.y) / _386);
        float _416;
        if (_240)
        {
            _416 = spvNMax(0.0, (_370.z - _381.z) * 4.0);
        }
        else
        {
            _416 = spvNMax(abs(_400.z), (_370.z - _381.z) * 4.0);
        }
        vec2 _427 = (_400.xy * vec2(0.5, -0.5)) * _Globals.HZBUvFactorAndInvFactor.xy;
        float _432 = _416 * 0.0625;
        vec3 _433 = vec3(_427, _400.z) * 0.0625;
        vec3 _435 = vec3(((_383 * vec2(0.5, -0.5)) + vec2(0.5)) * _Globals.HZBUvFactorAndInvFactor.xy, _370.z) + (_433 * (fract(52.98291778564453125 * fract(dot(gl_FragCoord.xy + (vec2(32.66500091552734375, 11.81499958038330078125) * float(View.View_StateFrameIndexMod8)), vec2(0.067110560834407806396484375, 0.005837149918079376220703125)))) - 0.5));
        bvec4 _437;
        vec4 _440;
        uint _446;
        float _448;
        _437 = _106;
        _440 = _105;
        _446 = 0u;
        _448 = 0.0;
        bvec4 _438;
        vec4 _441;
        bool _443;
        float _445;
        float _449;
        bvec4 _530;
        vec4 _531;
        bool _532;
        bool _442 = false;
        float _444 = 1.0;
        SPIRV_CROSS_LOOP
        for (;;)
        {
            if (_446 < 16u)
            {
                vec2 _453 = _435.xy;
                float _455 = float(_446);
                float _456 = _455 + 1.0;
                float _459 = _435.z;
                float _463 = _455 + 2.0;
                float _468 = _455 + 3.0;
                float _473 = _455 + 4.0;
                float _479 = 0.5 * _309;
                float _480 = _444 + _479;
                _445 = _480 + _479;
                vec4 _493 = vec4(textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), _453 + (_433.xy * _456), _444).x, textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), _453 + (_433.xy * _463), _444).x, textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), _453 + (_433.xy * _468), _480).x, textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), _453 + (_433.xy * _473), _480).x);
                _441 = vec4(_459 + (_456 * _433.z), _459 + (_463 * _433.z), _459 + (_468 * _433.z), _459 + (_473 * _433.z)) - _493;
                vec4 _494 = vec4(_432);
                bvec4 _497 = lessThan(abs(_441 + _494), _494);
                bvec4 _498 = notEqual(_493, vec4(0.0));
                bool _503;
                if (_497.x)
                {
                    _503 = _498.x;
                }
                else
                {
                    _503 = false;
                }
                bool _508;
                if (_497.y)
                {
                    _508 = _498.y;
                }
                else
                {
                    _508 = false;
                }
                bool _513;
                if (_497.z)
                {
                    _513 = _498.z;
                }
                else
                {
                    _513 = false;
                }
                bool _518;
                if (_497.w)
                {
                    _518 = _498.w;
                }
                else
                {
                    _518 = false;
                }
                _438 = bvec4(_503, _508, _513, _518);
                _443 = (!((!((!((!_442) ? _503 : true)) ? _508 : true)) ? _513 : true)) ? _518 : true;
                SPIRV_CROSS_BRANCH
                if ((!_443) ? false : true)
                {
                    _530 = _438;
                    _531 = _441;
                    _532 = _443;
                    break;
                }
                _449 = _441.w;
                _437 = _438;
                _440 = _441;
                _442 = _443;
                _444 = _445;
                _446 += 4u;
                _448 = _449;
                continue;
            }
            else
            {
                _530 = _437;
                _531 = _440;
                _532 = _442;
                break;
            }
        }
        vec3 _571;
        SPIRV_CROSS_BRANCH
        if (_532)
        {
            float _545;
            SPIRV_CROSS_FLATTEN
            if (_530.z)
            {
                _545 = _531.y;
            }
            else
            {
                _545 = _531.z;
            }
            float _553;
            float _554;
            SPIRV_CROSS_FLATTEN
            if (_530.y)
            {
                _553 = _531.y;
                _554 = _531.x;
            }
            else
            {
                _553 = _530.z ? _531.z : _531.w;
                _554 = _545;
            }
            float _560;
            SPIRV_CROSS_FLATTEN
            if (_530.x)
            {
                _560 = _531.x;
            }
            else
            {
                _560 = _553;
            }
            float _561 = _530.x ? _448 : _554;
            _571 = _435 + (_433 * (((_530.x ? 0.0 : (_530.y ? 1.0 : (_530.z ? 2.0 : 3.0))) + float(_446)) + clamp(_561 / (_561 - _560), 0.0, 1.0)));
        }
        else
        {
            _571 = _435 + (_433 * float(_446));
        }
        vec4 _652;
        SPIRV_CROSS_BRANCH
        if (_532)
        {
            vec2 _591 = (((((_571.xy * _Globals.HZBUvFactorAndInvFactor.zw).xy * vec2(2.0, -2.0)) + vec2(-1.0, 1.0)).xy * View.View_ScreenPositionScaleBias.xy) + View.View_ScreenPositionScaleBias.wz).xy;
            vec2 _593 = (_591 - View.View_ScreenPositionScaleBias.wz) / View.View_ScreenPositionScaleBias.xy;
            vec4 _597 = vec4(_593, _571.z, 1.0);
            vec4 _600 = View.View_ClipToPrevClip * _597;
            vec4 _606 = textureLod(sampler2D(GBufferVelocityTexture, VulkanGlobalPointClampedSampler), _591, 0.0);
            vec2 _621;
            if (_606.x > 0.0)
            {
                vec2 _615 = ((_606.xy * 4.008016109466552734375) - vec2(2.0039775371551513671875)).xy;
                _621 = _597.xy - ((_615 * abs(_615)) * 0.5).xy;
            }
            else
            {
                _621 = _600.xy / vec2(_600.w);
            }
            vec2 _629 = clamp((abs(_593) * 5.0) - vec2(4.0), vec2(0.0), vec2(1.0));
            vec2 _636 = clamp((abs(_621) * 5.0) - vec2(4.0), vec2(0.0), vec2(1.0));
            vec3 _648 = -spvNMin(-textureLod(sampler2D(SceneColor, SceneColorSampler), (_621 * _Globals.PrevScreenPositionScaleBias.xy) + _Globals.PrevScreenPositionScaleBias.zw, 0.0).xyz, vec3(0.0));
            vec4 _649 = vec4(_648.x, _648.y, _648.z, _105.w);
            _649.w = 1.0;
            _652 = _649 * spvNMin(clamp(1.0 - dot(_629, _629), 0.0, 1.0), clamp(1.0 - dot(_636, _636), 0.0, 1.0));
        }
        else
        {
            _652 = vec4(0.0);
        }
        vec4 _656 = (_652 * _315) * _Globals.SSRParams.x;
        vec3 _660 = _656.xyz * _Globals.PrevSceneColorPreExposureCorrection;
        _662 = vec4(_660.x, _660.y, _660.z, _656.w);
        break;
    } while(false);
    out_var_SV_Target0 = _662;
}

