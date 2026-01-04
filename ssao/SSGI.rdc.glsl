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
layout(local_size_x = 4, local_size_y = 4, local_size_z = 16) in;

float _139;
bvec4 _140;
uint _141;

layout(set = 0, binding = 1, std140) uniform type_View
{
    layout(offset = 192) mat4 View_TranslatedWorldToView;
    layout(offset = 448) mat4 View_ViewToClip;
    layout(offset = 1264) vec4 View_ScreenPositionScaleBias;
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;
    layout(offset = 2648) uint View_StateFrameIndexMod8;
} View;

layout(set = 0, binding = 0, std140) uniform type_Globals
{
    layout(offset = 16) vec4 HZBUvFactorAndInvFactor;
    layout(offset = 32) vec4 ColorBufferScaleBias;
    layout(offset = 48) vec2 ReducedColorUVMax;
    layout(offset = 56) float PixelPositionToFullResPixel;
    layout(offset = 64) vec2 FullResPixelOffset;
} _Globals;

layout(set = 0, binding = 9) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 0, binding = 4) uniform texture2D SceneDepthTexture;
layout(set = 0, binding = 5) uniform texture2D GBufferATexture;
layout(set = 0, binding = 6) uniform texture2D GBufferBTexture;
layout(set = 0, binding = 7) uniform texture2D FurthestHZBTexture;
layout(set = 0, binding = 8) uniform texture2D ColorTexture;
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D IndirectDiffuseOutput;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D AmbientOcclusionOutput;

shared uint SharedMemory0[512];

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
    uint _149 = gl_LocalInvocationIndex % 16u;
    uint _150 = gl_LocalInvocationIndex / 16u;
    SPIRV_CROSS_BRANCH
    if (_150 == 0u)
    {
        vec2 _176 = ((vec2((gl_WorkGroupID.xy * uvec2(4u)) + uvec2(_149 % 4u, (_149 >> 2u) % 4u)) * _Globals.PixelPositionToFullResPixel) + _Globals.FullResPixelOffset) * View.View_BufferSizeAndInvSize.zw;
        vec3 _195 = (vec3(textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), _176, 0.0).xyz) * 2.0) - vec3(1.0);
        mediump vec3 mp_copy_195 = _195;
        uvec3 _214 = uvec3(clamp(((View.View_TranslatedWorldToView * vec4(normalize(mp_copy_195), 0.0)).xyz * 0.5) + vec3(0.5), vec3(0.0), vec3(1.0)) * 255.0);
        SharedMemory0[_149] = (_214.x | (_214.y << 8u)) | (_214.z << 16u);
        SharedMemory0[16u | _149] = floatBitsToUint(((uint((textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), _176, 0.0).w * 255.0) + 0.5) & 15u) != 0u) ? textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), _176, 0.0).x : (-1.0));
    }
    else
    {
        if ((gl_LocalInvocationIndex / 64u) == 1u)
        {
            SharedMemory0[32u | _149] = 0u;
        }
    }
    barrier();
    uint _228 = SharedMemory0[_149];
    uint _231 = SharedMemory0[16u | _149];
    float _232 = uintBitsToFloat(_231);
    barrier();
    uvec2 _620;
    SPIRV_CROSS_BRANCH
    if (_232 > 0.0)
    {
        vec3 _245 = (vec3(uvec3(_228 & 255u, (_228 >> 8u) & 255u, (_228 >> 16u) & 255u)) * 0.007843137718737125396728515625) - vec3(1.0);
        uvec2 _251 = (gl_WorkGroupID.xy * uvec2(4u)) + uvec2(_149 % 4u, (_149 >> 2u) % 4u);
        vec2 _252 = vec2(_251);
        vec2 _262 = ((_252 * _Globals.PixelPositionToFullResPixel) + _Globals.FullResPixelOffset) * View.View_ViewSizeAndInvSize.zw;
        float _265 = (2.0 * _262.x) - 1.0;
        float _268 = 1.0 - (2.0 * _262.y);
        uvec3 _288 = (uvec3(ivec3(int(_251.x), int(_251.y), int(View.View_StateFrameIndexMod8))) * uvec3(1664525u)) + uvec3(1013904223u);
        uint _289 = _288.y;
        uint _290 = _288.z;
        uint _293 = _288.x + (_289 * _290);
        uint _295 = _289 + (_290 * _293);
        uint _297 = _290 + (_293 * _295);
        uint _299 = _293 + (_295 * _297);
        uvec3 _303 = uvec3(_299, _295 + (_297 * _299), _141) >> uvec3(16u);
        float _318 = _245.z;
        float _321 = float((_318 >= 0.0) ? 1 : (-1));
        float _323 = (-1.0) / (_321 + _318);
        float _324 = _245.x;
        float _325 = _245.y;
        float _327 = (_324 * _325) * _323;
        vec2 _343 = (vec2(fract((float(_150) * 0.0625) + (float(_303.x) * 1.52587890625e-05)), float((bitfieldReverse(_150) >> 16u) ^ _303.y) * 1.52587890625e-05) * 2.0) - vec2(0.999999940395355224609375);
        vec2 _344 = abs(_343);
        float _345 = _344.x;
        float _346 = _344.y;
        float _348 = spvNMax(_345, _346);
        float _355 = 0.785398185253143310546875 * ((spvNMin(_345, _346) / (_348 + 5.4210108624275221700372640043497e-20)) + (2.0 * float(_346 >= _345)));
        vec3 _377 = mat3(vec3(1.0 + ((_321 * _323) * (_324 * _324)), _321 * _327, (-_321) * _324), vec3(_327, _321 + (_323 * (_325 * _325)), -_325), _245) * vec4(vec3(uintBitsToFloat((floatBitsToUint(vec2(cos(_355), sin(_355))) & uvec2(2147483647u)) | (floatBitsToUint(_343) & uvec2(2147483648u))), _348).xy * _348, sqrt(1.0 - (_348 * _348)), _139).xyz;
        vec3 _378 = vec3(_265, _268, _232);
        float _381 = _377.z;
        vec4 _398 = vec4(vec4(_377.xy, _381, 0.0).xy * vec2(View.View_ViewToClip[0u].x, View.View_ViewToClip[1u].y), _381 * View.View_ViewToClip[2u].z, _381) + vec4(_265, _268, _232, 1.0);
        vec3 _408 = (_398.xyz * (1.0 / _398.w)) - _378;
        vec2 _409 = _378.xy;
        vec2 _410 = _408.xy;
        float _412 = 0.5 * length(_410);
        vec2 _421 = vec2(1.0) - (spvNMax(abs(_410 + (_409 * _412)) - vec2(_412), vec2(0.0)) / abs(_410));
        vec3 _426 = _408 * (spvNMin(_421.x, _421.y) / _412);
        float _427 = _426.z;
        float _450 = spvNMax(abs(_427), (_232 - ((_378 + (View.View_ViewToClip * vec4(0.0, 0.0, 1.0, 0.0)).xyz) * 0.5).z) * 2.0) * 0.125;
        vec3 _451 = vec3((_426.xy * vec2(0.5, -0.5)) * _Globals.HZBUvFactorAndInvFactor.xy, _427) * 0.125;
        vec3 _453 = vec3(((_409 * vec2(0.5, -0.5)) + vec2(0.5)) * _Globals.HZBUvFactorAndInvFactor.xy, _232) + (_451 * (fract(52.98291778564453125 * fract(dot((_252 + vec2(0.5)) + (vec2(32.66500091552734375, 11.81499958038330078125) * float(View.View_StateFrameIndexMod8)), vec2(0.067110560834407806396484375, 0.005837149918079376220703125)))) - 0.89999997615814208984375));
        bvec4 _455;
        uint _462;
        _455 = _140;
        _462 = 0u;
        bvec4 _456;
        bool _459;
        float _461;
        bvec4 _544;
        float _545;
        bool _546;
        bool _458 = false;
        float _460 = 1.0;
        SPIRV_CROSS_LOOP
        for (;;)
        {
            if (_462 < 8u)
            {
                vec2 _467 = _453.xy;
                vec2 _468 = _451.xy;
                float _469 = float(_462);
                float _470 = _469 + 1.0;
                float _473 = _453.z;
                float _474 = _451.z;
                float _477 = _469 + 2.0;
                float _482 = _469 + 3.0;
                float _487 = _469 + 4.0;
                float _493 = _460 + 1.0;
                _461 = _460 + 2.0;
                vec4 _506 = vec4(textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), _467 + (_468 * _470), _460).x, textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), _467 + (_468 * _477), _460).x, textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), _467 + (_468 * _482), _493).x, textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), _467 + (_468 * _487), _493).x);
                vec4 _508 = vec4(_450);
                bvec4 _511 = lessThan(abs((vec4(_473 + (_470 * _474), _473 + (_477 * _474), _473 + (_482 * _474), _473 + (_487 * _474)) - _506) + _508), _508);
                bvec4 _512 = notEqual(_506, vec4(0.0));
                bool _517;
                if (_511.x)
                {
                    _517 = _512.x;
                }
                else
                {
                    _517 = false;
                }
                bool _522;
                if (_511.y)
                {
                    _522 = _512.y;
                }
                else
                {
                    _522 = false;
                }
                bool _527;
                if (_511.z)
                {
                    _527 = _512.z;
                }
                else
                {
                    _527 = false;
                }
                bool _532;
                if (_511.w)
                {
                    _532 = _512.w;
                }
                else
                {
                    _532 = false;
                }
                _456 = bvec4(_517, _522, _527, _532);
                _459 = (!((!((!((!_458) ? _517 : true)) ? _522 : true)) ? _527 : true)) ? _532 : true;
                SPIRV_CROSS_BRANCH
                if ((!_459) ? false : true)
                {
                    _544 = _456;
                    _545 = _461;
                    _546 = _459;
                    break;
                }
                _455 = _456;
                _458 = _459;
                _460 = _461;
                _462 += 4u;
                continue;
            }
            else
            {
                _544 = _455;
                _545 = _460;
                _546 = _458;
                break;
            }
        }
        vec3 _568;
        SPIRV_CROSS_BRANCH
        if (_546)
        {
            _568 = _453 + (_451 * (float(_462) + spvNMin(spvNMin(_544.x ? 1.0 : 5.0, spvNMin(_544.y ? 2.0 : 5.0, _544.z ? 3.0 : 5.0)), _544.w ? 4.0 : 5.0)));
        }
        else
        {
            _568 = _453 + (_451 * float(_462));
        }
        uvec2 _619;
        SPIRV_CROSS_BRANCH
        if (_546)
        {
            vec3 _599 = textureLod(sampler2D(ColorTexture, VulkanGlobalPointClampedSampler), spvNMin(((((((_568.xy * _Globals.HZBUvFactorAndInvFactor.zw).xy * vec2(2.0, -2.0)) + vec2(-1.0, 1.0)).xy * View.View_ScreenPositionScaleBias.xy) + View.View_ScreenPositionScaleBias.wz).xy * _Globals.ColorBufferScaleBias.xy) + _Globals.ColorBufferScaleBias.zw, _Globals.ReducedColorUVMax), _545).xyz;
            vec3 _603 = _599 * (1.0 / (1.0 + dot(_599, vec3(0.2126390039920806884765625, 0.715168654918670654296875, 0.072192318737506866455078125))));
            uint _616 = packHalf2x16(vec2(1.0, 0.0));
            _619 = uvec2((packHalf2x16(vec2(_603.x, 0.0)) << 16u) | packHalf2x16(vec2(_603.y, 0.0)), (packHalf2x16(vec2(_603.z, 0.0)) << 16u) | _616);
        }
        else
        {
            _619 = uvec2(0u);
        }
        _620 = _619;
    }
    else
    {
        _620 = uvec2(0u);
    }
    uint _622 = _149 + (_150 * 16u);
    SharedMemory0[_622] = _620.x;
    SharedMemory0[256u | _622] = _620.y;
    barrier();
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 16u)
    {
        uint _633 = 256u | gl_LocalInvocationIndex;
        uint _646 = gl_LocalInvocationIndex + 16u;
        uint _649 = 256u | _646;
        uint _666 = gl_LocalInvocationIndex + 32u;
        uint _669 = 256u | _666;
        uint _686 = gl_LocalInvocationIndex + 48u;
        uint _689 = 256u | _686;
        uint _706 = gl_LocalInvocationIndex + 64u;
        uint _709 = 256u | _706;
        uint _726 = gl_LocalInvocationIndex + 80u;
        uint _729 = 256u | _726;
        uint _746 = gl_LocalInvocationIndex + 96u;
        uint _749 = 256u | _746;
        uint _766 = gl_LocalInvocationIndex + 112u;
        uint _769 = 256u | _766;
        uint _786 = gl_LocalInvocationIndex + 128u;
        uint _789 = 256u | _786;
        uint _806 = gl_LocalInvocationIndex + 144u;
        uint _809 = 256u | _806;
        uint _826 = gl_LocalInvocationIndex + 160u;
        uint _829 = 256u | _826;
        float _835 = (((((((((unpackHalf2x16(SharedMemory0[gl_LocalInvocationIndex] >> 16u).x + unpackHalf2x16(SharedMemory0[_646] >> 16u).x) + unpackHalf2x16(SharedMemory0[_666] >> 16u).x) + unpackHalf2x16(SharedMemory0[_686] >> 16u).x) + unpackHalf2x16(SharedMemory0[_706] >> 16u).x) + unpackHalf2x16(SharedMemory0[_726] >> 16u).x) + unpackHalf2x16(SharedMemory0[_746] >> 16u).x) + unpackHalf2x16(SharedMemory0[_766] >> 16u).x) + unpackHalf2x16(SharedMemory0[_786] >> 16u).x) + unpackHalf2x16(SharedMemory0[_806] >> 16u).x) + unpackHalf2x16(SharedMemory0[_826] >> 16u).x;
        float _842 = (((((((((unpackHalf2x16(SharedMemory0[_633] >> 16u).x + unpackHalf2x16(SharedMemory0[_649] >> 16u).x) + unpackHalf2x16(SharedMemory0[_669] >> 16u).x) + unpackHalf2x16(SharedMemory0[_689] >> 16u).x) + unpackHalf2x16(SharedMemory0[_709] >> 16u).x) + unpackHalf2x16(SharedMemory0[_729] >> 16u).x) + unpackHalf2x16(SharedMemory0[_749] >> 16u).x) + unpackHalf2x16(SharedMemory0[_769] >> 16u).x) + unpackHalf2x16(SharedMemory0[_789] >> 16u).x) + unpackHalf2x16(SharedMemory0[_809] >> 16u).x) + unpackHalf2x16(SharedMemory0[_829] >> 16u).x;
        uint _846 = gl_LocalInvocationIndex + 176u;
        uint _849 = 256u | _846;
        float _858 = ((((((((((unpackHalf2x16(SharedMemory0[gl_LocalInvocationIndex]).x + unpackHalf2x16(SharedMemory0[_646]).x) + unpackHalf2x16(SharedMemory0[_666]).x) + unpackHalf2x16(SharedMemory0[_686]).x) + unpackHalf2x16(SharedMemory0[_706]).x) + unpackHalf2x16(SharedMemory0[_726]).x) + unpackHalf2x16(SharedMemory0[_746]).x) + unpackHalf2x16(SharedMemory0[_766]).x) + unpackHalf2x16(SharedMemory0[_786]).x) + unpackHalf2x16(SharedMemory0[_806]).x) + unpackHalf2x16(SharedMemory0[_826]).x) + unpackHalf2x16(SharedMemory0[_846]).x;
        float _865 = ((((((((((unpackHalf2x16(SharedMemory0[_633]).x + unpackHalf2x16(SharedMemory0[_649]).x) + unpackHalf2x16(SharedMemory0[_669]).x) + unpackHalf2x16(SharedMemory0[_689]).x) + unpackHalf2x16(SharedMemory0[_709]).x) + unpackHalf2x16(SharedMemory0[_729]).x) + unpackHalf2x16(SharedMemory0[_749]).x) + unpackHalf2x16(SharedMemory0[_769]).x) + unpackHalf2x16(SharedMemory0[_789]).x) + unpackHalf2x16(SharedMemory0[_809]).x) + unpackHalf2x16(SharedMemory0[_829]).x) + unpackHalf2x16(SharedMemory0[_849]).x;
        uint _866 = gl_LocalInvocationIndex + 192u;
        uint _869 = 256u | _866;
        uint _886 = gl_LocalInvocationIndex + 208u;
        uint _889 = 256u | _886;
        uint _906 = gl_LocalInvocationIndex + 224u;
        uint _909 = 256u | _906;
        uint _926 = gl_LocalInvocationIndex + 240u;
        uint _929 = 256u | _926;
        vec3 _947 = vec3(((((_835 + unpackHalf2x16(SharedMemory0[_846] >> 16u).x) + unpackHalf2x16(SharedMemory0[_866] >> 16u).x) + unpackHalf2x16(SharedMemory0[_886] >> 16u).x) + unpackHalf2x16(SharedMemory0[_906] >> 16u).x) + unpackHalf2x16(SharedMemory0[_926] >> 16u).x, (((_858 + unpackHalf2x16(SharedMemory0[_866]).x) + unpackHalf2x16(SharedMemory0[_886]).x) + unpackHalf2x16(SharedMemory0[_906]).x) + unpackHalf2x16(SharedMemory0[_926]).x, ((((_842 + unpackHalf2x16(SharedMemory0[_849] >> 16u).x) + unpackHalf2x16(SharedMemory0[_869] >> 16u).x) + unpackHalf2x16(SharedMemory0[_889] >> 16u).x) + unpackHalf2x16(SharedMemory0[_909] >> 16u).x) + unpackHalf2x16(SharedMemory0[_929] >> 16u).x) * 0.0625;
        uvec2 _959 = (gl_WorkGroupID.xy * uvec2(4u)) + uvec2(gl_LocalInvocationIndex % 4u, (gl_LocalInvocationIndex >> 2u) % 4u);
        imageStore(IndirectDiffuseOutput, ivec2(_959), vec4(_947 * (1.0 / (1.0 - dot(_947, vec3(0.2126390039920806884765625, 0.715168654918670654296875, 0.072192318737506866455078125)))), 1.0));
        imageStore(AmbientOcclusionOutput, ivec2(_959), vec4(1.0 - (((((_865 + unpackHalf2x16(SharedMemory0[_869]).x) + unpackHalf2x16(SharedMemory0[_889]).x) + unpackHalf2x16(SharedMemory0[_909]).x) + unpackHalf2x16(SharedMemory0[_929]).x) * 0.0625)));
    }
}

