#version 460
#extension GL_EXT_spirv_intrinsics : require
#extension GL_KHR_shader_subgroup_basic : require
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
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform type_Globals
{
    vec4 DispatchThreadIdToBufferUV;
    vec2 InvSize;
    vec2 InputViewportMaxBound;
} _Globals;

layout(set = 0, binding = 5) uniform texture2D ParentTextureMip;
layout(set = 0, binding = 6) uniform sampler ParentTextureMipSampler;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D FurthestHZBOutput_0;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D FurthestHZBOutput_1;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D FurthestHZBOutput_2;
layout(set = 0, binding = 4, r32f) uniform writeonly image2D FurthestHZBOutput_3;

shared uint SharedMinDeviceZ[64];

spirv_instruction(set = "GLSL.std.450", id = 79) float spvNMin(float, float);
spirv_instruction(set = "GLSL.std.450", id = 79) vec2 spvNMin(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 spvNMin(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 79) vec4 spvNMin(vec4, vec4);

void main()
{
    uvec2 _83 = (uvec2(8u) * gl_WorkGroupID.xy) + uvec2(((4u & (gl_LocalInvocationIndex << 2u)) | (2u & (gl_LocalInvocationIndex >> 1u))) | (1u & (gl_LocalInvocationIndex >> 4u)), ((4u & (gl_LocalInvocationIndex << 1u)) | (2u & (gl_LocalInvocationIndex >> 2u))) | (1u & (gl_LocalInvocationIndex >> 5u)));
    vec4 _103 = textureGather(sampler2D(ParentTextureMip, ParentTextureMipSampler), spvNMin((((vec2(_83) + vec2(0.5)) * _Globals.DispatchThreadIdToBufferUV.xy) + _Globals.DispatchThreadIdToBufferUV.zw) + (vec2(-0.25) * _Globals.InvSize), _Globals.InputViewportMaxBound - _Globals.InvSize));
    float _110 = spvNMin(spvNMin(_103.x, spvNMin(_103.y, _103.z)), _103.w);
    imageStore(FurthestHZBOutput_0, ivec2(_83), vec4(_110));
    SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_110);
    if (64u > gl_SubgroupSize)
    {
        barrier();
    }
    uvec2 _139;
    float _140;
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 16u)
    {
        uint _123 = SharedMinDeviceZ[gl_LocalInvocationIndex + 16u];
        uint _127 = SharedMinDeviceZ[gl_LocalInvocationIndex + 32u];
        uint _131 = SharedMinDeviceZ[gl_LocalInvocationIndex + 48u];
        float _135 = spvNMin(spvNMin(_110, spvNMin(uintBitsToFloat(_123), uintBitsToFloat(_127))), uintBitsToFloat(_131));
        uvec2 _136 = _83 >> uvec2(1u);
        imageStore(FurthestHZBOutput_1, ivec2(_136), vec4(_135));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_135);
        _139 = _136;
        _140 = _135;
    }
    else
    {
        _139 = _83;
        _140 = _110;
    }
    if (16u > gl_SubgroupSize)
    {
        barrier();
    }
    uvec2 _165;
    float _166;
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 4u)
    {
        uint _149 = SharedMinDeviceZ[gl_LocalInvocationIndex + 4u];
        uint _153 = SharedMinDeviceZ[gl_LocalInvocationIndex + 8u];
        uint _157 = SharedMinDeviceZ[gl_LocalInvocationIndex + 12u];
        float _161 = spvNMin(spvNMin(_140, spvNMin(uintBitsToFloat(_149), uintBitsToFloat(_153))), uintBitsToFloat(_157));
        uvec2 _162 = _139 >> uvec2(1u);
        imageStore(FurthestHZBOutput_2, ivec2(_162), vec4(_161));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_161);
        _165 = _162;
        _166 = _161;
    }
    else
    {
        _165 = _139;
        _166 = _140;
    }
    if (4u > gl_SubgroupSize)
    {
        barrier();
    }
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 1u)
    {
        float _187 = spvNMin(spvNMin(_166, spvNMin(uintBitsToFloat(SharedMinDeviceZ[gl_LocalInvocationIndex + 1u]), uintBitsToFloat(SharedMinDeviceZ[gl_LocalInvocationIndex + 2u]))), uintBitsToFloat(SharedMinDeviceZ[gl_LocalInvocationIndex + 3u]));
        imageStore(FurthestHZBOutput_3, ivec2(_165 >> uvec2(1u)), vec4(_187));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_187);
    }
}

