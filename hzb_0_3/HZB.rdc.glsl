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

layout(set = 0, binding = 9) uniform texture2D ParentTextureMip;
layout(set = 0, binding = 10) uniform sampler ParentTextureMipSampler;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D FurthestHZBOutput_0;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D FurthestHZBOutput_1;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D FurthestHZBOutput_2;
layout(set = 0, binding = 4, r32f) uniform writeonly image2D FurthestHZBOutput_3;
layout(set = 0, binding = 5, r32f) uniform writeonly image2D ClosestHZBOutput_0;
layout(set = 0, binding = 6, r32f) uniform writeonly image2D ClosestHZBOutput_1;
layout(set = 0, binding = 7, r32f) uniform writeonly image2D ClosestHZBOutput_2;
layout(set = 0, binding = 8, r32f) uniform writeonly image2D ClosestHZBOutput_3;

shared uint SharedMinDeviceZ[64];
shared float SharedMaxDeviceZ[64];

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
    uvec2 _92 = (uvec2(8u) * gl_WorkGroupID.xy) + uvec2(((4u & (gl_LocalInvocationIndex << 2u)) | (2u & (gl_LocalInvocationIndex >> 1u))) | (1u & (gl_LocalInvocationIndex >> 4u)), ((4u & (gl_LocalInvocationIndex << 1u)) | (2u & (gl_LocalInvocationIndex >> 2u))) | (1u & (gl_LocalInvocationIndex >> 5u)));
    vec4 _112 = textureGather(sampler2D(ParentTextureMip, ParentTextureMipSampler), spvNMin((((vec2(_92) + vec2(0.5)) * _Globals.DispatchThreadIdToBufferUV.xy) + _Globals.DispatchThreadIdToBufferUV.zw) + (vec2(-0.25) * _Globals.InvSize), _Globals.InputViewportMaxBound - _Globals.InvSize));
    float _113 = _112.x;
    float _114 = _112.y;
    float _115 = _112.z;
    float _118 = _112.w;
    float _119 = spvNMin(spvNMin(_113, spvNMin(_114, _115)), _118);
    float _122 = spvNMax(spvNMax(_113, spvNMax(_114, _115)), _118);
    imageStore(FurthestHZBOutput_0, ivec2(_92), vec4(_119));
    imageStore(ClosestHZBOutput_0, ivec2(_92), vec4(unpackHalf2x16(packHalf2x16(vec2(_122, 0.0)) + 1u).x));
    SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_119);
    SharedMaxDeviceZ[gl_LocalInvocationIndex] = _122;
    if (64u > gl_SubgroupSize)
    {
        barrier();
    }
    uvec2 _173;
    float _174;
    float _175;
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 16u)
    {
        uint _140 = gl_LocalInvocationIndex + 16u;
        uint _142 = SharedMinDeviceZ[_140];
        float _145 = SharedMaxDeviceZ[_140];
        uint _146 = gl_LocalInvocationIndex + 32u;
        uint _148 = SharedMinDeviceZ[_146];
        float _151 = SharedMaxDeviceZ[_146];
        uint _152 = gl_LocalInvocationIndex + 48u;
        uint _154 = SharedMinDeviceZ[_152];
        float _157 = SharedMaxDeviceZ[_152];
        float _160 = spvNMin(spvNMin(_119, spvNMin(uintBitsToFloat(_142), uintBitsToFloat(_148))), uintBitsToFloat(_154));
        float _163 = spvNMax(spvNMax(_122, spvNMax(_145, _151)), _157);
        uvec2 _164 = _92 >> uvec2(1u);
        imageStore(FurthestHZBOutput_1, ivec2(_164), vec4(_160));
        imageStore(ClosestHZBOutput_1, ivec2(_164), vec4(unpackHalf2x16(packHalf2x16(vec2(_163, 0.0)) + 1u).x));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_160);
        SharedMaxDeviceZ[gl_LocalInvocationIndex] = _163;
        _173 = _164;
        _174 = _163;
        _175 = _160;
    }
    else
    {
        _173 = _92;
        _174 = _122;
        _175 = _119;
    }
    if (16u > gl_SubgroupSize)
    {
        barrier();
    }
    uvec2 _215;
    float _216;
    float _217;
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 4u)
    {
        uint _182 = gl_LocalInvocationIndex + 4u;
        uint _184 = SharedMinDeviceZ[_182];
        float _187 = SharedMaxDeviceZ[_182];
        uint _188 = gl_LocalInvocationIndex + 8u;
        uint _190 = SharedMinDeviceZ[_188];
        float _193 = SharedMaxDeviceZ[_188];
        uint _194 = gl_LocalInvocationIndex + 12u;
        uint _196 = SharedMinDeviceZ[_194];
        float _199 = SharedMaxDeviceZ[_194];
        float _202 = spvNMin(spvNMin(_175, spvNMin(uintBitsToFloat(_184), uintBitsToFloat(_190))), uintBitsToFloat(_196));
        float _205 = spvNMax(spvNMax(_174, spvNMax(_187, _193)), _199);
        uvec2 _206 = _173 >> uvec2(1u);
        imageStore(FurthestHZBOutput_2, ivec2(_206), vec4(_202));
        imageStore(ClosestHZBOutput_2, ivec2(_206), vec4(unpackHalf2x16(packHalf2x16(vec2(_205, 0.0)) + 1u).x));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_202);
        SharedMaxDeviceZ[gl_LocalInvocationIndex] = _205;
        _215 = _206;
        _216 = _205;
        _217 = _202;
    }
    else
    {
        _215 = _173;
        _216 = _174;
        _217 = _175;
    }
    if (4u > gl_SubgroupSize)
    {
        barrier();
    }
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 1u)
    {
        uint _224 = gl_LocalInvocationIndex + 1u;
        uint _230 = gl_LocalInvocationIndex + 2u;
        uint _236 = gl_LocalInvocationIndex + 3u;
        float _244 = spvNMin(spvNMin(_217, spvNMin(uintBitsToFloat(SharedMinDeviceZ[_224]), uintBitsToFloat(SharedMinDeviceZ[_230]))), uintBitsToFloat(SharedMinDeviceZ[_236]));
        float _247 = spvNMax(spvNMax(_216, spvNMax(SharedMaxDeviceZ[_224], SharedMaxDeviceZ[_230])), SharedMaxDeviceZ[_236]);
        uvec2 _248 = _215 >> uvec2(1u);
        imageStore(FurthestHZBOutput_3, ivec2(_248), vec4(_244));
        imageStore(ClosestHZBOutput_3, ivec2(_248), vec4(unpackHalf2x16(packHalf2x16(vec2(_247, 0.0)) + 1u).x));
        SharedMinDeviceZ[gl_LocalInvocationIndex] = floatBitsToUint(_244);
        SharedMaxDeviceZ[gl_LocalInvocationIndex] = _247;
    }
}

