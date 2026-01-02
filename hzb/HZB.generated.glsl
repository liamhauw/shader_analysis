#version 460
#extension GL_KHR_shader_subgroup_basic : require

// -----------------------------------------------------------------------------
// HZB.generated.glsl
//
// Goal:
// - Provide a readable GLSL version of the exact code path used by the RenderDoc
//   decompile in `hzb/HZB.rdc.glsl`, cross-referenced to UE's `hzb/HZB.usf`.
//
// What this shader does:
// - Builds a Hierarchical Z-Buffer (HZB) pyramid for 4 mip levels in one dispatch:
//   - Mip 0: 8x8 outputs (one per thread) from 2x2 gather of the parent mip.
//   - Mip 1: 4x4 outputs from 2x2 reduction over mip 0.
//   - Mip 2: 2x2 outputs from 2x2 reduction over mip 1.
//   - Mip 3: 1x1 output  from 2x2 reduction over mip 2.
//
// Conventions (matching UE naming):
// - "Furthest" HZB stores the MIN of device Z (smaller Z = farther in UE's device Z
//   conventions for this build pass).
// - "Closest" HZB stores the MAX of device Z, conservatively rounded *up* to fp16.
//
// Exact path selection (derived from the decompile):
// - COMPUTESHADER path (`HZBBuildCS`)
// - DIM_MIP_LEVEL_COUNT = 4
// - DIM_FURTHEST = 1, DIM_CLOSEST = 1
// - VIS_BUFFER_FORMAT = 0 (no VisBuffer sampling/patch-up)
// - DIM_FROXELS = 0
// - Uses the "InitialTilePixelPositionForReduction2x2(MAX_MIP_BATCH_SIZE-1, idx)"
//   swizzle to make the reductions work with linear `GroupThreadIndex`.
// -----------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Globals
{
    // (DispatchThreadId + 0.5) * xy + zw  -> BufferUV
    vec4 DispatchThreadIdToBufferUV;

    // 1.0 / ParentTextureMip resolution (in UV space).
    vec2 InvSize;

    // Clamp to avoid sampling outside the viewport when view size is odd.
    vec2 InputViewportMaxBound;
} g;

// Parent mip depth texture (the mip level to downsample).
layout(set = 0, binding = 9) uniform texture2D ParentTextureMip;
layout(set = 0, binding = 10) uniform sampler ParentTextureMipSampler;

// Output mip chain (furthest = min, closest = max rounded up to fp16).
layout(set = 0, binding = 1, r32f) uniform writeonly image2D FurthestHZBOutput_0;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D FurthestHZBOutput_1;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D FurthestHZBOutput_2;
layout(set = 0, binding = 4, r32f) uniform writeonly image2D FurthestHZBOutput_3;

layout(set = 0, binding = 5, r32f) uniform writeonly image2D ClosestHZBOutput_0;
layout(set = 0, binding = 6, r32f) uniform writeonly image2D ClosestHZBOutput_1;
layout(set = 0, binding = 7, r32f) uniform writeonly image2D ClosestHZBOutput_2;
layout(set = 0, binding = 8, r32f) uniform writeonly image2D ClosestHZBOutput_3;

// Group-shared scratch for 2x2 reductions across mip levels.
shared uint  SharedMinDeviceZ[8u * 8u];
shared float SharedMaxDeviceZ[8u * 8u];

// UE: RoundUpF16(DeviceZ) = f16tof32(f32tof16(DeviceZ) + 1)
// GLSL equivalent (matches the RenderDoc decompile idiom exactly).
float roundUpF16(float x)
{
    // packHalf2x16 returns uint bits holding 2x fp16; add 1 to round up the first lane.
    uint h = packHalf2x16(vec2(x, 0.0));
    h += 1u;
    return unpackHalf2x16(h).x;
}

vec4 gather4RedFromBufferUV(vec2 bufferUV)
{
    vec2 uv = min(bufferUV + vec2(-0.25) * g.InvSize, g.InputViewportMaxBound - g.InvSize);
    return textureGather(sampler2D(ParentTextureMip, ParentTextureMipSampler), uv);
}

// This is the exact swizzle pattern seen in `HZB.rdc.glsl` for:
//   InitialTilePixelPositionForReduction2x2(MAX_MIP_BATCH_SIZE - 1, GroupThreadIndex)
// with GROUP_TILE_SIZE=8 and MAX_MIP_BATCH_SIZE=4.
//
// Why we keep it:
// - The reductions assume that the first 16 threads correspond to the 4x4 mip1
//   pixels, and that their 2x2 parents are laid out at predictable offsets in LDS.
// - A naive `uvec2(gl_LocalInvocationID.xy)` mapping breaks that assumption.
uvec2 initialTilePixelPosForReduction2x2(uint groupThreadIndex)
{
    uint idx = groupThreadIndex;

    // Bit-twiddle copied from the decompile (kept verbatim for correctness).
    uint x = ((4u & (idx << 2u)) | (2u & (idx >> 1u))) | (1u & (idx >> 4u));
    uint y = ((4u & (idx << 1u)) | (2u & (idx >> 2u))) | (1u & (idx >> 5u));
    return uvec2(x, y);
}

void storeMip(uint mip, ivec2 pos, float minDeviceZ, float maxDeviceZ)
{
    // Furthest = min
    // Closest  = max rounded up to fp16 (conservative).
    if (mip == 0u)
    {
        imageStore(FurthestHZBOutput_0, pos, vec4(minDeviceZ));
        imageStore(ClosestHZBOutput_0,  pos, vec4(roundUpF16(maxDeviceZ)));
    }
    else if (mip == 1u)
    {
        imageStore(FurthestHZBOutput_1, pos, vec4(minDeviceZ));
        imageStore(ClosestHZBOutput_1,  pos, vec4(roundUpF16(maxDeviceZ)));
    }
    else if (mip == 2u)
    {
        imageStore(FurthestHZBOutput_2, pos, vec4(minDeviceZ));
        imageStore(ClosestHZBOutput_2,  pos, vec4(roundUpF16(maxDeviceZ)));
    }
    else // mip == 3
    {
        imageStore(FurthestHZBOutput_3, pos, vec4(minDeviceZ));
        imageStore(ClosestHZBOutput_3,  pos, vec4(roundUpF16(maxDeviceZ)));
    }
}

void main()
{
    uint  groupThreadIndex = gl_LocalInvocationIndex;   // 0..63
    uvec2 groupId          = gl_WorkGroupID.xy;

    // Map 0..63 to an 8x8 pixel position in the group, using UE's reduction-friendly swizzle.
    uvec2 groupThreadId = initialTilePixelPosForReduction2x2(groupThreadIndex);

    // Output pixel coordinate in mip 0 (one thread writes one pixel here).
    uvec2 groupOffset        = 8u * groupId;
    uvec2 dispatchThreadId   = groupOffset + groupThreadId;
    ivec2 outputPixelPosI    = ivec2(dispatchThreadId);

    // Parent sampling UV.
    vec2 bufferUV =
        (vec2(dispatchThreadId) + vec2(0.5)) * g.DispatchThreadIdToBufferUV.xy +
        g.DispatchThreadIdToBufferUV.zw;

    // Gather the 2x2 neighborhood from the parent mip.
    vec4 deviceZ4 = gather4RedFromBufferUV(bufferUV);

    // Reduce within the gathered quad.
    float minDeviceZ = min(min(deviceZ4.x, deviceZ4.y), min(deviceZ4.z, deviceZ4.w));
    float maxDeviceZ = max(max(deviceZ4.x, deviceZ4.y), max(deviceZ4.z, deviceZ4.w));

    // Store mip 0 (always).
    storeMip(0u, outputPixelPosI, minDeviceZ, maxDeviceZ);

    // Seed LDS for further reductions.
    SharedMinDeviceZ[groupThreadIndex] = floatBitsToUint(minDeviceZ);
    SharedMaxDeviceZ[groupThreadIndex] = maxDeviceZ;

    // Build mip1..mip3, matching UE loop:
    //   TileSize       = GROUP_TILE_SIZE >> MipLevel
    //   ReduceBankSize = TileSize * TileSize
    //
    // Sync rule (matches `HZB.rdc.glsl` subgroup-size comparisons):
    //   If (ReduceBankSize * 4) > subgroupSize -> barrier()
    //
    // Intuition: each reduced pixel reads 4 values; if more than one subgroup wrote to
    // LDS, we need to sync before reading.
    uvec2 outPos = dispatchThreadId;
    for (uint mip = 1u; mip < 4u; ++mip)
    {
        uint tileSize       = 8u >> mip;         // 4, 2, 1
        uint reduceBankSize = tileSize * tileSize; // 16, 4, 1

        if ((reduceBankSize * 4u) > gl_SubgroupSize)
        {
            barrier();
        }

        if (groupThreadIndex < reduceBankSize)
        {
            // Load the four children for this parent pixel from LDS.
            // Layout is exactly as in UE:
            //   LDSIndex = GroupThreadIndex + i * ReduceBankSize, i=0..3
            float childMin0 = minDeviceZ;
            float childMax0 = maxDeviceZ;

            uint lds1 = groupThreadIndex + 1u * reduceBankSize;
            uint lds2 = groupThreadIndex + 2u * reduceBankSize;
            uint lds3 = groupThreadIndex + 3u * reduceBankSize;

            float childMin1 = uintBitsToFloat(SharedMinDeviceZ[lds1]);
            float childMin2 = uintBitsToFloat(SharedMinDeviceZ[lds2]);
            float childMin3 = uintBitsToFloat(SharedMinDeviceZ[lds3]);

            float childMax1 = SharedMaxDeviceZ[lds1];
            float childMax2 = SharedMaxDeviceZ[lds2];
            float childMax3 = SharedMaxDeviceZ[lds3];

            minDeviceZ = min(min(childMin0, childMin1), min(childMin2, childMin3));
            maxDeviceZ = max(max(childMax0, childMax1), max(childMax2, childMax3));

            // Next mip pixel coordinate is halved (>>1).
            outPos >>= uvec2(1u);

            storeMip(mip, ivec2(outPos), minDeviceZ, maxDeviceZ);

            // Write back reduced values for the next loop iteration.
            SharedMinDeviceZ[groupThreadIndex] = floatBitsToUint(minDeviceZ);
            SharedMaxDeviceZ[groupThreadIndex] = maxDeviceZ;
        }
    }
}


