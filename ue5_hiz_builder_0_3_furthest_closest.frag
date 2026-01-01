#version 460
#extension GL_KHR_shader_subgroup_basic : require

// -----------------------------------------------------------------------------
// Readable version of the decompiled shader:
// - Builds hierarchical Z buffer (HZB) mip levels in a single dispatch.
// - For Reverse-Z depth convention (near=1, far=0) this typically means:
//   - "Furthest"  = min(depth)   (most far depth is smallest)
//   - "Closest"   = max(depth)   (most near depth is largest)
// - Writes 4 mip levels (0..3), each with Furthest + Closest targets.
// -----------------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Globals
{
    // xy: scale from output pixel coords to parent mip UV
    // zw: bias/offset in UV space
    vec4 dispatchThreadIdToUvScaleBias;

    // 1 / size of ParentTextureMip (texel size)
    vec2 parentInvSize;

    // maximum UV bound used for clamping (to avoid out-of-bounds gather)
    vec2 inputViewportMaxBound;
} g;

layout(set = 0, binding = 9)  uniform texture2D uParentMip;
layout(set = 0, binding = 10) uniform sampler   uParentMipSampler;

// r32f targets, but we store the value in .x (vec4(value,0,0,0))
layout(set = 0, binding = 1, r32f) uniform writeonly image2D uFurthestMip0;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D uFurthestMip1;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D uFurthestMip2;
layout(set = 0, binding = 4, r32f) uniform writeonly image2D uFurthestMip3;

layout(set = 0, binding = 5, r32f) uniform writeonly image2D uClosestMip0;
layout(set = 0, binding = 6, r32f) uniform writeonly image2D uClosestMip1;
layout(set = 0, binding = 7, r32f) uniform writeonly image2D uClosestMip2;
layout(set = 0, binding = 8, r32f) uniform writeonly image2D uClosestMip3;

// Shared arrays for workgroup reductions.
// Furthest is stored as float bits in uint to avoid any unintended conversions.
shared uint  sFurthestBits[64];
shared float sClosest[64];

// The decompiled shader uses:
//   unpackHalf2x16(packHalf2x16(vec2(maxZ, 0)) + 1u).x
// which effectively:
// - quantizes to fp16 grid
// - and nudges up by 1 ULP in the packed half representation
// This makes the "closest" (max) value conservative (it won't underestimate).
float conservativeMaxQuantizeToF16(float v)
{
    return unpackHalf2x16(packHalf2x16(vec2(v, 0.0)) + 1u).x;
}

// Converts gl_LocalInvocationIndex (0..63) into a swizzled 8x8 coordinate.
// This swizzle pattern often improves memory access / quad locality.
uvec2 laneToLocalCoord8x8(uint lane)
{
    uint x =
        ((4u & (lane << 2u)) | (2u & (lane >> 1u))) |
        (1u & (lane >> 4u));

    uint y =
        ((4u & (lane << 1u)) | (2u & (lane >> 2u))) |
        (1u & (lane >> 5u));

    return uvec2(x, y); // each is 0..7
}

void main()
{
    uint  lane          = gl_LocalInvocationIndex;     // 0..63 inside the 8x8 workgroup
    uvec2 groupBasePix  = gl_WorkGroupID.xy * uvec2(8u);
    uvec2 outPixMip0    = groupBasePix + laneToLocalCoord8x8(lane);

    // Build UV for parent mip gather:
    // - sample at pixel center (pix + 0.5)
    // - scale/bias into UV space
    // - -0.25 texel aligns textureGather with the intended 2x2 neighborhood
    vec2 uv =
        (vec2(outPixMip0) + vec2(0.5)) * g.dispatchThreadIdToUvScaleBias.xy +
        g.dispatchThreadIdToUvScaleBias.zw +
        vec2(-0.25) * g.parentInvSize;

    // Clamp so gather won't read outside input viewport.
    uv = min(uv, g.inputViewportMaxBound - g.parentInvSize);

    // Gather 2x2 quad of depth values.
    vec4 gathered = textureGather(sampler2D(uParentMip, uParentMipSampler), uv);

    float z0 = gathered.x;
    float z1 = gathered.y;
    float z2 = gathered.z;
    float z3 = gathered.w;

    // Reverse-Z HZB semantics (typical):
    // - furthest = min depth
    // - closest  = max depth
    float furthestZ = min(min(z0, min(z1, z2)), z3);
    float closestZ  = max(max(z0, max(z1, z2)), z3);

    // Mip 0: each lane writes one pixel.
    imageStore(uFurthestMip0, ivec2(outPixMip0), vec4(furthestZ, 0, 0, 0));
    imageStore(uClosestMip0,  ivec2(outPixMip0), vec4(conservativeMaxQuantizeToF16(closestZ), 0, 0, 0));

    // Seed shared memory for reductions.
    sFurthestBits[lane] = floatBitsToUint(furthestZ);
    sClosest[lane]      = closestZ;

    // Only needed when a subgroup doesn't cover all 64 lanes.
    if (64u > gl_SubgroupSize)
        barrier();

    // -------------------------------------------------------------------------
    // Reduce to mip 1 (8x8 -> 4x4)
    // 16 lanes: each reduces 4 lanes spaced by 16.
    uvec2 outPixMip1   = outPixMip0;
    float furthestMip1 = furthestZ;
    float closestMip1  = closestZ;

    if (lane < 16u)
    {
        uint lane1 = lane + 16u;
        uint lane2 = lane + 32u;
        uint lane3 = lane + 48u;

        float f1 = uintBitsToFloat(sFurthestBits[lane1]);
        float f2 = uintBitsToFloat(sFurthestBits[lane2]);
        float f3 = uintBitsToFloat(sFurthestBits[lane3]);

        float c1 = sClosest[lane1];
        float c2 = sClosest[lane2];
        float c3 = sClosest[lane3];

        furthestMip1 = min(min(furthestZ, min(f1, f2)), f3);
        closestMip1  = max(max(closestZ,  max(c1, c2)), c3);

        outPixMip1 = outPixMip0 >> uvec2(1u);

        imageStore(uFurthestMip1, ivec2(outPixMip1), vec4(furthestMip1, 0, 0, 0));
        imageStore(uClosestMip1,  ivec2(outPixMip1), vec4(conservativeMaxQuantizeToF16(closestMip1), 0, 0, 0));

        // Prepare for next reduction stage.
        sFurthestBits[lane] = floatBitsToUint(furthestMip1);
        sClosest[lane]      = closestMip1;
    }

    if (16u > gl_SubgroupSize)
        barrier();

    // -------------------------------------------------------------------------
    // Reduce to mip 2 (4x4 -> 2x2)
    // 4 lanes: each reduces 4 lanes spaced by 4.
    uvec2 outPixMip2   = outPixMip1;
    float furthestMip2 = furthestMip1;
    float closestMip2  = closestMip1;

    if (lane < 4u)
    {
        uint lane1 = lane + 4u;
        uint lane2 = lane + 8u;
        uint lane3 = lane + 12u;

        float f1 = uintBitsToFloat(sFurthestBits[lane1]);
        float f2 = uintBitsToFloat(sFurthestBits[lane2]);
        float f3 = uintBitsToFloat(sFurthestBits[lane3]);

        float c1 = sClosest[lane1];
        float c2 = sClosest[lane2];
        float c3 = sClosest[lane3];

        furthestMip2 = min(min(furthestMip1, min(f1, f2)), f3);
        closestMip2  = max(max(closestMip1,  max(c1, c2)), c3);

        outPixMip2 = outPixMip1 >> uvec2(1u);

        imageStore(uFurthestMip2, ivec2(outPixMip2), vec4(furthestMip2, 0, 0, 0));
        imageStore(uClosestMip2,  ivec2(outPixMip2), vec4(conservativeMaxQuantizeToF16(closestMip2), 0, 0, 0));

        // Prepare for final reduction stage.
        sFurthestBits[lane] = floatBitsToUint(furthestMip2);
        sClosest[lane]      = closestMip2;
    }

    if (4u > gl_SubgroupSize)
        barrier();

    // -------------------------------------------------------------------------
    // Reduce to mip 3 (2x2 -> 1x1)
    // lane 0 reduces lanes 0..3.
    if (lane == 0u)
    {
        float f1 = uintBitsToFloat(sFurthestBits[1u]);
        float f2 = uintBitsToFloat(sFurthestBits[2u]);
        float f3 = uintBitsToFloat(sFurthestBits[3u]);

        float c1 = sClosest[1u];
        float c2 = sClosest[2u];
        float c3 = sClosest[3u];

        float furthestMip3 = min(min(furthestMip2, min(f1, f2)), f3);
        float closestMip3  = max(max(closestMip2,  max(c1, c2)), c3);

        uvec2 outPixMip3 = outPixMip2 >> uvec2(1u);

        imageStore(uFurthestMip3, ivec2(outPixMip3), vec4(furthestMip3, 0, 0, 0));
        imageStore(uClosestMip3,  ivec2(outPixMip3), vec4(conservativeMaxQuantizeToF16(closestMip3), 0, 0, 0));
    }
}


