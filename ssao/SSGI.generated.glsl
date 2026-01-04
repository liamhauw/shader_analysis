#version 460
#extension GL_EXT_control_flow_attributes : require
#define SPIRV_CROSS_FLATTEN [[flatten]]
#define SPIRV_CROSS_BRANCH [[dont_flatten]]
#define SPIRV_CROSS_UNROLL [[unroll]]
#define SPIRV_CROSS_LOOP [[dont_unroll]]
#extension GL_EXT_spirv_intrinsics : require

// Screen Space Global Illumination (SSGI) Compute Shader
// This shader performs screen-space ray tracing to compute indirect diffuse lighting and ambient occlusion
// Configuration: 16 rays per pixel, 8 steps per ray, 4x4 pixel tiles

layout(local_size_x = 4, local_size_y = 4, local_size_z = 16) in;

// Shared memory for inter-thread communication within a workgroup
// Stores compressed normals, depth values, and ray hit results
shared uint SharedMemory0[512];

// Uniform buffers
layout(set = 0, binding = 1, std140) uniform ViewBuffer
{
    layout(offset = 192) mat4 View_TranslatedWorldToView;
    layout(offset = 448) mat4 View_ViewToClip;
    layout(offset = 1264) vec4 View_ScreenPositionScaleBias;
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;
    layout(offset = 2648) uint View_StateFrameIndexMod8;
} View;

layout(set = 0, binding = 0, std140) uniform GlobalsBuffer
{
    layout(offset = 16) vec4 HZBUvFactorAndInvFactor;
    layout(offset = 32) vec4 ColorBufferScaleBias;
    layout(offset = 48) vec2 ReducedColorUVMax;
    layout(offset = 56) float PixelPositionToFullResPixel;
    layout(offset = 64) vec2 FullResPixelOffset;
} Globals;

// Texture samplers and images
layout(set = 0, binding = 9) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 0, binding = 4) uniform texture2D SceneDepthTexture;
layout(set = 0, binding = 5) uniform texture2D GBufferATexture;  // Normal and roughness
layout(set = 0, binding = 6) uniform texture2D GBufferBTexture;  // Additional GBuffer data
layout(set = 0, binding = 7) uniform texture2D FurthestHZBTexture;  // Hierarchical Z-Buffer
layout(set = 0, binding = 8) uniform texture2D ColorTexture;  // Scene color for indirect lighting
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D IndirectDiffuseOutput;
layout(set = 0, binding = 3, r32f) uniform writeonly image2D AmbientOcclusionOutput;

// Helper functions for min/max operations
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
    // Thread identification within the workgroup
    // Each workgroup processes a 4x4 tile of pixels with 16 rays per pixel
    uint rayIndex = gl_LocalInvocationIndex % 16u;  // Which ray (0-15)
    uint waveIndex = gl_LocalInvocationIndex / 16u;  // Which wave/lane (0-15)
    
    // Phase 1: Load GBuffer data into shared memory
    // Only the first wave (rayIndex == 0) loads GBuffer data for each pixel
    SPIRV_CROSS_BRANCH
    if (waveIndex == 0u)
    {
        // Calculate pixel position within the tile
        uint pixelX = rayIndex % 4u;
        uint pixelY = (rayIndex >> 2u) % 4u;
        uvec2 pixelOffset = uvec2(pixelX, pixelY);
        
        // Calculate full-resolution pixel position
        uvec2 tileBase = gl_WorkGroupID.xy * uvec2(4u);
        uvec2 pixelPos = tileBase + pixelOffset;
        
        // Convert to UV coordinates
        vec2 fullResPixelPos = (vec2(pixelPos) * Globals.PixelPositionToFullResPixel) + Globals.FullResPixelOffset;
        vec2 bufferUV = fullResPixelPos * View.View_BufferSizeAndInvSize.zw;
        
        // Sample GBuffer A (normal vector)
        vec3 worldNormal = (textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).xyz * 2.0) - vec3(1.0);
        
        // Transform normal to view space and compress to 8-bit per channel
        vec3 viewNormal = normalize((View.View_TranslatedWorldToView * vec4(worldNormal, 0.0)).xyz);
        uvec3 compressedNormal = uvec3(clamp((viewNormal * 0.5 + vec3(0.5), vec3(0.0), vec3(1.0)) * 255.0));
        
        // Pack normal into single uint: R in bits 0-7, G in bits 8-15, B in bits 16-23
        SharedMemory0[rayIndex] = (compressedNormal.x | (compressedNormal.y << 8u)) | (compressedNormal.z << 16u);
        
        // Sample GBuffer B to check if pixel is valid (not unlit)
        float gbufferB_w = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).w;
        uint shadingModelId = uint((gbufferB_w * 255.0) + 0.5) & 15u;
        
        // Sample scene depth if pixel is valid (not unlit), otherwise store -1
        float deviceZ = (shadingModelId != 0u) 
            ? textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).x 
            : (-1.0);
        
        SharedMemory0[16u | rayIndex] = floatBitsToUint(deviceZ);
    }
    else if ((gl_LocalInvocationIndex / 64u) == 1u)
    {
        // Clear bins (unused in this path)
        SharedMemory0[32u | rayIndex] = 0u;
    }
    
    // Synchronize all threads before proceeding
    barrier();
    
    // Phase 2: Read GBuffer data from shared memory
    uint compressedNormal = SharedMemory0[rayIndex];
    float deviceZ = uintBitsToFloat(SharedMemory0[16u | rayIndex]);
    
    barrier();
    
    // Phase 3: Ray casting
    uvec2 rayResult = uvec2(0u);  // Packed color and AO result
    
    SPIRV_CROSS_BRANCH
    if (deviceZ > 0.0)  // Only trace rays for valid pixels
    {
        // Decompress normal from view space
        uvec3 normalBytes = uvec3(
            compressedNormal & 255u,
            (compressedNormal >> 8u) & 255u,
            (compressedNormal >> 16u) & 255u
        );
        vec3 viewNormal = (vec3(normalBytes) * 0.007843137718737125396728515625) - vec3(1.0);  // /255.0 * 2.0 - 1.0
        
        // Calculate pixel position and screen coordinates
        uint pixelX = rayIndex % 4u;
        uint pixelY = (rayIndex >> 2u) % 4u;
        uvec2 pixelPos = (gl_WorkGroupID.xy * uvec2(4u)) + uvec2(pixelX, pixelY);
        vec2 fullResPixelPos = (vec2(pixelPos) * Globals.PixelPositionToFullResPixel) + Globals.FullResPixelOffset;
        vec2 viewportUV = fullResPixelPos * View.View_ViewSizeAndInvSize.zw;
        
        // Convert to screen space coordinates [-1, 1]
        float screenX = (2.0 * viewportUV.x) - 1.0;
        float screenY = 1.0 - (2.0 * viewportUV.y);
        
        // Generate random seed for this pixel using PCG hash
        uvec3 seed = (uvec3(ivec3(int(pixelPos.x), int(pixelPos.y), int(View.View_StateFrameIndexMod8))) * uvec3(1664525u)) + uvec3(1013904223u);
        uint hash1 = seed.y;
        uint hash2 = seed.z;
        uint hash3 = seed.x + (hash1 * hash2);
        uint hash4 = hash1 + (hash2 * hash3);
        uint hash5 = hash2 + (hash3 * hash4);
        uint hash6 = hash3 + (hash4 * hash5);
        uvec3 random = uvec3(hash6, hash4 + (hash5 * hash6), 0u) >> uvec3(16u);
        
        // Build tangent basis from normal for cosine-weighted hemisphere sampling
        float nz = viewNormal.z;
        float nzSign = float((nz >= 0.0) ? 1 : (-1));
        float invNz = (-1.0) / (nzSign + nz);
        float nx = viewNormal.x;
        float ny = viewNormal.y;
        float nxy = (nx * ny) * invNz;
        
        // Generate Hammersley sequence sample for this ray
        float hammersleyU = fract((float(waveIndex) * 0.0625) + (float(random.x) * 1.52587890625e-05));  // /65536.0
        float hammersleyV = float((bitfieldReverse(waveIndex) >> 16u) ^ random.y) * 1.52587890625e-05;
        vec2 hammersleySample = (vec2(hammersleyU, hammersleyV) * 2.0) - vec2(0.999999940395355224609375);
        
        // Cosine-weighted hemisphere sampling in tangent space
        vec2 absSample = abs(hammersleySample);
        float maxAbs = spvNMax(absSample.x, absSample.y);
        float angle = 0.785398185253143310546875 * ((spvNMin(absSample.x, absSample.y) / (maxAbs + 5.4210108624275221700372640043497e-20)) + (2.0 * float(absSample.y >= absSample.x)));
        
        // Transform sample to world space using tangent basis
        vec2 cosSin = vec2(cos(angle), sin(angle));
        vec2 signPreservedCosSin = vec2(
            uintBitsToFloat((floatBitsToUint(cosSin) & uvec2(2147483647u)) | (floatBitsToFloat(hammersleySample) & uvec2(2147483648u)))
        );
        vec3 tangentSample = vec3(signPreservedCosSin * maxAbs, sqrt(1.0 - (maxAbs * maxAbs)));
        
        // Apply tangent basis transformation
        mat3 tangentBasis = mat3(
            vec3(1.0 + ((nzSign * invNz) * (nx * nx)), nzSign * nxy, (-nzSign) * nx),
            vec3(nxy, nzSign + (invNz * (ny * ny)), -ny),
            viewNormal
        );
        vec3 viewRayDir = tangentBasis * tangentSample;
        
        // Initialize ray in screen space
        vec3 rayStartScreen = vec3(screenX, screenY, deviceZ);
        
        // Project ray direction to screen space
        vec4 rayEndClip = vec4(
            viewRayDir.xy * vec2(View.View_ViewToClip[0u].x, View.View_ViewToClip[1u].y),
            viewRayDir.z * View.View_ViewToClip[2u].z,
            viewRayDir.z
        ) + vec4(screenX, screenY, deviceZ, 1.0);
        
        vec3 rayEndScreen = (rayEndClip.xyz * (1.0 / rayEndClip.w)) - rayStartScreen;
        
        // Clip ray to screen bounds
        vec2 rayStartXY = rayStartScreen.xy;
        vec2 rayStepXY = rayEndScreen.xy;
        float rayStepLength = 0.5 * length(rayStepXY);
        vec2 clipFactor = vec2(1.0) - (spvNMax(abs(rayStepXY + (rayStartXY * rayStepLength)) - vec2(rayStepLength), vec2(0.0)) / abs(rayStepXY));
        vec3 clippedRayStep = rayEndScreen * (spvNMin(clipFactor.x, clipFactor.y) / rayStepLength);
        
        // Calculate depth tolerance for ray intersection
        float depthTolerance = spvNMax(
            abs(clippedRayStep.z),
            (deviceZ - ((rayStartScreen + (View.View_ViewToClip * vec4(0.0, 0.0, 1.0, 0.0)).xyz) * 0.5).z) * 2.0
        ) * 0.125;
        
        // Convert to HZB UV space
        vec3 rayStartUVz = vec3(
            (rayStartScreen.xy * vec2(0.5, -0.5) + vec2(0.5)) * Globals.HZBUvFactorAndInvFactor.xy,
            rayStartScreen.z
        );
        vec3 rayStepUVz = vec3(
            (clippedRayStep.xy * vec2(0.5, -0.5)) * Globals.HZBUvFactorAndInvFactor.xy,
            clippedRayStep.z
        ) * 0.125;
        
        // Add interleaved gradient noise offset for temporal jittering
        float stepOffset = fract(52.98291778564453125 * fract(dot(
            (vec2(pixelPos) + vec2(0.5)) + (vec2(32.66500091552734375, 11.81499958038330078125) * float(View.View_StateFrameIndexMod8)),
            vec2(0.067110560834407806396484375, 0.005837149918079376220703125)
        ))) - 0.89999997615814208984375;
        
        vec3 rayUVz = rayStartUVz + rayStepUVz * stepOffset;
        
        // Hierarchical Z-Buffer traversal (8 steps, processing 4 samples per iteration)
        // The loop processes samples in batches of 4 for efficiency
        bool foundHit = false;
        float hitLevel = 1.0;
        uint stepIndex = 0u;
        float currentLevel = 1.0;
        bvec4 finalHits = bvec4(false);
        
        SPIRV_CROSS_LOOP
        for (;;)
        {
            if (stepIndex < 8u)
            {
                // Sample 4 depth values at once (vectorized for efficiency)
                vec2 sampleUV = rayUVz.xy;
                vec2 stepUV = rayStepUVz.xy;
                float sampleZ = rayUVz.z;
                float stepZ = rayStepUVz.z;
                
                // Calculate step offsets for this batch (1, 2, 3, 4)
                float step0 = float(stepIndex) + 1.0;
                float step1 = float(stepIndex) + 2.0;
                float step2 = float(stepIndex) + 3.0;
                float step3 = float(stepIndex) + 4.0;
                
                // Mip levels increase as we step further along the ray
                float level0 = currentLevel + 1.0;
                float level1 = currentLevel + 2.0;
                
                // Sample HZB at 4 positions along the ray
                vec4 hzbDepths = vec4(
                    textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), sampleUV + (stepUV * step0), currentLevel).x,
                    textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), sampleUV + (stepUV * step1), currentLevel).x,
                    textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), sampleUV + (stepUV * step2), level0).x,
                    textureLod(sampler2D(FurthestHZBTexture, VulkanGlobalPointClampedSampler), sampleUV + (stepUV * step3), level0).x
                );
                
                // Calculate expected ray depths at sample positions
                vec4 rayDepths = vec4(
                    sampleZ + (step0 * stepZ),
                    sampleZ + (step1 * stepZ),
                    sampleZ + (step2 * stepZ),
                    sampleZ + (step3 * stepZ)
                );
                
                // Check for intersections: ray depth must be within tolerance of scene depth
                vec4 depthDiff = rayDepths - hzbDepths;
                bvec4 withinTolerance = lessThan(abs(depthDiff + vec4(depthTolerance)), vec4(depthTolerance));
                bvec4 validDepth = notEqual(hzbDepths, vec4(0.0));  // Don't count far plane as hit
                
                // A hit occurs when depth is within tolerance AND not at far plane
                bool hit0 = withinTolerance.x && validDepth.x;
                bool hit1 = withinTolerance.y && validDepth.y;
                bool hit2 = withinTolerance.z && validDepth.z;
                bool hit3 = withinTolerance.w && validDepth.w;
                
                bvec4 batchHits = bvec4(hit0, hit1, hit2, hit3);
                
                // Check if any sample in this batch hit (or if we already found a hit)
                foundHit = foundHit || hit0 || hit1 || hit2 || hit3;
                
                SPIRV_CROSS_BRANCH
                if (foundHit)
                {
                    // Store hit information and exit loop
                    finalHits = batchHits;
                    hitLevel = level1;
                    break;
                }
                
                // Continue to next batch
                currentLevel = level1;
                stepIndex += 4u;
            }
            else
            {
                break;
            }
        }
        
        // Phase 4: Sample color if hit found
        SPIRV_CROSS_BRANCH
        if (foundHit)
        {
            // Find exact hit position using the closest hit
            float hitStep = float(stepIndex) + spvNMin(
                spvNMin(finalHits.x ? 1.0 : 5.0, spvNMin(finalHits.y ? 2.0 : 5.0, finalHits.z ? 3.0 : 5.0)),
                finalHits.w ? 4.0 : 5.0
            );
            vec3 hitUVz = rayUVz + (rayStepUVz * hitStep);
            
            // Convert HZB UV back to color buffer UV
            vec2 hitColorUV = (((
                ((hitUVz.xy * Globals.HZBUvFactorAndInvFactor.zw) * vec2(2.0, -2.0)) + vec2(-1.0, 1.0)
            ) * View.View_ScreenPositionScaleBias.xy) + View.View_ScreenPositionScaleBias.wz) * Globals.ColorBufferScaleBias.xy + Globals.ColorBufferScaleBias.zw;
            hitColorUV = spvNMin(hitColorUV, Globals.ReducedColorUVMax);
            
            // Sample color texture
            vec3 sampleColor = textureLod(sampler2D(ColorTexture, VulkanGlobalPointClampedSampler), hitColorUV, hitLevel).xyz;
            
            // Apply Karis weighting (1 / (1 + Luminance)) for HDR
            vec3 luminanceWeights = vec3(0.2126390039920806884765625, 0.715168654918670654296875, 0.072192318737506866455078125);
            float luminance = dot(sampleColor, luminanceWeights);
            vec3 weightedColor = sampleColor * (1.0 / (1.0 + luminance));
            
            // Pack result: RGB in first uint, AO (1.0) in second uint
            rayResult = uvec2(
                (packHalf2x16(vec2(weightedColor.x, 0.0)) << 16u) | packHalf2x16(vec2(weightedColor.y, 0.0)),
                (packHalf2x16(vec2(weightedColor.z, 0.0)) << 16u) | packHalf2x16(vec2(1.0, 0.0))
            );
        }
        else
        {
            rayResult = uvec2(0u);
        }
    }
    else
    {
        rayResult = uvec2(0u);
    }
    
    // Phase 5: Store ray results in shared memory
    uint resultIndex = rayIndex + (waveIndex * 16u);
    SharedMemory0[resultIndex] = rayResult.x;
    SharedMemory0[256u | resultIndex] = rayResult.y;
    
    barrier();
    
    // Phase 6: Accumulate results from all rays (only first 16 threads, one per pixel)
    // Each thread accumulates results from all 16 rays for its pixel
    SPIRV_CROSS_BRANCH
    if (gl_LocalInvocationIndex < 16u)
    {
        // Accumulate RGB and AO from all 16 rays for this pixel
        // Results are stored in shared memory: [pixelIndex + rayIndex * 16]
        float sumR = 0.0;
        float sumG = 0.0;
        float sumB = 0.0;
        float sumAO = 0.0;
        
        // Sum first 12 rays (unrolled for performance)
        uint baseIndex = gl_LocalInvocationIndex;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 16u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 32u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 48u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 64u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 80u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 96u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 112u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 128u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 144u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 160u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 176u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        // Sum remaining 4 rays
        baseIndex = gl_LocalInvocationIndex + 192u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 208u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 224u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        baseIndex = gl_LocalInvocationIndex + 240u;
        sumR += unpackHalf2x16(SharedMemory0[baseIndex] >> 16u).x;
        sumG += unpackHalf2x16(SharedMemory0[baseIndex]).x;
        sumB += unpackHalf2x16(SharedMemory0[256u | baseIndex] >> 16u).x;
        sumAO += unpackHalf2x16(SharedMemory0[256u | baseIndex]).x;
        
        // Average over number of rays (16 rays total)
        vec3 accumulatedColor = vec3(sumR, sumG, sumB) * 0.0625;  // 1.0 / 16.0
        float accumulatedAO = sumAO * 0.0625;
        
        // Undo Karis weighting: multiply by (1 / (1 - Luminance))
        vec3 luminanceWeights = vec3(0.2126390039920806884765625, 0.715168654918670654296875, 0.072192318737506866455078125);
        float luminance = dot(accumulatedColor, luminanceWeights);
        accumulatedColor *= (1.0 / (1.0 - luminance));
        
        // Convert AO: 1.0 means fully occluded, 0.0 means fully lit
        accumulatedAO = 1.0 - accumulatedAO;
        
        // Calculate output pixel position
        uint pixelX = gl_LocalInvocationIndex % 4u;
        uint pixelY = (gl_LocalInvocationIndex >> 2u) % 4u;
        uvec2 outputPixel = (gl_WorkGroupID.xy * uvec2(4u)) + uvec2(pixelX, pixelY);
        
        // Write outputs
        imageStore(IndirectDiffuseOutput, ivec2(outputPixel), vec4(accumulatedColor, 1.0));
        imageStore(AmbientOcclusionOutput, ivec2(outputPixel), vec4(accumulatedAO));
    }
}

