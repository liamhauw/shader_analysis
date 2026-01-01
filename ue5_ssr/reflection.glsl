#version 460
// -----------------------------------------------------------------------------
// Readable version (manually cleaned) of a decompiled fragment shader.
//
// High-level purpose:
//   Screen Space Reflection (SSR) ray march using a Hierarchical Z-Buffer (HZB)
//   to find an intersection in screen space, then reproject to previous frame
//   (using velocity or clip transform) to fetch history scene color.
//
// Notes about naming:
//   - Names are inferred from usage and common Unreal/SSR patterns.
//   - Some GBuffer channel semantics depend on engine/material layout; where
//     uncertain, comments indicate "likely".
// -----------------------------------------------------------------------------

#if defined(GL_EXT_control_flow_attributes)
#extension GL_EXT_control_flow_attributes : require
#define SPIRV_CROSS_FLATTEN [[flatten]]
#define SPIRV_CROSS_BRANCH  [[dont_flatten]]
#define SPIRV_CROSS_UNROLL  [[unroll]]
#define SPIRV_CROSS_LOOP    [[dont_unroll]]
#else
#define SPIRV_CROSS_FLATTEN
#define SPIRV_CROSS_BRANCH
#define SPIRV_CROSS_UNROLL
#define SPIRV_CROSS_LOOP
#endif

#extension GL_EXT_spirv_intrinsics : require

// SPIR-V std.450 NaN-safe min/max used by the decompiler.
spirv_instruction(set = "GLSL.std.450", id = 79) float spvNMin(float, float);
spirv_instruction(set = "GLSL.std.450", id = 79) vec2  spvNMin(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3  spvNMin(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 79) vec4  spvNMin(vec4, vec4);
spirv_instruction(set = "GLSL.std.450", id = 80) float spvNMax(float, float);
spirv_instruction(set = "GLSL.std.450", id = 80) vec2  spvNMax(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3  spvNMax(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 80) vec4  spvNMax(vec4, vec4);

// -----------------------------------------------------------------------------
// Uniform buffers (original names preserved, members renamed where appropriate)
// -----------------------------------------------------------------------------

layout(set = 1, binding = 1, std140) uniform type_View
{
    // Translated world-space to clip-space (current frame)
    layout(offset = 0)   mat4 View_TranslatedWorldToClip;
    // Translated world-space to view-space
    layout(offset = 192) mat4 View_TranslatedWorldToView;
    // View-space to clip-space (projection)
    layout(offset = 448) mat4 View_ViewToClip;
    // Screen (NDC-ish) to translated world-space
    layout(offset = 832) mat4 View_ScreenToTranslatedWorld;

    // For orthographic path or fallback
    layout(offset = 1168) vec3 View_ViewForward;

    // DeviceZ -> WorldZ reconstruction parameters (engine-specific)
    layout(offset = 1248) vec4 View_InvDeviceZToWorldZTransform;

    // Scale/Bias used for mapping between NDC and screen space (engine-specific)
    layout(offset = 1264) vec4 View_ScreenPositionScaleBias;

    // Camera origin in translated world space
    layout(offset = 1296) vec3 View_TranslatedWorldCameraOrigin;

    // Clip -> previous frame clip transform
    layout(offset = 2176) mat4 View_ClipToPrevClip;

    // View rect min (pixels)
    layout(offset = 2368) vec4 View_ViewRectMin;
    // View size and inv size (xy=size, zw=invSize)
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    // Buffer size and inv size (xy=size, zw=invSize)
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;

    // Frame index mod 8 used to rotate dithering/noise
    layout(offset = 2648) uint View_StateFrameIndexMod8;
} View;

layout(set = 1, binding = 0, std140) uniform type_Globals
{
    // xy: scale from screen UV to HZB UV (or vice versa), zw: inverse factor.
    // In shader: xy used to scale UVs into HZB space; zw used to go back.
    vec4 HZBUvFactorAndInvFactor;

    // SSRParams:
    //   x: overall intensity scale
    //   y: roughness -> step/threshold shaping (used to compute _315)
    //   (z,w): not used in this snippet
    vec4 SSRParams;

    // History scene color needs pre-exposure correction into current exposure.
    float PrevSceneColorPreExposureCorrection;

    // Previous screen position scale/bias for sampling history scene color.
    vec4 PrevScreenPositionScaleBias;
} Globals;

// -----------------------------------------------------------------------------
// Textures/samplers
// -----------------------------------------------------------------------------

layout(set = 1, binding = 10) uniform sampler VulkanGlobalPointClampedSampler;

layout(set = 1, binding = 2)  uniform texture2D SceneDepthTexture;       // device depth
layout(set = 1, binding = 3)  uniform texture2D GBufferATexture;         // packed normal (likely)
layout(set = 1, binding = 4)  uniform texture2D GBufferBTexture;         // material params/flags
layout(set = 1, binding = 5)  uniform texture2D GBufferDTexture;         // extra material data
layout(set = 1, binding = 6)  uniform texture2D GBufferVelocityTexture;  // motion vectors
layout(set = 1, binding = 7)  uniform texture2D GBufferFTexture;         // anisotropy/tangent? (engine-specific)
layout(set = 1, binding = 8)  uniform texture2D HZBTexture;              // hierarchical Z buffer
layout(set = 1, binding = 9)  uniform texture2D SceneColor;              // history / previous scene color
layout(set = 1, binding = 11) uniform sampler SceneColorSampler;

layout(location = 0) out vec4 out_var_SV_Target0;

// -----------------------------------------------------------------------------
// Helper: reconstruct world Z from device depth (engine-provided transform)
// Decompiled formula:
//   worldZ = (d*A + B) + 1 / (d*C - D)
// -----------------------------------------------------------------------------
float ReconstructWorldZ(float deviceDepth)
{
    vec4 t = View.View_InvDeviceZToWorldZTransform;
    return ((deviceDepth * t.x) + t.y) + (1.0 / ((deviceDepth * t.z) - t.w));
}

// -----------------------------------------------------------------------------
// Helper: decode packed normal from [0,1] to [-1,1] and normalize.
// -----------------------------------------------------------------------------
vec3 DecodeAndNormalizeNormal(vec3 enc01)
{
    vec3 n = (enc01 * 2.0) - vec3(1.0);
    return normalize(n);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
void main()
{
    vec4 outColor;

    // The original shader used a do-while(false) to allow early 'break'.
    do
    {
        // ---------------------------------------------------------------------
        // 1) Compute UVs
        // ---------------------------------------------------------------------
        // Buffer UV in [0,1] over full render target.
        vec2 bufferUV = gl_FragCoord.xy * View.View_BufferSizeAndInvSize.zw;

        // View-rect normalized UV in [0,1] over the view (viewport) region.
        vec2 viewRectUV = (gl_FragCoord.xy - View.View_ViewRectMin.xy) * View.View_ViewSizeAndInvSize.zw;

        // NDC xy in [-1,1] (with y flipped to match clip conventions).
        float ndcX = (2.0 * viewRectUV.x) - 1.0;
        float ndcY = 1.0 - (2.0 * viewRectUV.y);

        // ---------------------------------------------------------------------
        // 2) Fetch depth and reconstruct world-space depth value used by engine
        // ---------------------------------------------------------------------
        float deviceDepth = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).x;
        float worldZ      = ReconstructWorldZ(deviceDepth);

        // ---------------------------------------------------------------------
        // 3) Fetch GBuffer data (material + normal + optional anisotropy)
        // ---------------------------------------------------------------------
        vec4 gbufferF = textureLod(sampler2D(GBufferFTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);
        vec4 gbufferB = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);

        float roughnessOrSmoothness = gbufferB.z; // used as "roughness-like" scalar in SSR

        // gbufferB.w encodes flags; decompiled uses *255 and bitfields.
        uint packedMaterialFlags = uint((gbufferB.w * 255.0) + 0.5);
        uint shadingModelOrType  = packedMaterialFlags & 15u; // low 4 bits

        // World-space normal (or translated-world normal) from GBufferA
        vec3 worldNormal = DecodeAndNormalizeNormal(textureLod(
            sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler),
            bufferUV, 0.0
        ).xyz);

        // The shader selectively loads GBufferD only for certain shading models.
        // It checks shadingModelOrType in [2..9], inclusive, via chained comparisons.
        // For other types, gbufferD is treated as 0.
        bool usesGBufferD =
            (shadingModelOrType == 2u) || (shadingModelOrType == 3u) ||
            (shadingModelOrType == 4u) || (shadingModelOrType == 5u) ||
            (shadingModelOrType == 6u) || (shadingModelOrType == 7u) ||
            (shadingModelOrType == 8u) || (shadingModelOrType == 9u);
        vec4 gbufferD = usesGBufferD
            ? textureLod(sampler2D(GBufferDTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0)
            : vec4(0.0);

        // Optional anisotropy information (engine-specific):
        // bit4..7 then bit0 indicates "has anisotropy/tangent" (from the decompiled test).
        bool hasAnisotropy = ((((packedMaterialFlags >> 4u) & 15u) & 1u) != 0u);

        vec3 anisotropyDir = vec3(0.0);
        float anisotropy   = 0.0; // signed in [-1,1] after decode
        if (hasAnisotropy)
        {
            // gbufferF.xyz in [0,1] -> [-1,1] unit direction
            anisotropyDir = normalize((gbufferF.xyz * 2.0) - vec3(1.0));
            // gbufferF.w in [0,1] -> [-1,1] signed strength
            anisotropy = (gbufferF.w * 2.0) - 1.0;
        }

        // ---------------------------------------------------------------------
        // 4) Determine projection mode and reconstruct a view ray direction
        // ---------------------------------------------------------------------
        // Unreal-style: projection type can be inferred from ViewToClip[3].w.
        // If >= 1.0, treat as orthographic path (or special projection mode).
        bool isOrthoOrSpecialProjection = (View.View_ViewToClip[3].w >= 1.0);

        // For perspective:
        //   we scale NDC xy by worldZ; for ortho, we keep ndc as-is.
        vec2 ndcXYScaledByZ = vec2(ndcX, ndcY) * worldZ;
        float screenXForWorld = isOrthoOrSpecialProjection ? ndcX : ndcXYScaledByZ.x;
        float screenYForWorld = isOrthoOrSpecialProjection ? ndcY : ndcXYScaledByZ.y;

        // Reconstruct translated world position from screen and depth.
        vec4 translatedWorldPos4 = View.View_ScreenToTranslatedWorld * vec4(screenXForWorld, screenYForWorld, worldZ, 1.0);
        vec3 translatedWorldPos  = translatedWorldPos4.xyz;

        // View ray direction (from camera to current pixel world position).
        vec3 viewDirWorld = normalize(translatedWorldPos - View.View_TranslatedWorldCameraOrigin);

        // For orthographic/special projection, use ViewForward instead.
        vec3 viewForwardOrViewDir = isOrthoOrSpecialProjection ? View.View_ViewForward : viewDirWorld;

        // ---------------------------------------------------------------------
        // 5) Adjust normal for anisotropy (when present and enabled by shading model)
        // ---------------------------------------------------------------------
        // gbufferD.x is used as weight; for shadingModel==4 it blends behaviors.
        float gbufferDWeight = gbufferD.x;
        bool isSpecialShadingModel4 = (shadingModelOrType == 4u);

        // When shading model 4 is active, anisotropy is suppressed depending on gbufferD.x.
        // Decompiled:
        //   anis = mix(anisotropy, 0, (isSM4 ? gbufferD.x : 0))
        float anisotropyMasked = mix(anisotropy, 0.0, isSpecialShadingModel4 ? gbufferDWeight : 0.0);
        float anisAbs = abs(anisotropyMasked);

        vec3 reflectionNormal = worldNormal;
        if (anisAbs > 0.0)
        {
            // Build a stable tangent-like direction:
            // - if anisotropyMasked >= 0: use anisotropyDir
            // - else: use cross(normal, anisotropyDir)
            // This matches the decompiled "mix(..., bvec3(anis>=0))" pattern.
            vec3 tangentLike = (anisotropyMasked >= 0.0)
                ? anisotropyDir
                : normalize(cross(worldNormal, anisotropyDir));

            // Bend the normal toward an anisotropic microfacet-like direction.
            // Strength scales with anisAbs and with a clamped function of roughnessOrSmoothness.
            float anisStrength = anisAbs * clamp(5.0 * roughnessOrSmoothness, 0.0, 1.0);

            // Decompiled construction:
            //   target = cross(cross(tangentLike, -V), tangentLike)
            //   N' = normalize( mix(N, target, anisStrength) )
            vec3 targetN = cross(cross(tangentLike, -viewForwardOrViewDir), tangentLike);
            reflectionNormal = normalize(mix(worldNormal, targetN, vec3(anisStrength)));
        }

        // ---------------------------------------------------------------------
        // 6) SSR eligibility + step factor
        // ---------------------------------------------------------------------
        // If shadingModel==4, roughness can be blended with gbufferD.y using gbufferD.x as weight.
        float effectiveRoughness = isSpecialShadingModel4
            ? mix(roughnessOrSmoothness, gbufferD.y, gbufferDWeight)
            : roughnessOrSmoothness;

        // Computes a visibility/enable factor in [0,1]:
        //   stepFactor = min(effectiveRoughness * SSRParams.y + 2, 1)
        // This decompiler uses NaN-safe min.
        float ssrEnableFactor = spvNMin((effectiveRoughness * Globals.SSRParams.y) + 2.0, 1.0);

        // Early-out:
        // If ssrEnableFactor <= 0, allow only shadingModel==0; else skip.
        // Decompiled:
        //   if ((!(_315 <= 0.0)) ? (_182 == 0u) : true) out=0
        // Meaning: if ssrEnableFactor > 0 and shadingModel==0 => disable,
        //         if ssrEnableFactor <=0 => disable.
        if ((ssrEnableFactor <= 0.0) || (shadingModelOrType == 0u))
        {
            outColor = vec4(0.0);
            break;
        }

        // ---------------------------------------------------------------------
        // 7) Compute reflection direction and maximum ray distance
        // ---------------------------------------------------------------------
        vec3 reflectDirWorld = reflect(viewForwardOrViewDir, reflectionNormal);

        // Large clamp matches decompiled min(worldZ, 1e6)
        float maxRayDistanceWorld = spvNMin(worldZ, 1000000.0);

        // Compute how far we can march before hitting near plane / leaving valid region.
        // Convert reflect dir to view space to inspect its z.
        float reflectDirViewZ = (View.View_TranslatedWorldToView * vec4(reflectDirWorld, 0.0)).z;

        float rayDistanceWorld;
        if (reflectDirViewZ < 0.0)
        {
            // (-0.95 * worldZ) / reflectDirViewZ, clamped.
            // This ensures we don't march past a plane in front of camera.
            rayDistanceWorld = spvNMin(((-0.95) * worldZ) / reflectDirViewZ, maxRayDistanceWorld);
        }
        else
        {
            rayDistanceWorld = maxRayDistanceWorld;
        }

        // ---------------------------------------------------------------------
        // 8) Project start/end into clip space to derive a screen-space ray delta
        // ---------------------------------------------------------------------
        vec4 clipStart = View.View_TranslatedWorldToClip * vec4(translatedWorldPos, 1.0);
        vec4 clipEnd   = View.View_TranslatedWorldToClip * vec4(translatedWorldPos + (reflectDirWorld * rayDistanceWorld), 1.0);

        vec3 ndcStart = clipStart.xyz * (1.0 / clipStart.w);
        // clipStart advanced along view-space z by rayDistanceWorld (projection-dependent)
        vec4 clipStartAdvanced = clipStart + (View.View_ViewToClip * vec4(0.0, 0.0, rayDistanceWorld, 0.0));
        vec3 ndcStartAdvanced  = clipStartAdvanced.xyz * (1.0 / clipStartAdvanced.w);

        vec3 ndcRay = (clipEnd.xyz * (1.0 / clipEnd.w)) - ndcStart; // delta in NDC
        vec2 ndcRayXY = ndcRay.xy;

        // ---------------------------------------------------------------------
        // 9) Compute normalized stepping in NDC so we stay inside screen bounds
        // ---------------------------------------------------------------------
        // This section ensures the ray step doesn't overshoot the viewport in XY.
        // Decompiled math computes a scale based on "distance to edges" heuristics.
        float halfLenXY = 0.5 * length(ndcRayXY);
        vec2 startXY = ndcStart.xy;

        // Avoid division by zero via behavior of spvNMax and abs; original relied on IEEE behavior.
        vec2 tEdge = vec2(1.0) - (spvNMax(abs(ndcRayXY + (startXY * halfLenXY)) - vec2(halfLenXY), vec2(0.0)) / abs(ndcRayXY));
        vec3 ndcRayClipped = ndcRay * (spvNMin(tEdge.x, tEdge.y) / halfLenXY);

        // Z step magnitude (used for depth compare tolerance), depends on projection mode.
        float ndcZStepMag;
        if (isOrthoOrSpecialProjection)
        {
            ndcZStepMag = spvNMax(0.0, (ndcStart.z - ndcStartAdvanced.z) * 4.0);
        }
        else
        {
            ndcZStepMag = spvNMax(abs(ndcRayClipped.z), (ndcStart.z - ndcStartAdvanced.z) * 4.0);
        }

        // ---------------------------------------------------------------------
        // 10) Convert to HZB space and setup 1/16 stepping
        // ---------------------------------------------------------------------
        // HZB UV scaling: NDC->ScreenUV mapping:
        //   screenUV = ndc * (0.5,-0.5) + 0.5
        // Then scaled by HZB factors.
        vec2 hzbStepXY = (ndcRayClipped.xy * vec2(0.5, -0.5)) * Globals.HZBUvFactorAndInvFactor.xy;

        // Per 4-sample group: the shader steps by 1/16 along the ray each iteration.
        float depthTolerance = ndcZStepMag * 0.0625; // _432
        vec3 stepPerSample   = vec3(hzbStepXY, ndcRayClipped.z) * 0.0625; // _433

        // Initial ray point in HZB UV + NDC Z, plus a small dither along the ray
        // to reduce banding (frame-indexed noise).
        float noiseSeed = dot(
            gl_FragCoord.xy + (vec2(32.6650009, 11.8150) * float(View.View_StateFrameIndexMod8)),
            vec2(0.06711056, 0.0058371499)
        );
        float rand01 = fract(52.9829178 * fract(noiseSeed)); // [0,1)
        float dither = rand01 - 0.5;                         // [-0.5,0.5)

        vec3 hzbRayStart = vec3(
            ((ndcStart.xy * vec2(0.5, -0.5)) + vec2(0.5)) * Globals.HZBUvFactorAndInvFactor.xy,
            ndcStart.z
        ) + (stepPerSample * fract(dither)); // small fractional offset along the ray

        // ---------------------------------------------------------------------
        // 11) HZB ray march (up to 16 steps, grouped as 4 samples per loop)
        // ---------------------------------------------------------------------
        // The decompiled shader samples 4 points at a time, and raises mip level
        // (lod) as it marches. It detects a "hit" when:
        //   abs(rayZ - hzbZ) < tolerance AND hzbZ != 0
        //
        // Variables in this block correspond to original _437/_440/_446/_448 etc.
        bvec4 hitMask = bvec4(false);
        vec4  rayMinusHZB = vec4(0.0);
        uint  stepIndex = 0u;           // counts samples in units of 4
        float prevWDelta = 0.0;         // last lane's rayMinusHZB.w used for interpolation
        bool  anyHitSoFar = false;
        float lod = 1.0;                // starts at 1, increases by ~roughness each group

        // Keep last evaluated values for selecting earliest hit
        bvec4 lastHitMask = bvec4(false);
        vec4  lastDelta   = vec4(0.0);
        bool  lastAnyHit  = false;

        SPIRV_CROSS_LOOP
        for (;;)
        {
            if (stepIndex >= 16u)
            {
                lastHitMask = hitMask;
                lastDelta   = rayMinusHZB;
                lastAnyHit  = anyHitSoFar;
                break;
            }

            vec2 baseUV = hzbRayStart.xy;
            float baseZ = hzbRayStart.z;

            float i0 = float(stepIndex) + 1.0;
            float i1 = float(stepIndex) + 2.0;
            float i2 = float(stepIndex) + 3.0;
            float i3 = float(stepIndex) + 4.0;

            // LOD grows with effectiveRoughness (_309). The decompiled does:
            //   lodNext = lod + 0.5*effectiveRoughness; then again for 3rd/4th sample
            float halfR = 0.5 * effectiveRoughness;
            float lodMid = lod + halfR;
            float lodNext = lodMid + halfR;

            // Sample HZB at four positions along the ray, at two different LODs.
            float hzb0 = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), baseUV + (stepPerSample.xy * i0), lod).x;
            float hzb1 = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), baseUV + (stepPerSample.xy * i1), lod).x;
            float hzb2 = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), baseUV + (stepPerSample.xy * i2), lodMid).x;
            float hzb3 = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), baseUV + (stepPerSample.xy * i3), lodMid).x;

            vec4 rayZ = vec4(
                baseZ + (i0 * stepPerSample.z),
                baseZ + (i1 * stepPerSample.z),
                baseZ + (i2 * stepPerSample.z),
                baseZ + (i3 * stepPerSample.z)
            );
            vec4 hzbZ = vec4(hzb0, hzb1, hzb2, hzb3);

            // delta = rayZ - hzbZ
            vec4 delta = rayZ - hzbZ;

            vec4 tol = vec4(depthTolerance);
            bvec4 withinTol = lessThan(abs(delta + tol), tol); // matches decompiled tolerance check
            bvec4 hzbValid  = notEqual(hzbZ, vec4(0.0));       // discard empty tiles

            bool hit0 = withinTol.x && hzbValid.x;
            bool hit1 = withinTol.y && hzbValid.y;
            bool hit2 = withinTol.z && hzbValid.z;
            bool hit3 = withinTol.w && hzbValid.w;

            bvec4 groupHitMask = bvec4(hit0, hit1, hit2, hit3);

            // anyHit = any previous hit OR any lane hit in this group
            bool anyHit = anyHitSoFar || hit0 || hit1 || hit2 || hit3;

            SPIRV_CROSS_BRANCH
            if (anyHit)
            {
                // Store group result and break; selection of earliest hit happens after loop.
                lastHitMask = groupHitMask;
                lastDelta   = delta;
                lastAnyHit  = anyHit;
                prevWDelta  = delta.w;
                stepIndex  += 4u; // preserve parity with original; used in hit position calc
                lod          = lodNext;
                break;
            }

            // No hit in this group; advance.
            prevWDelta  = delta.w;
            hitMask     = groupHitMask;
            rayMinusHZB = delta;
            anyHitSoFar = anyHit;
            lod         = lodNext;
            stepIndex  += 4u;
        }

        // ---------------------------------------------------------------------
        // 12) Compute hit point along the ray (with sub-step refinement)
        // ---------------------------------------------------------------------
        vec3 hzbHitPoint;
        bool hitFound = lastAnyHit;

        if (hitFound)
        {
            // The decompiled code chooses the earliest true lane among x/y/z/w with
            // a somewhat convoluted selection, then performs a linear interpolation:
            //   t = clamp(prevDelta / (prevDelta - currentDelta), 0, 1)
            //
            // We'll implement the same selection behavior:
            float chosenDeltaCurrent;
            float chosenDeltaPrev;
            float laneOffset;

            // Lane selection order: x -> y -> z -> w (earliest).
            if (lastHitMask.x)
            {
                chosenDeltaCurrent = lastDelta.x;
                chosenDeltaPrev    = prevWDelta; // decompiled uses _448 when lane x hits
                laneOffset         = 0.0;
            }
            else if (lastHitMask.y)
            {
                chosenDeltaCurrent = lastDelta.y;
                chosenDeltaPrev    = lastDelta.x;
                laneOffset         = 1.0;
            }
            else if (lastHitMask.z)
            {
                chosenDeltaCurrent = lastDelta.z;
                chosenDeltaPrev    = lastDelta.y;
                laneOffset         = 2.0;
            }
            else
            {
                chosenDeltaCurrent = lastDelta.w;
                chosenDeltaPrev    = lastDelta.z;
                laneOffset         = 3.0;
            }

            // Refine within the segment between previous sample and current sample.
            float refineT = clamp(chosenDeltaPrev / (chosenDeltaPrev - chosenDeltaCurrent), 0.0, 1.0);

            // Convert stepIndex to float for position along ray.
            // Note: stepIndex at this point is the group base; laneOffset selects inside group.
            float sampleIndex = float(stepIndex) + laneOffset + refineT;
            hzbHitPoint = hzbRayStart + (stepPerSample * sampleIndex);
        }
        else
        {
            // No hit: just end point at last marched sample count.
            hzbHitPoint = hzbRayStart + (stepPerSample * float(stepIndex));
        }

        // ---------------------------------------------------------------------
        // 13) If hit, reproject to previous frame and sample history scene color
        // ---------------------------------------------------------------------
        vec4 historyColor = vec4(0.0);
        if (hitFound)
        {
            // hzbHitPoint.xy currently in HZB UV scale.
            // Convert back to "screen UV" used by the velocity buffer and View scale/bias.
            //
            // Decompiled:
            //   _591 = (((((hzb.xy * invFactor).xy * (2,-2)) + (-1,1)) * ScreenPosScaleBias.xy) + ScreenPosScaleBias.wz)
            vec2 ndcFromHZB = (hzbHitPoint.xy * Globals.HZBUvFactorAndInvFactor.zw) * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
            vec2 currScreenPos = (ndcFromHZB * View.View_ScreenPositionScaleBias.xy) + View.View_ScreenPositionScaleBias.wz;

            // Also compute normalized NDC-ish position for clip transform usage.
            vec2 currNdc = (currScreenPos - View.View_ScreenPositionScaleBias.wz) / View.View_ScreenPositionScaleBias.xy;

            vec4 currClipLike = vec4(currNdc, hzbHitPoint.z, 1.0);
            vec4 prevClip = View.View_ClipToPrevClip * currClipLike;

            // Velocity texture is sampled at currScreenPos (not bufferUV).
            vec4 velocityPacked = textureLod(sampler2D(GBufferVelocityTexture, VulkanGlobalPointClampedSampler), currScreenPos, 0.0);

            // Two reprojection paths:
            //   - If velocityPacked.x > 0, decode velocity and apply a non-linear warp
            //     (matches UE's velocity encoding that squares magnitude while keeping sign).
            //   - Else, use clip transform prevClip.xy/prevClip.w
            vec2 prevNdc;
            if (velocityPacked.x > 0.0)
            {
                // Decode velocity from [0,1] into approx [-2,2], then apply v*abs(v)*0.5.
                vec2 v = (velocityPacked.xy * 4.0080161) - vec2(2.0039775);
                prevNdc = currClipLike.xy - ((v * abs(v)) * 0.5);
            }
            else
            {
                prevNdc = prevClip.xy / vec2(prevClip.w);
            }

            // Edge fade to reduce artifacts near screen border.
            vec2 edgeFadeCurr = clamp((abs(currNdc) * 5.0) - vec2(4.0), vec2(0.0), vec2(1.0));
            vec2 edgeFadePrev = clamp((abs(prevNdc) * 5.0) - vec2(4.0), vec2(0.0), vec2(1.0));

            float fade = spvNMin(
                clamp(1.0 - dot(edgeFadeCurr, edgeFadeCurr), 0.0, 1.0),
                clamp(1.0 - dot(edgeFadePrev, edgeFadePrev), 0.0, 1.0)
            );

            // Sample history scene color.
            vec2 prevScreenUV = (prevNdc * Globals.PrevScreenPositionScaleBias.xy) + Globals.PrevScreenPositionScaleBias.zw;
            vec3 prevSceneColor = textureLod(sampler2D(SceneColor, SceneColorSampler), prevScreenUV, 0.0).xyz;

            // Decompiled does: -min(-color,0) == max(color,0)
            prevSceneColor = -spvNMin(-prevSceneColor, vec3(0.0));

            historyColor = vec4(prevSceneColor, 1.0) * fade;
        }

        // ---------------------------------------------------------------------
        // 14) Final compose
        // ---------------------------------------------------------------------
        // intensity = historyColor * ssrEnableFactor * SSRParams.x
        vec4 ssr = (historyColor * ssrEnableFactor) * Globals.SSRParams.x;

        // Apply pre-exposure correction to RGB.
        vec3 rgb = ssr.xyz * Globals.PrevSceneColorPreExposureCorrection;
        outColor = vec4(rgb, ssr.w);

        break;
    } while (false);

    out_var_SV_Target0 = outColor;
}


