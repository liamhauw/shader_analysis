#version 460
// -----------------------------------------------------------------------------
// Screen Space Reflections (SSR) in a single fragment shader:
// - Reconstruct surface data from GBuffer (normal, roughness, depth)
// - Build a reflection ray in world space
// - Project the ray to screen space and march it against a hierarchical depth buffer (HZB)
// - If a hit is found, reproject to previous frame and sample history scene color
// - Apply roughness fade and exposure correction
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
// Uniform buffers
// -----------------------------------------------------------------------------

layout(set = 1, binding = 1, std140) uniform type_View
{
    // Translated world-space -> clip-space (current frame)
    layout(offset = 0)   mat4 View_TranslatedWorldToClip;
    // Translated world-space -> view-space
    layout(offset = 192) mat4 View_TranslatedWorldToView;
    // View-space -> clip-space (projection)
    layout(offset = 448) mat4 View_ViewToClip;
    // Screen-position (projection-type dependent) -> translated world-space
    layout(offset = 832) mat4 View_ScreenToTranslatedWorld;

    // Camera forward (used for special/orthographic cases)
    layout(offset = 1168) vec3 View_ViewForward;

    // DeviceZ -> linear depth reconstruction parameters
    layout(offset = 1248) vec4 View_InvDeviceZToWorldZTransform;

    // Scale/Bias for mapping between NDC and screen position
    layout(offset = 1264) vec4 View_ScreenPositionScaleBias;

    // Camera origin in translated world-space
    layout(offset = 1296) vec3 View_TranslatedWorldCameraOrigin;

    // Current clip -> previous clip transform
    layout(offset = 2176) mat4 View_ClipToPrevClip;

    // View-rect min (pixels)
    layout(offset = 2368) vec4 View_ViewRectMin;
    // View size and inverse (xy=size, zw=invSize)
    layout(offset = 2384) vec4 View_ViewSizeAndInvSize;
    // Buffer size and inverse (xy=size, zw=invSize)
    layout(offset = 2432) vec4 View_BufferSizeAndInvSize;

    // Frame index mod 8 (used for noise/dither)
    layout(offset = 2648) uint View_StateFrameIndexMod8;
} View;

layout(set = 1, binding = 0, std140) uniform type_Globals
{
    // xy: scale to HZB UV space, zw: inverse scale back to screen UV space
    vec4 HZBUvFactorAndInvFactor;

    // x: intensity scale, y: roughness fade shaping
    vec4 SSRParams;

    // Exposure correction for history scene color
    float PrevSceneColorPreExposureCorrection;

    // Scale/Bias for sampling previous frame scene color
    vec4 PrevScreenPositionScaleBias;
} Globals;

// -----------------------------------------------------------------------------
// Textures/samplers
// -----------------------------------------------------------------------------

layout(set = 1, binding = 10) uniform sampler VulkanGlobalPointClampedSampler;

layout(set = 1, binding = 2)  uniform texture2D SceneDepthTexture;       // device depth
layout(set = 1, binding = 3)  uniform texture2D GBufferATexture;         // encoded normal
layout(set = 1, binding = 4)  uniform texture2D GBufferBTexture;         // roughness + packed flags
layout(set = 1, binding = 5)  uniform texture2D GBufferDTexture;         // per-material custom data
layout(set = 1, binding = 6)  uniform texture2D GBufferVelocityTexture;  // motion vectors (history reprojection)
layout(set = 1, binding = 7)  uniform texture2D GBufferFTexture;         // anisotropy/tangent + strength (if present)
layout(set = 1, binding = 8)  uniform texture2D HZBTexture;              // hierarchical depth buffer
layout(set = 1, binding = 9)  uniform texture2D SceneColor;              // previous frame scene color
layout(set = 1, binding = 11) uniform sampler SceneColorSampler;

layout(location = 0) out vec4 out_var_SV_Target0;

void main()
{
    vec4 OutColor = vec4(0.0);

    // Pixel UV in full render target.
    vec2 UV = gl_FragCoord.xy * View.View_BufferSizeAndInvSize.zw;
    vec2 ViewportUV = (gl_FragCoord.xy - View.View_ViewRectMin.xy) * View.View_ViewSizeAndInvSize.zw;

    // NDC-like screen position in [-1,1] (y flipped).
    vec2 ScreenPos;
    ScreenPos.x = (2.0 * ViewportUV.x) - 1.0;
    ScreenPos.y = 1.0 - (2.0 * ViewportUV.y);

    // Read depth and reconstruct linear depth.
    float DeviceZ = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), UV, 0.0).x;

    vec4 InvDeviceZToWorldZ = View.View_InvDeviceZToWorldZTransform;
    float SceneDepth = ((DeviceZ * InvDeviceZToWorldZ.x) + InvDeviceZToWorldZ.y) +
                       (1.0 / ((DeviceZ * InvDeviceZToWorldZ.z) - InvDeviceZToWorldZ.w));

    // Read roughness and packed per-pixel material flags.
    vec4 GBufferB = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), UV, 0.0);
    float Roughness = GBufferB.z;
    uint PackedMaterialFlags = uint((GBufferB.w * 255.0) + 0.5);
    uint ShadingModelID = PackedMaterialFlags & 15u;
    bool bNoMaterial = (ShadingModelID == 0u);

    // Decode normal from [0,1] to [-1,1] and normalize.
    vec3 N = (textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), UV, 0.0).xyz * 2.0) - vec3(1.0);
    N = normalize(N);

    // Read custom data only for shading models that store it.
    bool bNeedsCustomData =
        (ShadingModelID == 2u) || (ShadingModelID == 3u) ||
        (ShadingModelID == 4u) || (ShadingModelID == 5u) ||
        (ShadingModelID == 6u) || (ShadingModelID == 7u) ||
        (ShadingModelID == 8u) || (ShadingModelID == 9u);
    vec4 CustomData = bNeedsCustomData
        ? textureLod(sampler2D(GBufferDTexture, VulkanGlobalPointClampedSampler), UV, 0.0)
        : vec4(0.0);

    // Optionally override roughness using custom data (e.g., clear-coat top layer).
    bool bIsClearCoatLike = (ShadingModelID == 4u);
    if (bIsClearCoatLike)
    {
        float ClearCoat = CustomData.x;
        float ClearCoatRoughness = CustomData.y;
        Roughness = mix(Roughness, ClearCoatRoughness, ClearCoat);
    }

    // Roughness-based fade to reduce noise and skip SSR for very rough pixels.
    float RoughnessFade = spvNMin(Roughness * Globals.SSRParams.y + 2.0, 1.0);
    SPIRV_CROSS_BRANCH
    if (RoughnessFade <= 0.0 || bNoMaterial)
    {
        out_var_SV_Target0 = vec4(0.0);
        return;
    }

    // Reconstruct translated world position from screen position and depth.
    bool bIsOrtho = (View.View_ViewToClip[3].w >= 1.0);
    vec2 ScreenPosForWorld = bIsOrtho ? ScreenPos : (ScreenPos * SceneDepth);
    vec3 PositionTranslatedWorld = (View.View_ScreenToTranslatedWorld * vec4(ScreenPosForWorld, SceneDepth, 1.0)).xyz;
    vec3 V = normalize(View.View_TranslatedWorldCameraOrigin - PositionTranslatedWorld); // surface -> camera

    // Optional anisotropy: adjust the shading normal based on tangent direction and anisotropy strength.
    vec4 GBufferF = textureLod(sampler2D(GBufferFTexture, VulkanGlobalPointClampedSampler), UV, 0.0);
    bool bHasAnisotropy = ((((PackedMaterialFlags >> 4u) & 15u) & 1u) != 0u);
    float SSRAnisotropy = 0.0;
    vec3 WorldTangent = vec3(0.0);
    if (bHasAnisotropy)
    {
        WorldTangent = normalize((GBufferF.xyz * 2.0) - vec3(1.0));
        SSRAnisotropy = (GBufferF.w * 2.0) - 1.0; // [-1,1]
    }

    // Optionally suppress anisotropy using custom data (e.g., clear-coat top layer only).
    float AnisotropyBlendValue = bIsClearCoatLike ? CustomData.x : 0.0;
    SSRAnisotropy = mix(SSRAnisotropy, 0.0, AnisotropyBlendValue);

    if (abs(SSRAnisotropy) > 0.0)
    {
        vec3 TangentLike = (SSRAnisotropy >= 0.0) ? WorldTangent : normalize(cross(N, WorldTangent));
        float AnisStrength = abs(SSRAnisotropy) * clamp(5.0 * Roughness, 0.0, 1.0);
        vec3 TargetN = cross(cross(TangentLike, V), TangentLike);
        N = normalize(mix(N, TargetN, vec3(AnisStrength)));
    }

    // Reflection direction in world space.
    vec3 WorldRayDirection = reflect(-V, N);

    // Ray-march parameters.
    const uint NumSteps = 16u;
    const float StartMipLevel = 1.0;
    const float SlopeCompareToleranceScale = 4.0;
    const float WorldTMaxClamp = 1000000.0;

    // Clamp max ray distance and prevent marching past the near plane (in view space).
    float WorldTMax = spvNMin(SceneDepth, WorldTMaxClamp);
    vec3 ViewRayDirection = (View.View_TranslatedWorldToView * vec4(WorldRayDirection, 0.0)).xyz;
    float RayEndWorldDistance = (ViewRayDirection.z < 0.0) ? spvNMin((-0.95 * SceneDepth) / ViewRayDirection.z, WorldTMax) : WorldTMax;

    vec3 RayEndWorld = PositionTranslatedWorld + WorldRayDirection * RayEndWorldDistance;

    // Project ray endpoints to screen space.
    vec4 RayStartClip = View.View_TranslatedWorldToClip * vec4(PositionTranslatedWorld, 1.0);
    vec4 RayEndClip   = View.View_TranslatedWorldToClip * vec4(RayEndWorld, 1.0);

    vec3 RayStartScreen = RayStartClip.xyz * (1.0 / RayStartClip.w);
    vec3 RayEndScreen   = RayEndClip.xyz * (1.0 / RayEndClip.w);

    // Project a pure depth advance to estimate depth slope in screen space.
    vec4 RayDepthClip = RayStartClip + (View.View_ViewToClip * vec4(0.0, 0.0, RayEndWorldDistance, 0.0));
    vec3 RayDepthScreen = RayDepthClip.xyz * (1.0 / RayDepthClip.w);

    vec3 RayStepScreen = RayEndScreen - RayStartScreen;

    // Clip the ray so it stays within the screen bounds.
    float RayStepScreenInvFactor = 0.5 * length(RayStepScreen.xy);
    vec2 S = vec2(1.0) - (spvNMax(abs(RayStepScreen.xy + RayStartScreen.xy * RayStepScreenInvFactor) - vec2(RayStepScreenInvFactor), vec2(0.0)) / abs(RayStepScreen.xy));
    float RayStepFactor = spvNMin(S.x, S.y) / RayStepScreenInvFactor;

    RayStepScreen *= RayStepFactor;

    // Depth comparison tolerance increases with projected depth slope and ray depth change.
    float RayCompareTolerance;
    if (bIsOrtho)
    {
        RayCompareTolerance = spvNMax(0.0, (RayStartScreen.z - RayDepthScreen.z) * SlopeCompareToleranceScale);
    }
    else
    {
        RayCompareTolerance = spvNMax(abs(RayStepScreen.z), (RayStartScreen.z - RayDepthScreen.z) * SlopeCompareToleranceScale);
    }

    // Convert screen-space ray into HZB UV + depth coordinates.
    vec3 RayStartUVz = vec3((RayStartScreen.xy * vec2(0.5, -0.5) + vec2(0.5)) * Globals.HZBUvFactorAndInvFactor.xy, RayStartScreen.z);
    vec3 RayStepUVz  = vec3((RayStepScreen.xy  * vec2(0.5, -0.5))             * Globals.HZBUvFactorAndInvFactor.xy, RayStepScreen.z);

    float Step = 1.0 / float(NumSteps);
    float CompareTolerance = RayCompareTolerance * Step;

    float LastDiff = 0.0;
    float Level = StartMipLevel;

    // Scale step by the number of iterations and add a per-pixel offset to reduce banding.
    RayStepUVz *= Step;

    float noiseSeed = dot(
        gl_FragCoord.xy + (vec2(32.6650009, 11.8150) * float(View.View_StateFrameIndexMod8)),
        vec2(0.06711056, 0.0058371499)
    );
    float StepOffset = fract(52.9829178 * fract(noiseSeed)) - 0.5;
    vec3 RayUVz = RayStartUVz + RayStepUVz * StepOffset;

    bool bFoundHit = false;
    vec4 MultipleSampleDepthDiff = vec4(0.0);
    bvec4 bMultipleSampleHit = bvec4(false);
    uint i = 0u;

    // Skip hits against invalid depth values.
    const float FarDepthValue = 0.0;

    SPIRV_CROSS_LOOP
    for (i = 0u; i < NumSteps; i += 4u)
    {
        vec2 SamplesUV0 = RayUVz.xy + (float(i) + 1.0) * RayStepUVz.xy;
        vec2 SamplesUV1 = RayUVz.xy + (float(i) + 2.0) * RayStepUVz.xy;
        vec2 SamplesUV2 = RayUVz.xy + (float(i) + 3.0) * RayStepUVz.xy;
        vec2 SamplesUV3 = RayUVz.xy + (float(i) + 4.0) * RayStepUVz.xy;

        vec4 SamplesZ = vec4(
            RayUVz.z + (float(i) + 1.0) * RayStepUVz.z,
            RayUVz.z + (float(i) + 2.0) * RayStepUVz.z,
            RayUVz.z + (float(i) + 3.0) * RayStepUVz.z,
            RayUVz.z + (float(i) + 4.0) * RayStepUVz.z
        );

        vec4 SamplesMip;
        SamplesMip.xy = vec2(Level);
        Level += (8.0 / float(NumSteps)) * Roughness;
        SamplesMip.zw = vec2(Level);
        Level += (8.0 / float(NumSteps)) * Roughness;

        // Sample hierarchical depth at four points along the ray (batching reduces loop overhead).
        vec4 SampleDepth = vec4(
            textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV0, SamplesMip.x).x,
            textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV1, SamplesMip.y).x,
            textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV2, SamplesMip.z).x,
            textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV3, SamplesMip.w).x
        );

        // DepthDiff > 0 means the ray is in front of the sampled depth; depthDiff < 0 means behind it.
        MultipleSampleDepthDiff = SamplesZ - SampleDepth;
        bMultipleSampleHit = bvec4(
            (abs(MultipleSampleDepthDiff.x + CompareTolerance) < CompareTolerance) && (SampleDepth.x != FarDepthValue),
            (abs(MultipleSampleDepthDiff.y + CompareTolerance) < CompareTolerance) && (SampleDepth.y != FarDepthValue),
            (abs(MultipleSampleDepthDiff.z + CompareTolerance) < CompareTolerance) && (SampleDepth.z != FarDepthValue),
            (abs(MultipleSampleDepthDiff.w + CompareTolerance) < CompareTolerance) && (SampleDepth.w != FarDepthValue)
        );

        bFoundHit = bFoundHit || bMultipleSampleHit.x || bMultipleSampleHit.y || bMultipleSampleHit.z || bMultipleSampleHit.w;
        SPIRV_CROSS_BRANCH
        if (bFoundHit)
        {
            break;
        }

        LastDiff = MultipleSampleDepthDiff.w;
    }

    vec3 HitUVz = vec3(0.0);

    SPIRV_CROSS_BRANCH
    if (bFoundHit)
    {
        // Choose the earliest hit in the 4-sample batch, then refine it using a linear intersection estimate.
        float DepthDiff0 = MultipleSampleDepthDiff.z;
        float DepthDiff1 = MultipleSampleDepthDiff.w;
        float Time0 = 3.0;

        SPIRV_CROSS_FLATTEN
        if (bMultipleSampleHit.z)
        {
            DepthDiff0 = MultipleSampleDepthDiff.y;
            DepthDiff1 = MultipleSampleDepthDiff.z;
            Time0 = 2.0;
        }
        SPIRV_CROSS_FLATTEN
        if (bMultipleSampleHit.y)
        {
            DepthDiff0 = MultipleSampleDepthDiff.x;
            DepthDiff1 = MultipleSampleDepthDiff.y;
            Time0 = 1.0;
        }
        SPIRV_CROSS_FLATTEN
        if (bMultipleSampleHit.x)
        {
            DepthDiff0 = LastDiff;
            DepthDiff1 = MultipleSampleDepthDiff.x;
            Time0 = 0.0;
        }

        Time0 += float(i);
        float TimeLerp = clamp(DepthDiff0 / (DepthDiff0 - DepthDiff1), 0.0, 1.0);
        float IntersectTime = Time0 + TimeLerp;

        HitUVz = RayUVz + RayStepUVz * IntersectTime;
    }
    else
    {
        // No hit: fall back to a conservative position.
        HitUVz = RayUVz + RayStepUVz * float(i);
    }

    // Convert from HZB UV space back to screen position space.
    HitUVz.xy *= Globals.HZBUvFactorAndInvFactor.zw;
    HitUVz.xy = HitUVz.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    HitUVz.xy = HitUVz.xy * View.View_ScreenPositionScaleBias.xy + View.View_ScreenPositionScaleBias.wz;

    // Reproject the hit point into the previous frame and sample history scene color.
    vec4 SampleColor = vec4(0.0);
    float Vignette = 0.0;

    if (bFoundHit)
    {
        // Reproject with clip transform, optionally overridden by per-pixel motion vectors.
        vec2 ThisScreen = (HitUVz.xy - View.View_ScreenPositionScaleBias.wz) / View.View_ScreenPositionScaleBias.xy;
        vec4 ThisClip = vec4(ThisScreen, HitUVz.z, 1.0);
        vec4 PrevClip = View.View_ClipToPrevClip * ThisClip;
        vec2 PrevScreen = PrevClip.xy / PrevClip.w;

        vec4 EncodedVelocity = textureLod(sampler2D(GBufferVelocityTexture, VulkanGlobalPointClampedSampler), HitUVz.xy, 0.0);
        if (EncodedVelocity.x > 0.0)
        {
            vec2 v = (EncodedVelocity.xy * 4.0080161) - vec2(2.0039775);
            PrevScreen = ThisClip.xy - ((v * abs(v)) * 0.5);
        }

        vec2 PrevUV = PrevScreen * Globals.PrevScreenPositionScaleBias.xy + Globals.PrevScreenPositionScaleBias.zw;

        // Edge fade reduces artifacts when the hit reprojects near or beyond the screen border.
        vec2 Vig0 = clamp(abs(ThisScreen) * 5.0 - vec2(4.0), vec2(0.0), vec2(1.0));
        vec2 Vig1 = clamp(abs(PrevScreen) * 5.0 - vec2(4.0), vec2(0.0), vec2(1.0));
        float V0 = clamp(1.0 - dot(Vig0, Vig0), 0.0, 1.0);
        float V1 = clamp(1.0 - dot(Vig1, Vig1), 0.0, 1.0);
        Vignette = spvNMin(V0, V1);

        // Sample history scene color and clamp invalid/negative values to black.
        SampleColor.rgb = textureLod(sampler2D(SceneColor, SceneColorSampler), PrevUV, 0.0).rgb;
        SampleColor.rgb = -spvNMin(-SampleColor.rgb, vec3(0.0));
        SampleColor.a = 1.0;
        SampleColor *= Vignette;
    }

    // Compose final reflection contribution.
    OutColor = SampleColor;
    OutColor *= RoughnessFade;
    OutColor *= Globals.SSRParams.x;
    OutColor.rgb *= Globals.PrevSceneColorPreExposureCorrection;

    out_var_SV_Target0 = OutColor;
}


