#version 460
//
// SSR.generated.glsl
// ------------------
// A readable GLSL reconstruction of the *actual* code path exercised by the
// RenderDoc-decompiled shader in `ssr/ssr.rdc.frag`, mapped back to UE source:
// - `ssr/SSRTReflections.usf`
// - `ssr/SSRTRayCast.ush`
//
// Key "actual path" decisions inferred from the decompile:
// - SSR_QUALITY == 2  -> NumSteps = 16, NumRays = 1
// - Single ray path (NumRays == 1)
// - Non-glossy branch -> L = reflect(-V, N) (no GGX sampling)
// - RayCast uses bExtendRayToScreenBorder = true
// - HZB ray march StartMipLevel = 1.0, mip increases by (8/NumSteps)*Roughness = 0.5*Roughness per 2 samples
// - FarDepthValue for HZB compare is 0.0 in this capture (hits against depth=0 are rejected)
// - Velocity reprojection path enabled: samples `GBufferVelocityTexture` and uses DecodeVelocityFromTexture()
//
// Notes:
// - This file intentionally contains *only* that active path: no #if branches for other SSR qualities.
// - Several packing details (e.g., shading model id bits) follow the decompiled math directly.
// - Resource layouts mirror the RenderDoc/SPIRV-Cross output for easy cross-referencing.

// -----------------------------------------------------------------------------
// Resource bindings (mirrors `ssr/ssr.rdc.frag`)
// -----------------------------------------------------------------------------

layout(set = 1, binding = 1, std140) uniform ViewUBO
{
    // Matrices / transforms (names match decompile prefixes for easy diffing)
    mat4 View_TranslatedWorldToClip;
    mat4 View_TranslatedWorldToView;
    mat4 View_ViewToClip;
    mat4 View_ScreenToTranslatedWorld;

    vec3 View_ViewForward; // only used for ortho

    // DeviceZ -> worldZ conversion parameters (UE: View.InvDeviceZToWorldZTransform)
    vec4 View_InvDeviceZToWorldZTransform;

    // UE: View.ScreenPositionScaleBias
    vec4 View_ScreenPositionScaleBias;

    vec3 View_TranslatedWorldCameraOrigin;

    // UE: View.ClipToPrevClip
    mat4 View_ClipToPrevClip;

    // UE: View.ViewRectMin (xy), other fields unused here
    vec4 View_ViewRectMin;

    // UE: View.ViewSizeAndInvSize (xy=size, zw=inv size)
    vec4 View_ViewSizeAndInvSize;

    // UE: View.BufferSizeAndInvSize (xy=size, zw=inv size)
    vec4 View_BufferSizeAndInvSize;

    uint View_StateFrameIndexMod8;
} View;

layout(set = 1, binding = 0, std140) uniform GlobalsUBO
{
    // UE: HZBUvFactorAndInvFactor
    // .xy = factor into HZB atlas UV space
    // .zw = inverse factor (back to non-atlased UVs)
    vec4 HZBUvFactorAndInvFactor;

    // UE: SSRParams
    // .r = intensity
    // .g = RoughnessMaskMul
    vec4 SSRParams;

    float PrevSceneColorPreExposureCorrection;

    // Equivalent of View.ScreenPositionScaleBias, but for previous frame reprojected UVs.
    vec4 PrevScreenPositionScaleBias;
} Globals;

layout(set = 1, binding = 10) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 1, binding = 2)  uniform texture2D SceneDepthTexture;
layout(set = 1, binding = 3)  uniform texture2D GBufferATexture;
layout(set = 1, binding = 4)  uniform texture2D GBufferBTexture;
layout(set = 1, binding = 5)  uniform texture2D GBufferDTexture;
layout(set = 1, binding = 6)  uniform texture2D GBufferVelocityTexture;
layout(set = 1, binding = 7)  uniform texture2D GBufferFTexture;
layout(set = 1, binding = 8)  uniform texture2D HZBTexture;
layout(set = 1, binding = 9)  uniform texture2D SceneColor;
layout(set = 1, binding = 11) uniform sampler SceneColorSampler;

layout(location = 0) out vec4 outColor;

// -----------------------------------------------------------------------------
// Small math helpers (mirror UE naming)
// -----------------------------------------------------------------------------

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2  saturate(vec2 x)  { return clamp(x, vec2(0.0), vec2(1.0)); }

// UE: ConvertFromDeviceZ(DeviceZ) using View.InvDeviceZToWorldZTransform
float ConvertFromDeviceZ(float DeviceZ)
{
    // Matches the decompile:
    // WorldZ = (DeviceZ * A + B) + (1 / (DeviceZ * C - D))
    return (DeviceZ * View.View_InvDeviceZToWorldZTransform.x + View.View_InvDeviceZToWorldZTransform.y) +
           (1.0 / (DeviceZ * View.View_InvDeviceZToWorldZTransform.z - View.View_InvDeviceZToWorldZTransform.w));
}

// UE: ComputeHitVignetteFromScreenPos
float ComputeHitVignetteFromScreenPos(vec2 ScreenPos)
{
    // In UE this is SafeSaturate; the decompile uses clamp().
    vec2 v = saturate(abs(ScreenPos) * 5.0 - vec2(4.0));
    return saturate(1.0 - dot(v, v));
}

// UE: DecodeVelocityFromTexture (observed exact math from decompile)
vec2 DecodeVelocityFromTexture(vec4 EncodedVelocity)
{
    // Decompile:
    // vel = Encoded.xy * 4.008016109 - 2.003977537
    // return (vel * abs(vel)) * 0.5
    vec2 v = EncodedVelocity.xy * 4.008016109466553 - vec2(2.0039775371551514);
    return (v * abs(v)) * 0.5;
}

// UE: SampleScreenColor (decompile: clamp negatives to 0)
vec4 SampleScreenColorPrevFrame(vec2 PrevUV)
{
    vec3 c = textureLod(sampler2D(SceneColor, SceneColorSampler), PrevUV, 0.0).rgb;
    c = max(c, vec3(0.0)); // matches `-min(-c, 0)`
    return vec4(c, 1.0);
}

// -----------------------------------------------------------------------------
// Screen-space ray setup (from `SSRTRayCast.ush`)
// -----------------------------------------------------------------------------

// UE: GetStepScreenFactorToClipAtScreenEdge()
float GetStepScreenFactorToClipAtScreenEdge(vec2 RayStartScreen, vec2 RayStepScreen)
{
    const float invFactor = 0.5 * length(RayStepScreen);

    // Avoid division by zero in the same spirit as the UE code.
    vec2 denom = max(abs(RayStepScreen), vec2(1e-8));

    vec2 S = vec2(1.0) - (max(abs(RayStepScreen + RayStartScreen * invFactor) - vec2(invFactor), vec2(0.0)) / denom);
    return min(S.x, S.y) / invFactor;
}

struct SSRRay
{
    vec3 RayStartScreen; // NDC-like screen coords: xy in [-1..1], z = DeviceZ
    vec3 RayStepScreen;  // delta in same space
    float CompareTolerance;
};

// UE: IsOrthoProjection() – decompile used this condition
bool IsOrthoProjection()
{
    // Matches decompile: View.ViewToClip[3].w >= 1
    return View.View_ViewToClip[3][3] >= 1.0;
}

// UE: InitScreenSpaceRayFromWorldSpace() but hard-wired to the active SSR path:
// - bExtendRayToScreenBorder = true
// - SlopeCompareToleranceScale = 4 (SSR, not SSGI)
SSRRay InitScreenSpaceRayFromWorldSpace_ActivePath(
    vec3 RayOriginTranslatedWorld,
    vec3 WorldRayDirection,
    float WorldTMax,          // passed as SceneDepth in UE
    float SceneDepth,
    out bool bRayWasClipped)
{
    const float WorldTMaxClamp = 1000000.0;
    WorldTMax = min(WorldTMax, WorldTMaxClamp);

    // Compute distance at which to stop (clip against near plane in view space, per UE).
    vec3 ViewRayDirection = (View.View_TranslatedWorldToView * vec4(WorldRayDirection, 0.0)).xyz;
    float RayEndWorldDistance =
        (ViewRayDirection.z < 0.0) ? min((-0.95 * SceneDepth) / ViewRayDirection.z, WorldTMax) : WorldTMax;

    vec3 RayEndWorld = RayOriginTranslatedWorld + WorldRayDirection * RayEndWorldDistance;

    vec4 RayStartClip = View.View_TranslatedWorldToClip * vec4(RayOriginTranslatedWorld, 1.0);
    vec4 RayEndClip   = View.View_TranslatedWorldToClip * vec4(RayEndWorld, 1.0);

    vec3 RayStartScreen = RayStartClip.xyz / RayStartClip.w;
    vec3 RayEndScreen   = RayEndClip.xyz   / RayEndClip.w;

    // Depth-only ray (used to compute slope tolerance)
    vec4 RayDepthClip   = RayStartClip + (View.View_ViewToClip * vec4(0.0, 0.0, RayEndWorldDistance, 0.0));
    vec3 RayDepthScreen = RayDepthClip.xyz / RayDepthClip.w;

    SSRRay Ray;
    Ray.RayStartScreen = RayStartScreen;
    Ray.RayStepScreen  = RayEndScreen - RayStartScreen;

    // Active path: bExtendRayToScreenBorder = true, so we always clip (and report "was clipped")
    float ClipToScreenFactor = GetStepScreenFactorToClipAtScreenEdge(RayStartScreen.xy, Ray.RayStepScreen.xy);
    bRayWasClipped = true;
    Ray.RayStepScreen *= ClipToScreenFactor;

    const float SlopeCompareToleranceScale = 4.0; // SSR path in UE (see `RayCast()` in SSRTRayCast.ush)
    if (IsOrthoProjection())
    {
        Ray.CompareTolerance = max(0.0, (RayStartScreen.z - RayDepthScreen.z) * SlopeCompareToleranceScale);
    }
    else
    {
        Ray.CompareTolerance = max(abs(Ray.RayStepScreen.z), (RayStartScreen.z - RayDepthScreen.z) * SlopeCompareToleranceScale);
    }

    return Ray;
}

// -----------------------------------------------------------------------------
// HZB ray marching (from `CastScreenSpaceRay()` in SSRTRayCast.ush)
// -----------------------------------------------------------------------------

struct RayHitResult
{
    bool  hit;
    vec3  hitUVz;   // xy in SceneTexture UV space after conversion (UE: OutHitUVz.xy)
    float level;    // last mip level (mostly diagnostic)
};

RayHitResult CastScreenSpaceRay_ActivePath(
    SSRRay Ray,
    float Roughness,
    float StepOffset,
    vec4 InHZBUvFactorAndInvFactor)
{
    // Active path constants:
    const uint  NumSteps     = 16u;  // SSR_QUALITY == 2
    const float StartMip     = 1.0;  // UE RayCast() passes StartMipLevel=1
    const float FarDepthValue = 0.0; // inferred from decompile (reject hits where sampled depth==0)

    // RayStartUVz / RayStepUVz are in HZB UV atlas space (UE: multiplied by InHZBUvFactorAndInvFactor.xy)
    vec3 RayStartUVz = vec3((Ray.RayStartScreen.xy * vec2(0.5, -0.5) + vec2(0.5)) * InHZBUvFactorAndInvFactor.xy,
                            Ray.RayStartScreen.z);
    vec3 RayStepUVz  = vec3((Ray.RayStepScreen.xy  * vec2(0.5, -0.5)) * InHZBUvFactorAndInvFactor.xy,
                            Ray.RayStepScreen.z);

    const float Step = 1.0 / float(NumSteps);
    float CompareTolerance = Ray.CompareTolerance * Step;

    // UE: RayStepUVz *= Step; RayUVz = RayStartUVz + RayStepUVz * StepOffset;
    RayStepUVz *= Step;
    vec3 RayUVz = RayStartUVz + RayStepUVz * StepOffset;

    float Level = StartMip;
    float LastDiff = 0.0;

    bool foundHit = false;
    uint hitBatchStart = 0u;
    vec4 MultipleSampleDepthDiff = vec4(0.0);
    bvec4 bMultipleSampleHit = bvec4(false);

    // UE batches in groups of 4 (SSRT_SAMPLE_BATCH_SIZE)
    for (uint i = 0u; i < NumSteps; i += 4u)
    {
        // Sample positions (active path: linear stepping, not cone trace)
        vec2 SamplesUV[4];
        vec4 SamplesZ;
        vec4 SamplesMip;

        for (uint j = 0u; j < 4u; ++j)
        {
            float s = float(i) + float(j + 1u);
            SamplesUV[j] = RayUVz.xy + s * RayStepUVz.xy;
            SamplesZ[j]  = RayUVz.z  + s * RayStepUVz.z;
        }

        // Mip schedule (matches UE + decompile):
        // Samples 0..1 use Level, then Level += 0.5*Roughness
        // Samples 2..3 use new Level, then Level += 0.5*Roughness
        SamplesMip.xy = vec2(Level);
        Level += (8.0 / float(NumSteps)) * Roughness; // 0.5 * Roughness
        SamplesMip.zw = vec2(Level);
        Level += (8.0 / float(NumSteps)) * Roughness; // another 0.5 * Roughness

        // Sample HZB depth
        vec4 SampleDepth;
        SampleDepth.x = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV[0], SamplesMip.x).r;
        SampleDepth.y = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV[1], SamplesMip.y).r;
        SampleDepth.z = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV[2], SamplesMip.z).r;
        SampleDepth.w = textureLod(sampler2D(HZBTexture, VulkanGlobalPointClampedSampler), SamplesUV[3], SamplesMip.w).r;

        MultipleSampleDepthDiff = SamplesZ - SampleDepth;

        // UE: abs(diff + tol) < tol  AND SampleDepth != FarDepthValue
        bvec4 bNearEnough = lessThan(abs(MultipleSampleDepthDiff + vec4(CompareTolerance)), vec4(CompareTolerance));
        bvec4 bNotFar     = notEqual(SampleDepth, vec4(FarDepthValue));
        bMultipleSampleHit = bNearEnough && bNotFar;

        foundHit = foundHit || any(bMultipleSampleHit);

        if (foundHit)
        {
            hitBatchStart = i;
            break;
        }

        LastDiff = MultipleSampleDepthDiff.w;
    }

    vec3 OutHitUVz;

    if (foundHit)
    {
        // Active SSR refinement (see `#else // SSR` block in SSRTRayCast.ush):
        float DepthDiff0 = MultipleSampleDepthDiff[2];
        float DepthDiff1 = MultipleSampleDepthDiff[3];
        float Time0 = 3.0;

        if (bMultipleSampleHit[2])
        {
            DepthDiff0 = MultipleSampleDepthDiff[1];
            DepthDiff1 = MultipleSampleDepthDiff[2];
            Time0 = 2.0;
        }
        if (bMultipleSampleHit[1])
        {
            DepthDiff0 = MultipleSampleDepthDiff[0];
            DepthDiff1 = MultipleSampleDepthDiff[1];
            Time0 = 1.0;
        }
        if (bMultipleSampleHit[0])
        {
            DepthDiff0 = LastDiff;
            DepthDiff1 = MultipleSampleDepthDiff[0];
            Time0 = 0.0;
        }

        // Exact match to UE/decompile: add the batch base iteration (i) to local 0..3 time.
        Time0 += float(hitBatchStart);
        float Time1 = Time0 + 1.0;

        float TimeLerp = saturate(DepthDiff0 / (DepthDiff0 - DepthDiff1));
        float IntersectTime = Time0 + TimeLerp;

        OutHitUVz = RayUVz + RayStepUVz * IntersectTime;
    }
    else
    {
        // Miss: UE would return a conservative position; in the active path we later gate on `hit` anyway.
        OutHitUVz = RayUVz;
    }

    // Convert from HZB atlas UV space -> SceneTexture UV space (UE tail of CastScreenSpaceRay())
    // OutHitUVz.xy *= InvFactor; then ScreenPos conversion; then apply View.ScreenPositionScaleBias.
    OutHitUVz.xy *= InHZBUvFactorAndInvFactor.zw;
    OutHitUVz.xy = OutHitUVz.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    OutHitUVz.xy = OutHitUVz.xy * View.View_ScreenPositionScaleBias.xy + View.View_ScreenPositionScaleBias.wz;

    RayHitResult r;
    r.hit = foundHit;
    r.hitUVz = OutHitUVz;
    r.level = Level;
    return r;
}

// -----------------------------------------------------------------------------
// Noise (UE: InterleavedGradientNoise) – matches the decompiled expression
// -----------------------------------------------------------------------------

float InterleavedGradientNoise_ActivePath(vec2 pixelXY, uint frameIndexMod8)
{
    // Decompile:
    // fract(52.9829177856 * fract(dot(pixel + (vec2(32.665, 11.815) * frame), vec2(0.06711056, 0.0058371499))))
    vec2 p = pixelXY + (vec2(32.66500091552734, 11.8149995803833) * float(frameIndexMod8));
    float d = dot(p, vec2(0.0671105608344078, 0.005837149918079376));
    return fract(52.98291778564453 * fract(d));
}

// -----------------------------------------------------------------------------
// Main (maps to `ScreenSpaceReflectionsPS` single-ray path)
// -----------------------------------------------------------------------------

void main()
{
    // 1) Compute UVs used by SceneTextures (UE: UV = SvPosition.xy * View.BufferSizeAndInvSize.zw)
    vec2 bufferUV = gl_FragCoord.xy * View.View_BufferSizeAndInvSize.zw;

    // 2) Compute ScreenPos in UE space ([-1..1], y flipped), from ViewRect min/size (UE: ViewportUVToScreenPos)
    vec2 viewportUV = (gl_FragCoord.xy - View.View_ViewRectMin.xy) * View.View_ViewSizeAndInvSize.zw;
    vec2 screenPos = vec2(2.0 * viewportUV.x - 1.0, 1.0 - 2.0 * viewportUV.y);

    // 3) Read device Z and convert to scene depth (UE: SampleDeviceZFromSceneTextures + ConvertFromDeviceZ)
    float deviceZ = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).r;
    float sceneDepth = ConvertFromDeviceZ(deviceZ);

    // 4) Read gbuffer data needed for SSR:
    //    - normal (GBufferA, decoded from [0..1] -> [-1..1])
    //    - roughness (GBufferB.z)
    //    - shading model id (packed in GBufferB.w)
    vec3 encodedNormal = textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).xyz;
    vec3 N = normalize(encodedNormal * 2.0 - 1.0);

    vec4 gbufB = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);
    float Roughness = gbufB.z;

    uint packedB = uint(gbufB.w * 255.0 + 0.5);
    uint ShadingModelID = packedB & 15u; // matches decompile

    // Decompile condition for whether GBufferD is valid/used:
    // it treats shading model ids in [2..9] as "has custom data".
    bool HasCustomData = (ShadingModelID >= 2u) && (ShadingModelID <= 9u);
    vec4 gbufD = HasCustomData
        ? textureLod(sampler2D(GBufferDTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0)
        : vec4(0.0);

    // Clear coat roughness override (UE: GetRoughness(), SHADINGMODELID_CLEAR_COAT)
    // Decompile gates this on ShadingModelID == 4 and uses GBufferD.x (ClearCoat) and GBufferD.y (ClearCoatRoughness).
    if (ShadingModelID == 4u)
    {
        float ClearCoat = gbufD.x;
        float ClearCoatRoughness = gbufD.y;
        Roughness = mix(Roughness, ClearCoatRoughness, ClearCoat);
    }

    // 5) Roughness fade (UE: GetRoughnessFade)
    float RoughnessFade = min(Roughness * Globals.SSRParams.y + 2.0, 1.0);

    // Early out (UE: if RoughnessFade <= 0 || bNoMaterial)
    bool bNoMaterial = (ShadingModelID == 0u);
    if (RoughnessFade <= 0.0 || bNoMaterial)
    {
        outColor = vec4(0.0);
        return;
    }

    // 6) Reconstruct translated world position (UE: PositionTranslatedWorld)
    // Decompile matches:
    // vec4 w = View.ScreenToTranslatedWorld * vec4(screenPos * sceneDepth (persp), sceneDepth, 1)
    vec2 perspXY = screenPos * sceneDepth;
    bool ortho = IsOrthoProjection();
    vec2 clipXY = ortho ? screenPos : perspXY;
    vec3 PositionTranslatedWorld = (View.View_ScreenToTranslatedWorld * vec4(clipXY, sceneDepth, 1.0)).xyz;

    // 7) View direction V (UE: V = -GetCameraVectorFromTranslatedWorldPosition(PositionTranslatedWorld))
    // Decompile effectively uses camera->point for perspective and ViewForward for ortho.
    vec3 cameraToPoint = normalize(PositionTranslatedWorld - View.View_TranslatedWorldCameraOrigin);
    vec3 viewDir = ortho ? View.View_ViewForward : cameraToPoint;

    // 8) Optional anisotropy normal modification (active in this capture).
    //
    // In UE, this corresponds to the `#if SUPPORTS_ANISOTROPIC_MATERIALS` block calling
    // ModifyGGXAnisotropicNormalRoughness(...). The decompiled shader contains the
    // *post-lowering* math, so we reproduce it directly.
    //
    // Sources (decompile naming):
    // - GBufferF.xyz stores a direction vector (decoded to [-1..1]) used for the anisotropic basis
    // - GBufferF.w stores an anisotropy scalar in [-1..1]
    vec4 gbufF = textureLod(sampler2D(GBufferFTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);

    // Decompile gate: (((packedB >> 4) & 15) & 1) != 0
    bool HasAnisotropy = ((((packedB >> 4u) & 15u) & 1u) != 0u);
    vec3 AnisoDir = HasAnisotropy ? normalize(gbufF.xyz * 2.0 - vec3(1.0)) : vec3(0.0);
    float Aniso = HasAnisotropy ? (gbufF.w * 2.0 - 1.0) : 0.0;

    // Clear coat fade for anisotropy (UE comment: top layer assumed non-anisotropic).
    if (ShadingModelID == 4u)
    {
        float ClearCoat = gbufD.x;
        Aniso = mix(Aniso, 0.0, ClearCoat);
    }

    vec3 N_ssr = N;
    float anisoAbs = abs(Aniso);
    if (anisoAbs > 0.0)
    {
        // Decompile:
        // basis = (Aniso >= 0) ? AnisoDir : normalize(cross(N, AnisoDir))
        vec3 basis = (Aniso >= 0.0) ? AnisoDir : normalize(cross(N, AnisoDir));

        // Decompile:
        // target = cross(cross(basis, -viewDir), basis)
        vec3 target = cross(cross(basis, -viewDir), basis);

        // Fade amount: abs(Aniso) * clamp(5*Roughness, 0..1)
        float fade = anisoAbs * clamp(5.0 * Roughness, 0.0, 1.0);
        N_ssr = normalize(mix(N, target, vec3(fade)));
    }

    // 9) Single ray direction (active path: non-glossy)
    vec3 L = reflect(viewDir, N_ssr);

    // 10) Ray cast against HZB (UE: RayCast -> CastScreenSpaceRay)
    bool rayWasClipped;
    SSRRay ray = InitScreenSpaceRayFromWorldSpace_ActivePath(
        PositionTranslatedWorld,
        L,
        /* WorldTMax */ sceneDepth,
        /* SceneDepth */ sceneDepth,
        rayWasClipped);

    // Active path StepOffset: InterleavedGradientNoise - 0.5
    float StepOffset = InterleavedGradientNoise_ActivePath(gl_FragCoord.xy, View.View_StateFrameIndexMod8) - 0.5;

    RayHitResult hit = CastScreenSpaceRay_ActivePath(
        ray,
        Roughness,
        StepOffset,
        Globals.HZBUvFactorAndInvFactor);

    if (!hit.hit)
    {
        outColor = vec4(0.0);
        return;
    }

    // 11) Reproject hit to previous frame and apply vignette (UE: ReprojectHit + ComputeHitVignetteFromScreenPos)
    vec2 ThisScreen = (hit.hitUVz.xy - View.View_ScreenPositionScaleBias.wz) / View.View_ScreenPositionScaleBias.xy;
    vec4 ThisClip = vec4(ThisScreen, hit.hitUVz.z, 1.0);

    // Decompile multiplies as matrix * vector (column-major)
    vec4 PrevClip = View.View_ClipToPrevClip * ThisClip;
    vec2 PrevScreen = PrevClip.xy / PrevClip.w;

    // Velocity reproject override (active path uses it)
    vec4 EncVel = textureLod(sampler2D(GBufferVelocityTexture, VulkanGlobalPointClampedSampler), hit.hitUVz.xy, 0.0);
    if (EncVel.x > 0.0)
    {
        PrevScreen = ThisClip.xy - DecodeVelocityFromTexture(EncVel);
    }

    float vignette = min(ComputeHitVignetteFromScreenPos(ThisScreen), ComputeHitVignetteFromScreenPos(PrevScreen));

    // Convert PrevScreen -> PrevUV using PrevScreenPositionScaleBias (UE global)
    vec2 PrevUV = PrevScreen * Globals.PrevScreenPositionScaleBias.xy + Globals.PrevScreenPositionScaleBias.zw;

    // 12) Sample previous scene color and compose SSR output (UE tail of SSRTReflections.usf)
    vec4 sampleColor = SampleScreenColorPrevFrame(PrevUV) * vignette;

    // Apply fades/scales in the same order as the decompile/UE:
    vec4 color = sampleColor * RoughnessFade;
    color *= Globals.SSRParams.x; // intensity
    color.rgb *= Globals.PrevSceneColorPreExposureCorrection;

    outColor = color;
}


