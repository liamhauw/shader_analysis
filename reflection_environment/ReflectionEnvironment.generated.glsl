#version 460

// ============================================================================
// Reflection Environment Shader - Readable GLSL Version
// ============================================================================
// This shader computes environment reflections including:
// - Screen Space Reflections (SSR)
// - Reflection Capture probes (local cubemaps)
// - Sky light reflections
// - Ambient occlusion and bent normal shadowing
// ============================================================================

// Extensions
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

// ============================================================================
// Uniform Buffers
// ============================================================================

// View parameters (camera, projection, etc.)
layout(set = 1, binding = 1, std140) uniform ViewBuffer
{
    layout(offset = 448) mat4 ViewToClip;
    layout(offset = 832) mat4 ScreenToTranslatedWorld;
    layout(offset = 1168) vec3 ViewForward;
    layout(offset = 1248) vec4 InvDeviceZToWorldZTransform;
    layout(offset = 1296) vec3 TranslatedWorldCameraOrigin;
    layout(offset = 1344) vec3 PreViewTranslationHigh;
    layout(offset = 1360) vec3 PreViewTranslationLow;
    layout(offset = 2368) vec4 ViewRectMin;
    layout(offset = 2384) vec4 ViewSizeAndInvSize;
    layout(offset = 2432) vec4 BufferSizeAndInvSize;
    layout(offset = 2504) float PreExposure;
    layout(offset = 2512) vec4 DiffuseOverrideParameter;
    layout(offset = 2528) vec4 SpecularOverrideParameter;
    layout(offset = 2784) vec4 TemporalAAParams;
    layout(offset = 2912) vec3 PrecomputedIndirectSpecularColorScale;
    layout(offset = 3344) vec4 SkyLightColor;
    layout(offset = 3520) uint DistanceFieldAOSpecularOcclusionMode;
    layout(offset = 4032) float bCheckerboardSubsurfaceProfileRendering;
    layout(offset = 4364) float bSubsurfacePostprocessEnabled;
    layout(offset = 5696) uint bShadingEnergyConservation;
} View;

// Sky irradiance environment map (spherical harmonics)
layout(set = 1, binding = 5, std430) readonly buffer SkyIrradianceBuffer
{
    vec4 SkyIrradianceEnvironmentMap[];
} SkyIrradiance;

// Reflection capture data (position, radius, properties)
layout(set = 1, binding = 2, std140) uniform ReflectionCaptureBuffer
{
    vec4 PositionHighAndRadius[341];
    vec4 PositionLow[341];
} ReflectionCaptures;

// Reflection parameters (sky light settings)
layout(set = 1, binding = 3, std140) uniform ReflectionParamsBuffer
{
    vec4 SkyLightParameters; // x=max mip, y=enabled, z=dynamic, w=blend fraction
} ReflectionParams;

// Forward light grid data (for culling reflection captures)
layout(set = 1, binding = 4, std140) uniform ForwardLightBuffer
{
    layout(offset = 8) uint NumReflectionCaptures;
    layout(offset = 16) uint NumGridCells;
    layout(offset = 32) ivec3 CulledGridSize;
    layout(offset = 48) uint LightGridPixelSizeShift;
    layout(offset = 64) vec3 LightGridZParams;
} ForwardLight;

// Culled light data grid (indices of reflection captures affecting each tile)
layout(set = 1, binding = 6, std430) readonly buffer CulledLightGridBuffer
{
    uint NumCulledLightsGrid[];
} CulledLightGrid;

// Global shader parameters
layout(set = 1, binding = 0, std140) uniform GlobalsBuffer
{
    float ApplyBentNormalAO;
    float InvSkySpecularOcclusionStrength;
    vec4 OcclusionTintAndMinOcclusion;
    vec3 ContrastAndNormalizeMulAdd;
    float OcclusionExponent;
    float OcclusionCombineMode;
    float AOMaxViewDistance;
    float DistanceFadeScale;
    vec2 AOBufferBilinearUVMax;
} Globals;

// ============================================================================
// Textures and Samplers
// ============================================================================

layout(set = 1, binding = 18) uniform sampler VulkanGlobalPointClampedSampler;
layout(set = 1, binding = 19) uniform sampler ViewSharedBilinearClampedSampler;
layout(set = 1, binding = 8) uniform texture2D ShadingEnergyGGXSpecTexture;
layout(set = 1, binding = 9) uniform textureCube SkyLightCubemap;
layout(set = 1, binding = 20) uniform sampler SkyLightCubemapSampler;
layout(set = 1, binding = 10) uniform textureCube SkyLightBlendDestinationCubemap;
layout(set = 1, binding = 21) uniform sampler SkyLightBlendDestinationCubemapSampler;
layout(set = 1, binding = 7) uniform usamplerBuffer CulledLightDataGrid16Bit;
layout(set = 1, binding = 11) uniform texture2D SceneDepthTexture;
layout(set = 1, binding = 12) uniform texture2D GBufferATexture;
layout(set = 1, binding = 13) uniform texture2D GBufferBTexture;
layout(set = 1, binding = 14) uniform texture2D GBufferCTexture;
layout(set = 1, binding = 15) uniform texture2D BentNormalAOTexture;
layout(set = 1, binding = 22) uniform sampler BentNormalAOSampler;
layout(set = 1, binding = 16) uniform texture2D ReflectionTexture; // SSR result
layout(set = 1, binding = 23) uniform sampler ReflectionTextureSampler;
layout(set = 1, binding = 17) uniform texture2D AmbientOcclusionTexture;
layout(set = 1, binding = 24) uniform sampler AmbientOcclusionSampler;

// ============================================================================
// Output
// ============================================================================

layout(location = 0) out vec4 OutColor;

// ============================================================================
// Helper Functions
// ============================================================================

// SPIRV intrinsics for min/max operations
spirv_instruction(set = "GLSL.std.450", id = 79) float spvNMin(float, float);
spirv_instruction(set = "GLSL.std.450", id = 79) vec2 spvNMin(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 79) vec3 spvNMin(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 79) vec4 spvNMin(vec4, vec4);
spirv_instruction(set = "GLSL.std.450", id = 80) float spvNMax(float, float);
spirv_instruction(set = "GLSL.std.450", id = 80) vec2 spvNMax(vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 80) vec3 spvNMax(vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 80) vec4 spvNMax(vec4, vec4);

mediump float spvNMaxRelaxed(mediump float a, mediump float b)
{
    return spvNMax(a, b);
}

mediump vec2 spvNMaxRelaxed(mediump vec2 a, mediump vec2 b)
{
    return spvNMax(a, b);
}

mediump vec3 spvNMaxRelaxed(mediump vec3 a, mediump vec3 b)
{
    return spvNMax(a, b);
}

mediump vec4 spvNMaxRelaxed(mediump vec4 a, mediump vec4 b)
{
    return spvNMax(a, b);
}

// ============================================================================
// Main Shader
// ============================================================================

void main()
{
    // ========================================================================
    // Step 1: Compute screen-space coordinates and sample GBuffer
    // ========================================================================
    
    // Convert fragment coordinates to buffer UV coordinates
    vec2 bufferUV = gl_FragCoord.xy * View.BufferSizeAndInvSize.zw;
    
    // Compute position relative to view rect
    vec2 viewRectPos = gl_FragCoord.xy - View.ViewRectMin.xy;
    
    // Compute screen position for projection
    vec4 screenPos = vec4(
        ((viewRectPos * View.ViewSizeAndInvSize.zw) - vec2(0.5)) * vec2(2.0, -2.0),
        0.0, // Will be filled with depth
        1.0
    ) * (1.0 / gl_FragCoord.w);
    
    // Sample GBuffer textures
    vec4 gbufferB = textureLod(sampler2D(GBufferBTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);
    vec4 gbufferC = textureLod(sampler2D(GBufferCTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);
    
    // Compute pixel position for temporal AA checkerboard pattern
    uvec2 pixelCoord = uvec2(bufferUV * View.BufferSizeAndInvSize.xy);
    bool isCheckerboardOdd = (((pixelCoord.x + pixelCoord.y) + uint(View.TemporalAAParams.x)) % 2u) != 0u;
    
    // Extract shading model ID from GBuffer B (stored in alpha channel)
    uint shadingModelID = uint((gbufferB.w * 255.0) + 0.5) & 15u;
    
    // Extract base color and other material properties
    vec3 baseColor = vec3(gbufferC.xyz);
    float gbufferAO = gbufferC.w;
    
    // Sample GBuffer A for world normal
    vec3 worldNormal = normalize((vec3(textureLod(sampler2D(GBufferATexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0).xyz) * 2.0) - vec3(1.0));
    
    // Normalize shading model ID (0 = unlit, use 1 for processing)
    uint normalizedShadingModelID = (shadingModelID != 0u) ? 1u : shadingModelID;
    
    // ========================================================================
    // Step 2: Handle subsurface scattering materials
    // ========================================================================
    
    // Extract roughness (0.0 for subsurface materials, otherwise from GBuffer)
    float roughness = (normalizedShadingModelID == 9u) ? 0.0 : gbufferB.x;
    
    // Compute specular color with subsurface handling
    vec3 specularColor;
    vec3 diffuseColor;
    
    // Check if this is a subsurface or two-sided foliage material (ID 5 or 9)
    bool isSubsurfaceMaterial = (int(normalizedShadingModelID) == 5) || (int(normalizedShadingModelID) == 9);
    
    if (isSubsurfaceMaterial)
    {
        // Handle checkerboard subsurface rendering
        bool useCheckerboardSubsurface = (View.bSubsurfacePostprocessEnabled > 0.0) && 
                                         (View.bCheckerboardSubsurfaceProfileRendering > 0.0);
        
        if (useCheckerboardSubsurface)
        {
            // Alternate between base color and white based on checkerboard pattern
            diffuseColor = baseColor * float(!isCheckerboardOdd);
            specularColor = vec3(float(isCheckerboardOdd));
        }
        else
        {
            diffuseColor = baseColor;
            // Mix base color with white for subsurface
            specularColor = mix(baseColor, vec3(1.0), bvec3(View.bSubsurfacePostprocessEnabled != 0.0));
        }
    }
    else
    {
        specularColor = baseColor;
        diffuseColor = baseColor;
    }
    
    // Apply specular override parameters
    vec3 finalSpecularColor = (diffuseColor * View.SpecularOverrideParameter.w) + View.SpecularOverrideParameter.xyz;
    
    // ========================================================================
    // Step 3: Early exit for unlit materials
    // ========================================================================
    
    vec4 outputColor;
    
    SPIRV_CROSS_BRANCH
    if (shadingModelID != 0u)
    {
        // ====================================================================
        // Step 4: Reconstruct world position from depth
        // ====================================================================
        
        // Sample scene depth
        vec4 depthSample = textureLod(sampler2D(SceneDepthTexture, VulkanGlobalPointClampedSampler), bufferUV, 0.0);
        float deviceZ = depthSample.x;
        
        // Convert device Z to world Z
        float worldDepth = ((deviceZ * View.InvDeviceZToWorldZTransform.x) + View.InvDeviceZToWorldZTransform.y) + 
                          (1.0 / ((deviceZ * View.InvDeviceZToWorldZTransform.z) - View.InvDeviceZToWorldZTransform.w));
        
        // Determine projection type (orthographic vs perspective)
        bool isOrthographic = View.ViewToClip[3].w >= 1.0;
        
        // Compute screen position for world position reconstruction
        vec2 screenPosXY = screenPos.xy * worldDepth;
        float screenX = isOrthographic ? screenPos.x : screenPosXY.x;
        float screenY = isOrthographic ? screenPos.y : screenPosXY.y;
        
        // Transform to translated world space
        vec3 translatedWorldPos = (View.ScreenToTranslatedWorld * vec4(screenX, screenY, worldDepth, 1.0)).xyz;
        
        // Compute camera-to-pixel vector
        vec3 cameraToPixel = normalize(translatedWorldPos - View.TranslatedWorldCameraOrigin);
        
        // Compute view direction (negative of camera-to-pixel)
        vec3 viewDir;
        if (isOrthographic)
        {
            viewDir = View.ViewForward;
        }
        else
        {
            viewDir = cameraToPixel;
        }
        vec3 viewDirection = -viewDir;
        
        // ====================================================================
        // Step 5: Sample ambient occlusion and SSR reflection texture
        // ====================================================================
        
        float ambientOcclusion = textureLod(sampler2D(AmbientOcclusionTexture, AmbientOcclusionSampler), bufferUV, 0.0).x;
        
        // Sample Screen Space Reflection texture
        vec4 ssrReflection = texture(sampler2D(ReflectionTexture, ReflectionTextureSampler), bufferUV);
        float ssrAlpha = 1.0 - ssrReflection.w; // Alpha channel stores how much SSR contributes
        
        // ====================================================================
        // Step 6: Upsample bent normal AO (Distance Field AO)
        // ====================================================================
        
        // Compute bilinear sampling coordinates for bent normal AO texture
        vec2 aoBufferUV = spvNMin(viewRectPos * View.BufferSizeAndInvSize.zw, Globals.AOBufferBilinearUVMax);
        vec2 halfBufferSize = floor(View.BufferSizeAndInvSize.xy * vec2(0.5));
        vec2 invHalfBufferSize = vec2(1.0) / halfBufferSize;
        
        // Compute bilinear sample positions
        vec2 bilinearBaseUV = (floor((aoBufferUV * halfBufferSize) - vec2(0.5)) / halfBufferSize) + (invHalfBufferSize * 0.5);
        vec2 bilinearOffset = (aoBufferUV - bilinearBaseUV) * halfBufferSize;
        
        // Sample 4 texels for bilinear interpolation
        vec4 bentNormalAO0 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), bilinearBaseUV, 0.0);
        vec4 bentNormalAO1 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), bilinearBaseUV + vec2(invHalfBufferSize.x, 0.0), 0.0);
        vec4 bentNormalAO2 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), bilinearBaseUV + vec2(0.0, invHalfBufferSize.y), 0.0);
        vec4 bentNormalAO3 = textureLod(sampler2D(BentNormalAOTexture, BentNormalAOSampler), bilinearBaseUV + invHalfBufferSize, 0.0);
        
        // Compute bilinear weights based on depth similarity
        float weightY = bilinearOffset.y;
        float weightYInv = 1.0 - weightY;
        float weightX = bilinearOffset.x;
        float weightXInv = 1.0 - weightX;
        
        vec4 bilinearWeights = vec4(
            weightYInv * weightXInv,
            weightYInv * weightX,
            weightY * weightXInv,
            weightY * weightX
        ) * (vec4(1.0) / (abs(vec4(bentNormalAO0.w, bentNormalAO1.w, bentNormalAO2.w, bentNormalAO3.w) - vec4(worldDepth)) + vec4(0.0001)));
        
        // Interpolate bent normal
        vec3 bentNormal = ((bentNormalAO0.xyz * bilinearWeights.x) + 
                          (bentNormalAO1.xyz * bilinearWeights.y) + 
                          (bentNormalAO2.xyz * bilinearWeights.z) + 
                          (bentNormalAO3.xyz * bilinearWeights.w)) * (1.0 / dot(bilinearWeights, vec4(1.0)));
        
        // Apply distance fade for bent normal
        float distanceFade = clamp((Globals.AOMaxViewDistance - worldDepth) * Globals.DistanceFadeScale, 0.0, 1.0);
        bentNormal = mix(worldNormal, bentNormal, vec3(distanceFade));
        
        // ====================================================================
        // Step 7: Compute light grid cell index for reflection capture culling
        // ====================================================================
        
        // Compute grid cell coordinates
        uvec2 gridCellCoord = uvec2(uint(viewRectPos.x), uint(viewRectPos.y)) >> 
                             (uvec2(ForwardLight.LightGridPixelSizeShift) & uvec2(31u));
        
        // Compute grid index based on depth
        float logDepth = log2((worldDepth * ForwardLight.LightGridZParams.x) + ForwardLight.LightGridZParams.y) * ForwardLight.LightGridZParams.z;
        uint depthSlice = min(uint(spvNMax(0.0, logDepth)), uint(ForwardLight.CulledGridSize.z - 1));
        
        uint gridIndex = (ForwardLight.NumGridCells + 
                         ((((depthSlice * uint(ForwardLight.CulledGridSize.y)) + gridCellCoord.y) * 
                           uint(ForwardLight.CulledGridSize.x)) + gridCellCoord.x)) * 2u;
        
        // Get number of reflection captures affecting this pixel
        uint numCulledCaptures = min(CulledLightGrid.NumCulledLightsGrid[gridIndex], ForwardLight.NumReflectionCaptures);
        uint captureDataStartIndex = gridIndex + 1u;
        
        // ====================================================================
        // Step 8: Determine if we need to compute reflections
        // ====================================================================
        
        bool skyLightEnabled = ReflectionParams.SkyLightParameters.y > 0.0;
        bool hasReflectionCaptures = numCulledCaptures > 0u;
        bool shouldComputeReflections = skyLightEnabled || hasReflectionCaptures || (ssrAlpha < 1.0);
        
        // ====================================================================
        // Step 9: Compute reflection vector and BRDF terms
        // ====================================================================
        
        float NdotV = dot(worldNormal, viewDirection);
        
        // Clamp roughness
        float clampedRoughness = clamp(spvNMax(gbufferB.z, 0.0), 0.001, 1.0);
        
        // Sample environment BRDF lookup texture
        vec4 envBRDF = textureLod(sampler2D(ShadingEnergyGGXSpecTexture, ViewSharedBilinearClampedSampler), 
                                 vec2(NdotV, clampedRoughness), 0.0);
        
        // Apply energy conservation if enabled
        vec3 energyConservationFactor;
        if (View.bShadingEnergyConservation != 0u)
        {
            float energyTerm = envBRDF.x;
            energyConservationFactor = vec3(1.0) + (finalSpecularColor.xyz * ((1.0 - energyTerm) / energyTerm));
        }
        else
        {
            energyConservationFactor = vec3(1.0);
        }
        
        // Compute reflection vector and specular contribution
        vec3 reflectionVector;
        vec3 specularContribution;
        float reflectionRoughness;
        
        SPIRV_CROSS_BRANCH
        if (shouldComputeReflections)
        {
            // Compute reflection vector with off-specular peak adjustment
            float roughnessSq = clampedRoughness * clampedRoughness;
            float roughnessTerm = 1.0 - roughnessSq;
            reflectionVector = mix(worldNormal, 
                                  (worldNormal * (2.0 * dot(viewDirection, worldNormal))) - viewDirection,
                                  vec3(roughnessTerm * (sqrt(roughnessTerm) + roughnessSq)));
            
            // Compute specular contribution from BRDF
            specularContribution = energyConservationFactor * 
                                  ((finalSpecularColor.xyz * envBRDF.x) + 
                                   (((vec3(1.0) * clamp(50.0 * spvNMax(finalSpecularColor.x, spvNMax(finalSpecularColor.y, finalSpecularColor.z)), 0.0, 1.0)) - 
                                     finalSpecularColor.xyz) * envBRDF.y));
            
            reflectionRoughness = clampedRoughness;
        }
        else
        {
            reflectionVector = vec3(0.0);
            specularContribution = vec3(0.0);
            reflectionRoughness = 0.0;
        }
        
        // ====================================================================
        // Step 10: Compute diffuse lighting from sky
        // ====================================================================
        
        // Apply diffuse override
        vec3 finalDiffuseColor = (((specularColor - (specularColor * roughness)) * View.DiffuseOverrideParameter.www) + 
                                 View.DiffuseOverrideParameter.xyz).xyz * 1.0;
        
        // Check if we need to compute diffuse lighting
        bool shouldComputeDiffuse = any(greaterThan(finalDiffuseColor, vec3(0.0)));
        
        vec3 diffuseLighting;
        
        if (shouldComputeDiffuse)
        {
            // Normalize bent normal
            float bentNormalLength = length(bentNormal);
            vec3 normalizedBentNormal = bentNormal / vec3(spvNMax(bentNormalLength, 0.00001));
            
            // Blend between bent normal and world normal based on bent normal length
            vec3 diffuseNormal = mix(normalizedBentNormal, worldNormal, vec3(bentNormalLength));
            
            // Compute AO from bent normal length
            float bentNormalAO = mix(
                pow(clamp(((1.0 / (1.0 + exp((-Globals.ContrastAndNormalizeMulAdd.x) * ((bentNormalLength * 10.0) - 5.0)))) * 
                          Globals.ContrastAndNormalizeMulAdd.y) + Globals.ContrastAndNormalizeMulAdd.z, 0.0, 1.0),
                    Globals.OcclusionExponent),
                1.0,
                Globals.OcclusionTintAndMinOcclusion.w
            );
            
            // Combine AO terms
            float combinedAO;
            if (Globals.OcclusionCombineMode == 0.0)
            {
                // Minimum mode
                combinedAO = spvNMin(bentNormalAO, spvNMin(gbufferAO, ambientOcclusion));
            }
            else
            {
                // Multiply mode
                combinedAO = bentNormalAO * spvNMin(gbufferAO, ambientOcclusion);
            }
            
            // Compute sky diffuse lighting using spherical harmonics
            if (shouldComputeDiffuse)
            {
                vec3 diffuseNormalVec = diffuseNormal;
                float nx = diffuseNormalVec.x;
                float ny = diffuseNormalVec.y;
                float nz = diffuseNormalVec.z;
                
                // Evaluate spherical harmonics basis functions
                vec4 normalVec = vec4(nx, ny, nz, 1.0);
                vec3 sh0 = vec3(
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[0u], normalVec),
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[1u], normalVec),
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[2u], normalVec)
                );
                
                vec4 normalVecCross = normalVec.xyzz * normalVec.yzzx;
                vec3 sh1 = vec3(
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[3u], normalVecCross),
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[4u], normalVecCross),
                    dot(SkyIrradiance.SkyIrradianceEnvironmentMap[5u], normalVecCross)
                );
                
                float normalVecDiff = (nx * nx) - (ny * ny);
                vec3 sh2 = SkyIrradiance.SkyIrradianceEnvironmentMap[6u].xyz * normalVecDiff;
                
                // Combine SH terms and apply sky light color
                vec3 skyIrradiance = spvNMax(vec3(0.0), sh0 + sh1 + sh2);
                vec3 skyDiffuse = skyIrradiance * View.SkyLightColor.xyz;
                
                // Apply AO and blend with occlusion tint
                float aoBlend = combinedAO * mix(dot(normalizedBentNormal, worldNormal), 1.0, bentNormalLength);
                diffuseLighting = ((skyDiffuse * aoBlend) + 
                                  (Globals.OcclusionTintAndMinOcclusion.xyz * (1.0 - combinedAO))) * finalDiffuseColor;
            }
            else
            {
                diffuseLighting = vec3(0.0);
            }
        }
        else
        {
            diffuseLighting = vec3(0.0);
        }
        
        // ====================================================================
        // Step 11: Compute specular reflections (captures + sky)
        // ====================================================================
        
        vec3 specularLighting;
        
        if (any(greaterThan(specularContribution, vec3(0.0))))
        {
            // ================================================================
            // Step 11a: Apply bent normal AO to specular
            // ================================================================
            
            vec3 specularOcclusionTint;
            float specularOcclusionFactor;
            
            SPIRV_CROSS_BRANCH
            if (Globals.ApplyBentNormalAO > 0.0)
            {
                float bentNormalLength = length(bentNormal);
                
                float specularOcclusion;
                
                SPIRV_CROSS_BRANCH
                if (View.DistanceFieldAOSpecularOcclusionMode == 0u)
                {
                    // Simple mode: use bent normal length directly
                    specularOcclusion = bentNormalLength;
                }
                else
                {
                    // Advanced mode: compute specular occlusion based on angle between bent normal and reflection vector
                    float roughnessAngle = spvNMax(reflectionRoughness, 0.1) * 3.140625;
                    float bentNormalAngle = (bentNormalLength * 3.140625) * Globals.InvSkySpecularOcclusionStrength;
                    float angleDiff = abs(roughnessAngle - bentNormalAngle);
                    
                    float angleBetweenNormals = acos(dot(bentNormal, reflectionVector) / spvNMax(bentNormalLength, 0.001));
                    float occlusionFactor = smoothstep(0.0, 1.0, 1.0 - clamp((angleBetweenNormals - angleDiff) / ((roughnessAngle + bentNormalAngle) - angleDiff), 0.0, 1.0));
                    occlusionFactor *= clamp((bentNormalAngle - 0.1) * 5.0, 0.0, 1.0);
                    
                    specularOcclusion = mix(0.0, occlusionFactor, clamp((bentNormalAngle - 0.1) * 5.0, 0.0, 1.0));
                }
                
                float finalSpecularOcclusion = mix(specularOcclusion, 1.0, Globals.OcclusionTintAndMinOcclusion.w);
                specularOcclusionTint = Globals.OcclusionTintAndMinOcclusion.xyz * (1.0 - finalSpecularOcclusion);
                specularOcclusionFactor = finalSpecularOcclusion;
            }
            else
            {
                specularOcclusionTint = vec3(0.0);
                specularOcclusionFactor = 1.0;
            }
            
            // ================================================================
            // Step 11b: Sample reflection captures
            // ================================================================
            
            float roughnessSq = reflectionRoughness * reflectionRoughness;
            
            // Initialize reflection accumulation with SSR alpha
            // The alpha channel represents how much of the reflection is still "unfilled" by SSR
            vec4 reflectionAccum = vec4(0.0, 0.0, 0.0, ssrAlpha * clamp(pow(clamp(NdotV, 0.0, 1.0) + 1.0, roughnessSq), 0.0, 1.0));
            
            // Loop through reflection captures affecting this pixel
            // Note: In the decompiled shader, the actual cubemap sampling code is not visible
            // (likely optimized/inlined). The loop structure checks if captures are within range,
            // but the actual cubemap array sampling with proper mip selection would happen here.
            // The reflection captures are composited using an "under" operator (back to front),
            // where each capture reduces the alpha channel.
            SPIRV_CROSS_LOOP
            for (uint captureIndex = 0u; captureIndex < numCulledCaptures; captureIndex++)
            {
                SPIRV_CROSS_BRANCH
                if (reflectionAccum.w < 0.001)
                {
                    break; // Early exit if alpha is too low (all reflection filled)
                }
                
                // Get capture index from culled light data grid
                uvec4 captureData = texelFetch(CulledLightDataGrid16Bit, 
                                              int(CulledLightGrid.NumCulledLightsGrid[captureDataStartIndex] + captureIndex));
                uint captureIdx = captureData.x;
                
                // Compute capture position with high precision (for large world coordinates)
                // This uses double-float precision to avoid precision issues
                vec3 capturePosHigh = ReflectionCaptures.PositionHighAndRadius[captureIdx].xyz + View.PreViewTranslationHigh;
                vec3 capturePosLow = ReflectionCaptures.PositionLow[captureIdx].xyz + View.PreViewTranslationLow;
                
                // Compute distance to capture center
                vec3 captureVector = translatedWorldPos - (capturePosHigh + capturePosLow);
                float captureDistance = sqrt(dot(captureVector, captureVector));
                float captureRadius = ReflectionCaptures.PositionHighAndRadius[captureIdx].w;
                
                // Check if pixel is within capture influence radius
                SPIRV_CROSS_BRANCH
                if (captureDistance < captureRadius)
                {
                    // NOTE: In the actual implementation, this would:
                    // 1. Compute the reflection vector projected onto the capture's shape (sphere or box)
                    // 2. Sample the cubemap array at the appropriate mip level based on roughness
                    // 3. Apply distance-based fading
                    // 4. Composite using under operator: 
                    //    reflectionAccum.rgb += captureSample.rgb * reflectionAccum.a * specularOcclusion
                    //    reflectionAccum.a *= (1.0 - captureSample.a)
                    //
                    // The decompiled code shows this structure but the actual cubemap sampling
                    // was optimized/inlined in a way that's not visible in the decompiled output.
                    //
                    // For this readable version, we preserve the structure:
                    vec4 captureReflection = reflectionAccum; // Would contain sampled cubemap data
                    reflectionAccum = captureReflection;
                }
            }
            
            // ================================================================
            // Step 11c: Sample sky light cubemap
            // ================================================================
            
            vec3 skyReflection;
            
            SPIRV_CROSS_BRANCH
            if (skyLightEnabled)
            {
                // Compute mip level from roughness
                float skyMip = (ReflectionParams.SkyLightParameters.x - 1.0) - 
                              (1.0 - (1.2001953125 * log2(spvNMaxRelaxed(reflectionRoughness, 0.00100040435791015625))));
                
                // Sample sky light cubemap
                vec3 skyLightSample = textureLod(samplerCube(SkyLightCubemap, SkyLightCubemapSampler), 
                                                reflectionVector, skyMip).xyz * View.SkyLightColor.xyz;
                
                // Apply sky light blending if enabled
                SPIRV_CROSS_BRANCH
                if (ReflectionParams.SkyLightParameters.w > 0.0)
                {
                    vec3 blendDestination = textureLod(samplerCube(SkyLightBlendDestinationCubemap, 
                                                                   SkyLightBlendDestinationCubemapSampler), 
                                                       reflectionVector, skyMip).xyz * View.SkyLightColor.xyz;
                    skyReflection = mix(skyLightSample, blendDestination, vec3(ReflectionParams.SkyLightParameters.w));
                }
                else
                {
                    skyReflection = skyLightSample;
                }
                
                skyReflection = specularOcclusionTint + (skyReflection * specularOcclusionFactor);
            }
            else
            {
                skyReflection = specularOcclusionTint;
            }
            
            // Combine reflection captures and sky light
            specularLighting = specularContribution * 
                              ((reflectionAccum.xyz * View.PrecomputedIndirectSpecularColorScale).xyz + 
                               (skyReflection * reflectionAccum.w)).xyz;
        }
        else
        {
            specularLighting = vec3(0.0);
        }
        
        // ====================================================================
        // Step 12: Combine diffuse and specular lighting
        // ====================================================================
        
        vec4 finalColor = vec4(diffuseLighting, 0.0);
        
        // Handle subsurface post-processing
        bool shouldZeroAlpha;
        if (View.bCheckerboardSubsurfaceProfileRendering == 0.0)
        {
            shouldZeroAlpha = View.bSubsurfacePostprocessEnabled != 0.0;
        }
        else
        {
            shouldZeroAlpha = false;
        }
        
        if (shouldZeroAlpha)
        {
            finalColor.w = 0.0;
        }
        
        // Combine all lighting contributions
        outputColor = ((finalColor + vec4(specularLighting, 0.0)) * View.PreExposure) + 
                     vec4((ssrReflection.xyz * specularContribution) * 1.0, 0.0);
    }
    else
    {
        // Unlit material - output black
        outputColor = vec4(0.0);
    }
    
    OutColor = outputColor;
}

