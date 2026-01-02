Shader analysis

Generate a readable GLSL shader based on the two shaders in the ssr folder. `.rdc.glsl` is the result of decompiling RenderDoc, and `.usf` is the UE source code. Requirements:
1. The UE source code contains multiple code paths; the decompiled result should use the actual path used, and the generated shader should use the actual path used. 
2. The generated shader should have a clear structure and detailed comments.
