/*
Copyright 2016 Tommi M. Tykkälä

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#define STRINGIFY(A) #A


const char *passThruVS = STRINGIFY(
            void main()                                                        \n
{
                \n
                gl_Position = gl_Vertex;                                       \n
                gl_TexCoord[0] = gl_MultiTexCoord0;                            \n
                gl_FrontColor = gl_Color;                                      \n
            }                                                                  \n
            );

const char *rgbd2DPS = STRINGIFY(
        uniform sampler2D rgbTex;                                         \n
        uniform float thresholdAlpha;
        //uniform sampler2D depthTex;                                       \n
        void main()                                                       \n
{                                                                         \n
        float4 rgbVal  = texture2D(rgbTex,   gl_TexCoord[0].xy);            \n
        //rgbVal.x += clamp(gl_TexCoord[0].x,0,1);
        //float depth    = texture2D(depthTex, gl_TexCoord[0].xy);            \n
        //float4 frag = float4(rgbVal.xyz,depth); \n
        float4 frag = rgbVal; \n
//        frag.w = alpha;
        if (thresholdAlpha > 0 && frag.w > 0) frag.w = 1.0f;
        gl_FragColor = frag;
}                                                                  \n
);


const char *vignette2DPS = STRINGIFY(
                uniform sampler2D rgbTex;\n
                uniform sampler2D vignetteTex;\n
                void main()                                                       \n
        {                                                                         \n
                float4 rgbVal0  = texture2D(rgbTex,        gl_TexCoord[0].xy);            \n
                float4 rgbVal1  = texture2D(vignetteTex,   gl_TexCoord[0].xy);            \n
                float4 frag = rgbVal0; \n
                frag.w = 1; \n
                frag.x *= clamp(rgbVal1.x+0.2f,0,1);//(rgbVal0.x+rgbVal1.x)/2.0f; \n
                frag.y *= clamp(rgbVal1.y+0.2f,0,1); \n
                frag.z *= clamp(rgbVal1.z+0.2f,0,1); \n
                //frag.w *= rgbVal1.w; \n
                gl_FragColor = frag; \n
        }                                                                  \n
        );

// blend two RGBZ views using blending weights
// in case image0 has hole, use image1
// in case image1 has hole, use image0
// both have valid values:
//     1) compare depth, if discrepancy is big enough, choose the closer one
//     2) with similar depth, blend using the given weights
// output: normalized RGBZ

const char *rgba2DPS = STRINGIFY(
        uniform sampler2D rgbdTex0;   \n
        uniform sampler2D rgbdTex1;   \n
        uniform sampler2D weightTex0;   \n
        uniform sampler2D weightTex1;   \n
        uniform     float zepsilon;  \n
        void main()
{                                                                         \n
        float4 rgbVal0 = texture2D(rgbdTex0,   gl_TexCoord[0].xy);            \n
        float4 rgbVal1 = texture2D(rgbdTex1,   gl_TexCoord[0].xy);            \n
        float4 weightVal0 = texture2D(weightTex0,   gl_TexCoord[0].xy);            \n
        float4 weightVal1 = texture2D(weightTex1,   gl_TexCoord[0].xy);            \n
        //float weightSum = 0.0f;
        float diff = rgbVal1.w - rgbVal0.w;\n
        if (weightVal1.x == 0.0f) {
            gl_FragColor = rgbVal0;\n
        //    weightSum = weightVal0.x;
        } else if (weightVal0.x == 0.0f) {
            gl_FragColor = rgbVal1;\n
        //    weightSum = weightVal1.x;
        } else if  (diff > zepsilon) {\n
            gl_FragColor = rgbVal0;\n
        //    weightSum = weightVal0.x;
        } else if (diff < -zepsilon) {\n
            gl_FragColor = rgbVal1;\n
        //    weightSum = weightVal1.x;
        } else {\n
            gl_FragColor = (weightVal0.x*rgbVal0+weightVal1.x*rgbVal1)/(weightVal0.x+weightVal1.x);\n
        //    weightSum = weightVal0.x+weightVal1.x;
        } \n
//        gl_FragColor.w = weightSum; \n
}                                                                  \n
);

// blend two RGBZ views using blending weights
// in case image0 has hole, use image1
// in case image1 has hole, use image0
// both have valid values:
//     1) compare depth, if discrepancy is big enough, choose the closer one
//     2) with similar depth, blend using the given weights
// output: normalized RGBZ
const char *weightFusionPS = STRINGIFY(
        uniform sampler2D rgbdTex0;   \n
        uniform sampler2D rgbdTex1;   \n
        uniform sampler2D weightTex0;   \n
        uniform sampler2D weightTex1;   \n
        uniform     float zepsilon;  \n
        void main()
{                                                                         \n
        float4 rgbVal0 = texture2D(rgbdTex0,   gl_TexCoord[0].xy);            \n
        float4 rgbVal1 = texture2D(rgbdTex1,   gl_TexCoord[0].xy);            \n
        float4 weightVal0 = texture2D(weightTex0,   gl_TexCoord[0].xy);            \n
        float4 weightVal1 = texture2D(weightTex1,   gl_TexCoord[0].xy);            \n
        float weightSum = 0.0f;
        float diff = rgbVal1.w - rgbVal0.w;\n
        if (weightVal1.x == 0.0f) {
            weightSum = weightVal0.x;
        } else if (weightVal0.x == 0.0f) {
            weightSum = weightVal1.x;
        } else if  (diff > zepsilon) {\n
            weightSum = weightVal0.x;
        } else if (diff < -zepsilon) {\n
            weightSum = weightVal1.x;
        } else {\n
            weightSum = clamp(weightVal0.x+weightVal1.x,0,1);
        } \n
        gl_FragColor = vec4(weightSum,weightSum,weightSum,1); \n
}                                                                  \n
);

const char *weight2DPS = STRINGIFY(
        uniform sampler2D rgbdTex;   \n
        uniform float minDepth; \n
        uniform float maxDepth; \n
        uniform float initialWeight; \n
        void main()                                                       \n
{                                                                         \n
        float4 rgbVal = texture2D(rgbdTex,   gl_TexCoord[0].xy);            \n
        //float black = rgbVal.x+rgbVal.y+rgbVal.z;
        float depth = rgbVal.w*(maxDepth-minDepth)+minDepth; \n
        if (depth > minDepth) { \n
        //if (black > 1e-2f) { \n
            gl_FragColor = vec4(initialWeight,initialWeight,initialWeight,1);\n
         } else { \n
            gl_FragColor = vec4(0,0,0,1);\n
        } \n
}                                                                  \n
);

/*
const char *depth2DPS = STRINGIFY(
        uniform sampler2D depthTex;                                       \n
        void main()                                                       \n
{                                                                         \n
        float depth   = texture2D(depthTex, gl_TexCoord[0].xy);            \n
        float4 frag = float4(depth,depth,depth,1); \n
        gl_FragColor = frag;
}                                                                  \n
);*/

const char *depth2DPS = STRINGIFY(
        uniform sampler2D rgbTex;                                         \n
        uniform sampler2D depthTex;                                       \n
        uniform float flip;
        void main()                                                       \n
{                                                                         \n
        float2 texcoord; texcoord.x = gl_TexCoord[0].x;  texcoord.y = gl_TexCoord[0].y; \n
        if (flip > 0.5f) texcoord.y = 1-texcoord.y;
        float3 rgbVal = texture2D(rgbTex,   texcoord.xy);            \n
        float depth   = texture2D(depthTex, texcoord.xy);            \n
        float4 frag = float4(rgbVal.xyz,depth); \n
        gl_FragColor = frag;
            //        gl_FragColor.x = 0; \n //depth.xxxx;
//        gl_FragColor.y = 0; \n //depth.xxxx;
//        gl_FragColor.z = 0; \n //depth.xxxx;
//        gl_FragColor.w = depth; \n //depth.xxxx;
                 //gl_FragColor.x = color.x;
}                                                                  \n
);

// floor shader
const char *fixedTexRGBDVS = STRINGIFY(
			varying vec4 vertexPosEye;     // vertex position in novel view point  \n
void main()                                                  \n
{
    \n
			 gl_Position       = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz,1);  \n
			 gl_TexCoord[0]    = gl_MultiTexCoord0;                      \n
			 vertexPosEye      = gl_ModelViewMatrix * vec4(gl_Vertex.xyz,1);           \n
//             gl_FrontColor = gl_Color;                                \n
}                                                            \n
);

const char *fixedTexRGBDPS = STRINGIFY(
            uniform sampler2D tex; \n
            uniform float minDepth; \n
            uniform float maxDepth; \n
            varying vec4 vertexPosEye;                                                    \n
            void main() {                                                                 \n
				vec4 colorMap  = texture2D(tex, gl_TexCoord[0].xy);                       \n
				gl_FragColor.xyz = colorMap.xyz;                                          \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
            }                                                                             \n
);

// floor shader
const char *autoTexRGBDVS = STRINGIFY(
			varying vec4 vertexPosEye;     // vertex position in novel view point  \n
			varying vec2 texCoordRGB;      // vertex position in rgb view  \n
			uniform mat4 Tbaseline;        // baseline transform
			uniform float fx;
			uniform float fy;
			uniform float cx;
			uniform float cy;
			uniform float kc0;
			uniform float kc1;
			uniform float kc2;
			uniform float kc3;
			uniform float kc4;
void main()                                                  \n
{
	\n
             // fetch point in computer vision coordinates (right-handed, +y down, +z to screen)
             vec4 vertexPos = vec4(gl_Vertex.xyz,1);
             // convert point to opengl coordinates by flipping y and z
             vec4 vertexPosOpenGL = vec4(gl_Vertex.x,-gl_Vertex.y,-gl_Vertex.z,1);
             // apply opengl transforms
             gl_Position       = gl_ModelViewProjectionMatrix*vertexPosOpenGL;  \n
             vertexPosEye      = gl_ModelViewMatrix * vertexPosOpenGL;           \n
             // apply baseline transform to computer vision coordinates:
             vec4 vertexPosEyeRGB   = Tbaseline * vertexPos;           \n
			 vertexPosEyeRGB.x     /= vertexPosEyeRGB.z; \n
             vertexPosEyeRGB.y     /= vertexPosEyeRGB.z; \n

             // generate distorted coordinates
			 float dx = vertexPosEyeRGB.x*vertexPosEyeRGB.x; \n
			 float dy = vertexPosEyeRGB.y*vertexPosEyeRGB.y; \n
			 float r2 = dx+dy;  \n
			 float r4 = r2*r2; \n
			 float r6 = r4*r2; \n
			 float radialDist = 1 + kc0*r2 + kc1*r4 + kc4*r6; \n
             texCoordRGB = clamp(vec2((fx * radialDist * vertexPosEyeRGB.x+cx),(fy * radialDist * vertexPosEyeRGB.y+cy)),0,1);
//             gl_FrontColor = gl_Color;                                \n
}                                                            \n
);

const char *autoTexRGBDPS = STRINGIFY(
			uniform sampler2D tex; \n
			uniform float minDepth; \n
			uniform float maxDepth; \n
			varying vec4 vertexPosEye;                                                    \n
			varying vec2 texCoordRGB;                                                    \n
 void main() {                                                                 \n
                vec4 colorMap  = texture2D(tex, texCoordRGB.xy);                          \n
                gl_FragColor.xyz = colorMap.xyz;                                          \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
                //gl_FragColor.w = clamp((fabs(vertexPosEye.z)-minDepth)/(maxDepth-minDepth),0,1.0f); \n
             }                                                                             \n
);

// floor shader
const char *gouraudRGBDVS = STRINGIFY(
            uniform vec3 lightDir; \n
            varying vec4 vertexPosEye;     // vertex position in novel view point  \n
            varying float diffuse;         // diffuse lighting
void main()                                                  \n
{
    \n
             // fetch point in computer vision coordinates (right-handed, +y down, +z to screen)
             vec4 vertexPos = vec4(gl_Vertex.xyz,1);
             // convert point to opengl coordinates by flipping y and z
             vec4 vertexPosOpenGL = vec4(gl_Vertex.x,-gl_Vertex.y,-gl_Vertex.z,1);
             gl_Position       = gl_ModelViewProjectionMatrix*vertexPosOpenGL;  \n
             vertexPosEye      = gl_ModelViewMatrix*vertexPosOpenGL;           \n
             diffuse = clamp(dot(gl_Normal.xyz,lightDir),0,1);
}                                                            \n
);

const char *gouraudRGBDPS = STRINGIFY(
            uniform float minDepth; \n
            uniform float maxDepth; \n
            varying vec4 vertexPosEye;                                                    \n
            varying float diffuse;         // diffuse lighting
void main() {                                                                 \n
                gl_FragColor.xyz = 0.1f+diffuse*0.9f;//colorMap.xyz;                                          \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
            }                                                                             \n
);


const char *texPS = STRINGIFY(
            uniform sampler2D rgbTex; \n
			varying vec2 texCoords; \n
void main() {                                                                 \n
                vec4 colorMap  = texture2D(rgbTex, texCoords);                          \n
                gl_FragColor.xyz = colorMap.xyz;                                          \n
                gl_FragColor.w = 1.0; \n
             }                                                                             \n
);


// standard vertex transform
const char *colorVS = STRINGIFY(
varying vec2 texCoords;
void main()                                                  \n
{
    \n
             // convert point to opengl coordinates by flipping y and z
             //vec4 vertexPosOpenGL = vec4(gl_Vertex.x,gl_Vertex.z,-gl_Vertex.y,1);
             vec4 vertexPosOpenGL = vec4(gl_Vertex.x,gl_Vertex.y,gl_Vertex.z,1);
                // apply opengl transforms
             gl_Position       = gl_ModelViewProjectionMatrix*vertexPosOpenGL;  \n
             gl_FrontColor = gl_Color;                                \n
             texCoords = gl_MultiTexCoord0.xy;
}                                                            \n
);

const char *colorDepthPS = STRINGIFY(
uniform float minDepth; \n
uniform float maxDepth; \n
varying vec4 vertexPosEye;                                                    \n
void main() {                                                                 \n
                gl_FragColor.xyz = gl_Color.xyz;                                         \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
             }                                                                             \n
);

const char *depthDepthPS = STRINGIFY(
uniform float minDepth; \n
uniform float maxDepth; \n
varying vec4 vertexPosEye;                                                    \n
void main() {                                                                 \n
                float depth = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f);
                gl_FragColor.xyz = depth;                                         \n
                gl_FragColor.w = 1.0f;
            }                                                                             \n
);

// floor shader
const char *floorVS = STRINGIFY(
            varying vec4 vertexPosEye;  // vertex position in eye space  \n
void main()                                                  \n
{
    \n
             gl_Position    = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz,1);  \n
             gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
             vertexPosEye   = gl_ModelViewMatrix *vec4(gl_Vertex.xyz,1);           \n
//             gl_FrontColor = gl_Color;                                \n
}                                                            \n
);

const char *floorPS = STRINGIFY(
            uniform sampler2D tex; \n
            uniform float minDepth; \n
            uniform float maxDepth; \n
            varying vec4 vertexPosEye;                                                    \n
            void main() {                                                                 \n
                vec4 colorMap  = texture2D(tex, gl_TexCoord[0].xy);                       \n
                gl_FragColor.xyz = colorMap.xyz;                                          \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
            }                                                                             \n
);
