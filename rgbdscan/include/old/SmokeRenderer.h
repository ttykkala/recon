/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Smoke particle renderer with volumetric shadows

#ifndef SMOKE_RENDERER_H
#define SMOKE_RENDERER_H

#include <GL/glew.h>
#include "framebufferObject.h"
#include "GLSLProgram.h"
#include "nvMath.h"


using namespace nv;

class SmokeRenderer
{
    public:
        SmokeRenderer();
        ~SmokeRenderer();

        enum DisplayMode
        {
            POINTS,
            SPRITES,
            VOLUMETRIC,
            NUM_MODES
        };

        enum Target
        {
            LIGHT_BUFFER,
            SCENE_BUFFER
        };

        void setCube(vec3f origin, vec3f dim) {
            m_cubeOrigin = origin;
            m_cubeDim    = dim;
        }

        void setCameraPos(float x, float y, float z) {
            m_cameraPos.x = x;
            m_cameraPos.y = y;
            m_cameraPos.z = z;
            printf("setPos: %f %f %f\n",m_cameraPos.x,m_cameraPos.y,m_cameraPos.z);
        }
        void setDisplayMode(DisplayMode mode)
        {
            mDisplayMode = mode;
        }

        void setGridResolution(unsigned int x)
        {
            mNumVertices = x*x*x;
            mGridReso = x;
        }
        void setPositionBuffer(GLuint vbo)
        {
            mPosVbo = vbo;
        }
        void setCloudBuffer(GLuint vbo)
        {
            mCloudVbo = vbo;
        }
        void setPosition2dBuffer(GLuint vbo)
        {
            mPos2dVbo = vbo;
        }
        void setColorBuffer(GLuint vbo)
        {
            mColorVbo = vbo;
        }
        void setIndexBuffer(GLuint ib)
        {
            mIndexBuffer = ib;
        }

        void setPointSize(vec3f sz)
        {
            mPointSize = sz.x;
        }
        void resetView3d(int x0, int y0, int w, int h);
        void resetView2d(int x0, int y0, int w, int h);
        void setFOV(float fov)
        {
            mFov = fov;
        }

        void setAlpha(float x)
        {
            m_spriteAlpha = x;
        }
        void setColorAttenuation(vec3f c)
        {
            m_colorAttenuation = c;
        }
        void setLightColor(vec3f c);

        void setDoBlur(bool b)
        {
            m_doBlur = b;
        }
        void setBlurRadius(float x)
        {
            m_blurRadius = x;
        }

        void beginSceneRender(Target target);
        void endSceneRender(Target target);

        void setLightPosition(vec3f v)
        {
            m_lightPos = v;
        }
        void setLightTarget(vec3f v)
        {
            m_lightTarget = v;
        }

        vec4f getLightPositionEyeSpace()
        {
            return m_lightPosEye;
        }

        void calcVectors();
        vec3f getSortVector()
        {
            return m_halfVector;
        }

        void drawText(int x0, int y0, int width, int height, char *text);
        void render(int offX, int offY, int viewportW, int viewportH, char *text = NULL);
        void drawTestScene3D(int offX, int offY, float minDist, float maxDist);
        void renderPose(float *T, float fovX, float aspectRatio,float r, float g, float b, float len);
        void debugVectors();
        float *getK();
        void drawBox(vec3f origin,vec3f dim);
        void setCameraMatrix(float *modelView);
        void displayImage(GLuint texture, int x0, int y0, int width, int height, char *text=NULL);
        void displayRGBDImage(GLuint rgbTex, GLuint depthTex, int x0, int y0, int width, int height, float minDepth=0.0f, float maxDepth=10000.0f); 	  
	private:
        void drawPoints(int start, int count, bool sort, int mPosVbo, int mColorVbo, int mIndexBuffer);
        void drawPointTransparent(GLSLProgram *prog, int start, int count, int mPosVbo, int mColorVbo, int mIndexBuffer);
        void drawPointOpaque(GLSLProgram *prog, int start, int count, int mPosVbo, int mColorVbo, int mIndexBuffer);
        void drawPointBinaryTransparent(GLSLProgram *prog, int start, int count, int mPosVbo, int mColorVbo, int mIndexBuffer);
        void drawPointDepth(int start, int count, int mPosVbo, int mColorVbo, int mIndexBuffer);
		void renderRays();
        void traverseVoxels(const vec3f &origin,const vec3f &rayDir, const vec3f &invDir, int *sign, float enterT, float exitT, const vec3f *bounds, const vec3f &voxelDim);
        void calcRayDir(int xi, int yi, float *K, float *T, vec3f &rayDir, vec3f &invDir, int *sign);
        void draw2dPoints(int offx, int offy, int w, int h);
        void displayTexture(GLuint tex);
        void compositeResult();
        void blurLightBuffer();
        void setPerspective(float fovY, float aspectratio, float znear, float zfar);

        GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format);
        void createBuffers(int w, int h);
        void createLightBuffer();

        void drawQuad();
        void drawVector(vec3f v);

        float K[9];
        float cameraPose[16];
        float R[9];

        // particle data
        unsigned int        mNumVertices;
        unsigned int        mGridReso;

        GLuint              mPosVbo;
        GLuint              mCloudVbo;
        GLuint              mPos2dVbo;
        GLuint              mColorVbo;
        GLuint              mIndexBuffer;

        float               mPointSize;
        DisplayMode         mDisplayMode;

        // window
        unsigned int        mWindowW, mWindowH;
        float               mAspect, mInvFocalLen;
        float               mFov;

        int                 m_downSample;
        int                 m_imageW, m_imageH;

        float               m_spriteAlpha;
        bool                m_doBlur;
        float               m_blurRadius;

        vec3f               m_cubeOrigin,m_cubeDim;
        vec3f               m_lightVector, m_lightPos, m_lightTarget;
        vec3f               m_lightColor;
        vec3f               m_colorAttenuation;
        vec3f               m_cameraPos;
        float               m_lightDistance;

        matrix4f            m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
        vec3f               m_viewVector, m_halfVector;
        bool                m_invertedView;
        vec4f               m_eyePos;
        vec4f               m_halfVectorEye;
        vec4f               m_lightPosEye;

        // programs
        GLSLProgram         *m_particleProg;
        GLSLProgram         *m_textureProg;
		GLSLProgram         *m_depthProg;
        GLSLProgram         *m_simpleProg;
        GLSLProgram         *m_displayTexProg;
        GLSLProgram         *m_depth2DProg;
        GLSLProgram         *m_textureProg;
        GLuint floorTex;
        GLuint boxTex;

        GLuint              m_imageTex, m_depthTex;
        FramebufferObject   *m_imageFbo;
};

#endif
