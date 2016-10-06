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

#pragma once

#include <GL/glew.h>
#include "GLSLProgram.h"
#include <tracker/basic_math.h>
#include <Eigen/Geometry>
#include <calib/calib.h>
#include <utils/include/Keyframe.h>
#include <utils/include/LidarOctree.h>

class Renderer
{
    public:
        Renderer();
        ~Renderer();

        void setCube(const Eigen::Vector3f &origin, const Eigen::Vector3f &dim) {
            m_cubeOrigin = origin;
            m_cubeDim    = dim;
        }

        void setCameraPos(float x, float y, float z) {
            m_cameraPos(0) = x;
            m_cameraPos(1) = y;
            m_cameraPos(2) = z;
            printf("setPos: %f %f %f\n",m_cameraPos(0),m_cameraPos(1),m_cameraPos(2));
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
		void setFOV(float fov)
        {
            mFov = fov;
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

        void setPointSize(const Eigen::Vector3f &sz)
        {
            mPointSize = sz(0);
        }
        void setLightTarget(const Eigen::Vector3f &v)
        {
            m_lightTarget = v;
        }
        void resetView3d(int x0, int y0, int w, int h);
        void resetView2d(int x0, int y0, int w, int h);
        void setCalib(float fov, float w, float h);


        void drawText(int x0, int y0, int width, int height, char *text,float xPos = -0.8f, float yPos = 0.9f, float r = 1.0f, float g = 1.0f, float b = 1.0f);
        void render(int offX, int offY, int viewportW, int viewportH, char *text = NULL);        
        void renderPose(Eigen::Matrix4f &T, float fovX, float fovY, float aspectRatio,float r, float g, float b, float len);
        void renderPose(float *T, float fovX, float aspectRatio,float r, float g, float b, float len);
        float *getK();
        void drawBox(const Eigen::Vector3f &origin,const Eigen::Vector3f &dim, const Eigen::Vector3f &color);
        void setCameraMatrix(float *modelView);
        void setCameraMatrix(Eigen::Matrix4f &modelView);
        void setCameraPose(Eigen::Matrix4f &modelView);
        float *getCameraMatrix();
        float *getPoseMatrix();
        void drawFilledCircle(float x0, float y0, float radius, float r, float g, float b);
        void drawCursor(float x0, float y0, float dimx, float dimy);
        void draw3DPoints(Eigen::MatrixXf &featurePoints3D, Eigen::MatrixXf &statusPoints3D, float r, float g, float b);
        void displayImage(GLuint texture, float x0, float y0, float dimx, float dimy, float r, float g, float b, float a);
        void renderSphere(GLuint rgbTex, float radius, int verticalSlices, int horizontalSlices, bool flipWorld=false, float sphereAmount=0.50f, float radScale=0.5f, float vShift=0);
        void renderFrontSphere(GLuint rgbTex, float radius, int verticalSlices, int horizontalSlices, bool flipWorld=false, float sphereAmount=0.50f, float radScale=0.5f, float vShift=0, float yRot=0);
        void renderFrontWireSphere(float radius, int verticalSlices, int horizontalSlices, bool flipWorld=false, float sphereAmount=0.50f, float radScale=0.5f, float vShift=0, float yRot=0);
        void renderWireSphere(float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float sphereAmount, float radScale, float vShift=0);
        void vignettePass(GLuint rgbTex, GLuint vigTex, bool flipY=false);
        void displayRGBDImage(GLuint rgbTex, bool flipY=false, bool rot90=false, float zAngle=0.0f, bool binaryAlpha=false);
        void displayRGBDImage_old(GLuint rgbTex, GLuint depthTex, int flip=0);
        void displayWeightImage(GLuint rgbdTex,float initialWeight, int width, int height, float minDepth, float maxDepth, bool flipY=false);
        void combineRGBDLayers(GLuint rgbdTex0, GLuint weightTex0, GLuint rgbdTex1, GLuint weightTex1, int width, int height, float minDepth, float maxDepth, bool flipY, float zepsilon=1e-5f);
        void combineWeightLayers(GLuint rgbdTex0, GLuint weightTex0, GLuint rgbdTex1, GLuint weightTex1, int width, int height, float minDepth, float maxDepth, bool flipY, float zepsilon=1e-5f);
        void displayDepthImage(GLuint depthTex, int width, int height, int flip=0);
        void renderTexturedMesh(ProjectData *pointCloud, int texWidth, int texHeight, Calibration &calib, GLuint rgbTex);        
        void renderLidar(unsigned int vbo, int pointCount, float pointSize, float minDist, float maxDist);
        void drawMesh(unsigned int vbo, int npoints, unsigned ibo, int nfaces3);
        void drawTestScene3D(GLuint tex, float minDist, float maxDist);
        //void renderLidarDepth(LidarDataset &lidarData, float minDist, float maxDist);
        void renderTexturedVbo(Keyframe *kf, Calibration &calib);
        void renderGouraudVbo(Keyframe *kf, Calibration &calib, float *lightDir);
        void setProjectionMatrix(float *K, float texWidth, float texHeight, float nearZ, float farZ);
        void genProjectionMtx(float *Kin, float texWidth, float texHeight, float nearZ, float farZ, float *P);
        void displayTexture(GLuint tex);
private:       
        void renderRays();
        void traverseVoxels(const Eigen::Vector3f &origin,const Eigen::Vector3f &rayDir, const Eigen::Vector3f &invDir, int *sign, float enterT, float exitT, const Eigen::Vector3f *bounds, const Eigen::Vector3f &voxelDim);
        void calcRayDir(int xi, int yi, float *K, float *cameraPose, Eigen::Vector3f &rayDir, Eigen::Vector3f &invDir, int *sign);
        void compositeResult();
        void blurLightBuffer();
        void setPerspective(float fovY, float aspectratio, float znear, float zfar);

        GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format);
       // void createBuffers(int w, int h);
       // void createLightBuffer();
        void drawQuad();
        float K[9],Knew[9]; float zNear,zFar;
        float cameraPose[16];
        float cameraTransform[16];
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

        // window
        unsigned int        mWindowW, mWindowH;
        int                 m_downSample;
        int                 m_imageW, m_imageH;

        float               m_spriteAlpha;
		float               mFov;

        Eigen::Vector3f         m_cubeOrigin,m_cubeDim;
        Eigen::Vector3f               m_lightVector, m_lightPos, m_lightTarget;
        Eigen::Vector3f               m_lightColor;
        Eigen::Vector3f               m_colorAttenuation;
        Eigen::Vector3f               m_cameraPos;
        float               m_lightDistance;

        Eigen::Matrix4f            m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
        Eigen::Vector3f               m_viewVector, m_halfVector;
        bool                m_invertedView;
        Eigen::Vector3f               m_eyePos;
        Eigen::Vector3f               m_halfVectorEye;
        Eigen::Vector3f               m_lightPosEye;

        // programs
        GLSLProgram         *m_particleProg;
		GLSLProgram         *m_textureProg;
        GLSLProgram         *m_depthProg;
        GLSLProgram         *m_simpleProg;
        GLSLProgram         *m_displayTexProg;
        GLSLProgram         *m_rgbd2DProg;
        GLSLProgram         *m_depth2DProg;
        GLSLProgram         *m_rgba2DProg;
        GLSLProgram         *m_combine2DProg;
        GLSLProgram         *m_weightFusionProg;
        GLSLProgram         *m_weight2DProg;
        GLSLProgram         *m_autoTexShader;
        GLSLProgram         *m_texShader;
        GLSLProgram         *m_gouraudShader;
        GLSLProgram         *m_colorProg;
        GLSLProgram         *m_colorVertexDepthProg;
        GLSLProgram         *m_depthVertexDepthProg;
};

