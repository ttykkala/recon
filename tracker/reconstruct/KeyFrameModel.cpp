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
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/glew.h> // GLEW Library
#include <cudakernels/cuda_funcs.h>
#include <KeyFrameModel.h>
#include <opencv2/opencv.hpp>
#include <libfreenect.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include "timer/performanceCounter.h"
#include <cudakernels/hostUtils.h>
#include <reconstruct/basic_math.h>
#include <reconstruct/zconv.h>
#include <reconstruct/poisson/MultiGridOctreeData.h>
#include <reconstruct/poisson/Geometry.h>
#include <rendering/VertexBuffer2.h>
#include <rendering/TriangleBuffer2.h>
#include <multicore/multicore.h>
#include <calib/GroundTruth.h>
#include <structure/configuration.h>
#include <image2/Image2.h>
#include <sys/stat.h>
#include <map>

using namespace std;
using namespace cv;

//static Image2 imDistorted;
static unsigned char *imRGBDev = NULL;
static unsigned short *imDispDev = NULL;
static float *imDepthDev = NULL;

bool fileExists(const char *fn) {
    FILE *f = fopen(fn,"rb");
    if (f == NULL) return false;
    fclose(f);
    return true;
}

void KeyFrameModel::uploadCalibData() {

    if (m_calib.getCalibData() == NULL) {
        printf("attempting to upload from null pointer!\n");
        fflush(stdin);
        fflush(stdout);
        return;
    }
    if (calibDataDev == NULL) cudaMalloc( (void **)&calibDataDev, sizeof(float)*(CALIB_SIZE));
    cudaMemcpy(calibDataDev, m_calib.getCalibData(), sizeof(float)*CALIB_SIZE, cudaMemcpyHostToDevice);
}

int skipRow(char *buf) {
    int off = 0;
    while(buf[off] != '\n') off++; off++;
    return off;
}

int skipWhiteSpace(char *buf) {
    int off = 0;
    while(buf[off] != ' ') off++; off++;
    return off;
}


void parseKeyFramesTxt(const char *fn, int *firstFrame, int *lastFrame, int *distTol, int *angleTol, int *nFrames, char *poseFile, char *datasetPath) {
    FILE *f = fopen(fn,"rb");

    fseek(f,0,SEEK_END);
    long fileSize = ftell(f);
    fseek(f,0,SEEK_SET);
    char *txtBuf = new char[fileSize];
    int ret = fread(txtBuf,fileSize,1,f);
    fclose(f);

    int offset = 0;
    *firstFrame = atoi(&txtBuf[offset]);     offset += skipRow(&txtBuf[offset]);
    *lastFrame = atoi(&txtBuf[offset]);      offset += skipRow(&txtBuf[offset]);
    *nFrames = atoi(&txtBuf[offset]);        offset += skipWhiteSpace(&txtBuf[offset]);
    *distTol = atoi(&txtBuf[offset]);        offset += skipWhiteSpace(&txtBuf[offset]);
    *angleTol = atoi(&txtBuf[offset]);       offset += skipRow(&txtBuf[offset]);

    sscanf(&txtBuf[offset],"%s\n",&poseFile[0]); offset += skipRow(&txtBuf[offset]); //printf("p: %s\n",poseFile);
    sscanf(&txtBuf[offset],"%s\n",&datasetPath[0]); offset += skipRow(&txtBuf[offset]);

    delete[] txtBuf;
}

KeyFrameModel::KeyFrameModel(const char *keyFramePath, const int nLayers, int pixelSelectionAmount, int interpolatedKeyPointAmount,  bool optimizeGPUMemory, bool undistortRGB, bool renderableVBuffers, bool rgbVisualization) : nMultiResolutionLayers(nLayers)
{    
    calibDataDev = NULL;
    savedTransforms = NULL;
    trajectory=NULL;
    m_optimizeGPUMemory = optimizeGPUMemory;
    Calibration *calib = &m_calib;

    interpolatedKey = NULL;
    vbufferTemp = NULL;
    vertexFusionLayersDev  = NULL;
    vertexFusionIndicesDev = NULL;
    m_interpolatedKeyPointCount = interpolatedKeyPointAmount;
    m_renderableFlag = renderableVBuffers;
    m_rgbVisualization = rgbVisualization;
    // allocate memory for large UV map
    rgbTexture = cv::Mat(2048,2048,CV_8UC3); memset(rgbTexture.ptr(),0,2048*2048*3);
    rgbTextureLarge = cv::Mat(4096,4096,CV_8UC3); memset(rgbTextureLarge.ptr(),0,4096*4096*3);

    // default iterations:
    nIterations[0] = 10;
    nIterations[1] = 3;
    nIterations[2] = 2;

    char buf[512];
    sprintf(buf,"%s/keyframes.txt",keyFramePath);
    if (fileExists(buf)) loadKeyFrames(keyFramePath,calib, pixelSelectionAmount, interpolatedKeyPointAmount, undistortRGB);
};


KeyFrameModel::KeyFrameModel(int slamDistTol, int slamAngleTol, int nMaxKeys, Calibration *extCalib, const int nLayers, int pixelSelectionAmount, int interpolatedKeyPointAmount,  bool undistortRGB, bool renderableVBuffers, bool rgbVisualization) : nMultiResolutionLayers(nLayers)
{
    calibDataDev = NULL;
    trajectory=NULL;
    savedTransforms = NULL;
    m_optimizeGPUMemory = true;
    m_calib.copyCalib(extCalib);
    m_rgbVisualization = rgbVisualization;
    Calibration *calib = &m_calib;

    interpolatedKey = NULL;
    vbufferTemp = NULL;
    vertexFusionLayersDev  = NULL;
    vertexFusionIndicesDev = NULL;
    m_interpolatedKeyPointCount = interpolatedKeyPointAmount;
    m_renderableFlag = renderableVBuffers;
    setDimensions();

    distTol = slamDistTol;
    angleTol = slamAngleTol;
    maxNumKeyFrames = nMaxKeys;

    // default iterations:
    nIterations[0] = 10;
    nIterations[1] = 3;
    nIterations[2] = 2;
    printf("incremental model params set to %d %d %d\n",maxNumKeyFrames,distTol,angleTol);
};


void KeyFrameModel::setDimensions()
{
    width = 320; height = 240;
    printf("keyframe dimensions: %d x %d\n",width,height);

    if (imRGBDev != NULL) cudaFree(imRGBDev); imRGBDev = NULL;
    if (imDispDev != NULL) cudaFree(imDispDev); imDispDev = NULL;
    if (imDepthDev != NULL) cudaFree(imDepthDev); imDepthDev = NULL;
    if (vertexFusionLayersDev != NULL) cudaFree(vertexFusionLayersDev); vertexFusionLayersDev = NULL;
    if (vertexFusionIndicesDev != NULL) cudaFree(vertexFusionIndicesDev); vertexFusionIndicesDev = NULL;
    if (depth1C.data != NULL) depth1C.releaseData();

    if (vbufferTemp != NULL) { vbufferTemp->release(); delete vbufferTemp; vbufferTemp = NULL; }

    cudaMalloc((void**)&imRGBDev,width*height*3);
    cudaMalloc((void**)&imDispDev,width*height*4*sizeof(short));
    cudaMalloc((void**)&imDepthDev,width*height*sizeof(float));
    cudaMalloc((void**)&vertexFusionLayersDev,width*height*sizeof(float)*VERTEX_FUSION_LAYER_COUNT*VERTEXBUFFER_STRIDE);
    cudaMalloc((void**)&vertexFusionIndicesDev,width*height*sizeof(int)*VERTEX_FUSION_LAYER_COUNT);
    createHdrImage(NULL,width,height,1,&depth1C,ONLY_GPU_TEXTURE, false); depth1C.setName("depth1C");
    vbufferTemp = new VertexBuffer2(width*height);

    m_calib.setupCalibDataBuffer(width,height);
    uploadCalibData();
    generateGridObject(&gridObject[0],nGridX,nGridY,nGridZ,1000.0f,2000.0f,3000.0f,m_calib.getFovX());
}



float *KeyFrameModel::parseBundleData(const char *bundleFile, std::map<int, std::list<ProjectionData> > &projMap) {
    printf("bundlefile: %s\n",bundleFile);
    FILE *f = fopen(bundleFile,"rb");
    if (f == NULL) {
        printf("no bundle file found!\n");
        return NULL;
    }
    fseek(f,0,SEEK_END);
    long fileSize = ftell(f);
    fseek(f,0,SEEK_SET);
    char *fileBuffer = new char[fileSize];
    int ret = fread(fileBuffer,fileSize,1,f);
    fclose(f);

    int off = 0;
    if (fileBuffer[off] == '#') off += skipRow(&fileBuffer[off]);
    int nCameras = 0;
    int nPoints = 0;
    sscanf(&fileBuffer[off],"%d %d\n",&nCameras,&nPoints);
    printf("nCameras: %d, nPoints: %d\n",nCameras,nPoints);
    off += skipRow(&fileBuffer[off]);

    float *bundleParams = new float[nCameras*16]; memset(bundleParams,0,sizeof(float)*nCameras*16);
    float *invBundleParams = new float[nCameras*16]; memset(invBundleParams,0,sizeof(float)*nCameras*16);


    float R[9],Rt[9],t[3],c[3],p3d[3],p2d[3],r3d[3];
    int rgb[3];
    char buf[512];
    for (int i = 0; i < nCameras; i++) {
        off += skipRow(&fileBuffer[off]);
        sscanf(&fileBuffer[off],"%e %e %e\n",&R[0],&R[1],&R[2]); off += skipRow(&fileBuffer[off]);
        sscanf(&fileBuffer[off],"%e %e %e\n",&R[3],&R[4],&R[5]); off += skipRow(&fileBuffer[off]); //R[3] *= -1.0f; R[4] *= -1.0f; R[5] *= -1.0f;
        sscanf(&fileBuffer[off],"%e %e %e\n",&R[6],&R[7],&R[8]); off += skipRow(&fileBuffer[off]); //R[6] *= -1.0f; R[7] *= -1.0f; R[8] *= -1.0f;
        sscanf(&fileBuffer[off],"%e %e %e\n",&t[0],&t[1],&t[2]); off += skipRow(&fileBuffer[off]);
/*
        t[0] = 0;
        t[1] = 0;
        t[2] = 0;
*/

        invBundleParams[i*16+0]  = R[0]; invBundleParams[i*16+1]  = R[1]; invBundleParams[i*16+2]   = R[2]; invBundleParams[i*16+3]  = t[0];
        invBundleParams[i*16+4]  = R[3]; invBundleParams[i*16+5]  = R[4]; invBundleParams[i*16+6]   = R[5]; invBundleParams[i*16+7]  = t[1];
        invBundleParams[i*16+8]  = R[6]; invBundleParams[i*16+9]  = R[7]; invBundleParams[i*16+10]  = R[8]; invBundleParams[i*16+11] = t[2];
        invBundleParams[i*16+12] =    0; invBundleParams[i*16+13] =    0; invBundleParams[i*16+14]  =    0; invBundleParams[i*16+15] = 1;

        invertRT4(&invBundleParams[i*16], &bundleParams[i*16]);

        sprintf(buf,"bundle matrix %d",i);
        dumpMatrix(buf, &bundleParams[i*16], 4,4);
//      (R t) * (Rt -Rt * t) = (I -t + t) = I4x4
//      (0 1)   (   0     1)   (0    1  )
    }
    for (int i = 0; i < nPoints; i++) {
        sscanf(&fileBuffer[off],"%e %e %e\n",&p3d[0],&p3d[1],&p3d[2]); off += skipRow(&fileBuffer[off]);
   //     printf("point3d[%d]: %e %e %e\n",i, p3d[0],p3d[1],p3d[2]);
        sscanf(&fileBuffer[off],"%d %d %d\n",&rgb[0],&rgb[1],&rgb[2]); off += skipRow(&fileBuffer[off]);
  //      printf("color[%d]: %d %d %d\n",i, rgb[0],rgb[1],rgb[2]);
        int nViews = atoi(&fileBuffer[off]); //printf("nViews[%d]=%d\n",i,nViews);
        off += skipWhiteSpace(&fileBuffer[off]);
        int viewIndex,siftIndex;
        for (int j = 0; j < nViews; j++) {
            if (j < nViews-1) {
                sscanf(&fileBuffer[off],"%d %d %f %f ",&viewIndex,&siftIndex,&p2d[0],&p2d[1]);
                off += skipWhiteSpace(&fileBuffer[off]);
                off += skipWhiteSpace(&fileBuffer[off]);
                off += skipWhiteSpace(&fileBuffer[off]);
                off += skipWhiteSpace(&fileBuffer[off]);
            } else {
                sscanf(&fileBuffer[off],"%d %d %f %f\n",&viewIndex,&siftIndex,&p2d[0],&p2d[1]);
                off += skipRow(&fileBuffer[off]);
            }
            transformRT3(&invBundleParams[viewIndex*16], p3d, r3d);
//            printf("point[%d], %d, (%f,%f)\n",i,viewIndex,p2d[0]+319,239-p2d[1]);
            projMap[viewIndex].push_back(ProjectionData(float(p2d[0]+319)/2.0f,float(239-p2d[1])/2.0f,r3d[2]));
        }
    }
    delete[] invBundleParams;
    delete[] fileBuffer;
    return bundleParams;
}

void KeyFrameModel::fastImageMedian(Mat &src, int *medianVal) {
    unsigned char *srcData = src.ptr();

    unsigned int hist[256]; memset(hist,0,256*sizeof(int));
    unsigned int mass = 0;
    int offset = 0;
    for (int j  = 0; j < src.rows; j++) {
            for (int i  = 0; i < src.cols; i++,offset++) {
                    unsigned char v0 = srcData[offset];
                    hist[v0]++;
                    mass++;
            }
    }
    // seek median value
    int desiredMass = mass/2;
    int currentMass = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++) {
            currentMass += hist[i];
            if (currentMass >= desiredMass) { threshold = i; break;}
    }
    *medianVal = threshold;
}


void generateBaseMask(Mat &src, float borderPercent, Mat &baseMask) {
    int size = src.cols*src.rows;
    unsigned char *srcPtr = src.ptr();
    unsigned char *dstPtr = baseMask.ptr();

    int borderPixels = int(borderPercent * src.cols + 0.5f);

    memset(dstPtr,255,size);

    int offset = 0;
    for (int j = 0; j < src.rows; j++) {
        for (int i = 0; i < src.cols; i++,offset++) {
            if (j < borderPixels || i < borderPixels || j >= src.rows-borderPixels || i >= src.cols-borderPixels) dstPtr[offset] = 0;
            if (srcPtr[offset] == 0) dstPtr[offset] = 0;
        }
    }
}

int countFrames(const char *camRPath) {
    char fn[512];
    int frame = 1;
    int numFrames = 0;
    while (1) {
        sprintf(fn,"%s/bayer_rgbimage%04d.ppm",camRPath,frame);
        if (!fileExists(fn)) {
            sprintf(fn,"%s/%04d.ppm",camRPath,frame);
            if (fileExists(fn)) { numFrames++; frame++; continue;} else break;
        } else {
            numFrames++;
            frame++;
        }
    }
    return numFrames;
}

void KeyFrameModel::renderTrajectory(int frame) {
    if (trajectory == NULL) { return;}
    if (frame == -1) frame = trajectory->getPointCount()/2;
    else frame = frame % (trajectory->getPointCount()/2);
    glPushMatrix();
    glPointSize(4.0f);
 //   glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
   // glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

//    glLineWidth(15.0f);
//    glLineStipple(1,0xffff);
//    glEnable(GL_LINE_STIPPLE);

//    glEnable(GL_LINE_SMOOTH);
//    glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
    trajectory->render(frame);
    //glDisable(GL_POINT_SMOOTH);
    glPopMatrix();
}

void KeyFrameModel::renderInterpolatedPose(float r, float g, float b, float len) {
    if (interpolatedKey != NULL)
       renderPose(interpolatedKey,r,g,b,len);
}

void KeyFrameModel::renderPose(KeyFrame *kf, float r, float g, float b, float len) {
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);

    if (kf && kf->visible) {
        float o[3]; o[0] =      kf->T[12]; o[1] =     kf->T[13]; o[2] =      kf->T[14];
        float z[3]; z[0] =  -len*kf->T[8]; z[1] = -len*kf->T[9]; z[2] = -len*kf->T[10];
        float sX = tan(3.141592653f*kf->getFovX()/360.0f)*len;
        float sY = tan(3.141592653f*kf->getFovY()/360.0f)*len;

        float u[3]; u[0] = sX*kf->T[0]; u[1] = sX*kf->T[1]; u[2] = sX*kf->T[2];
        float v[3]; v[0] = sY*kf->T[4]; v[1] = sY*kf->T[5]; v[2] = sY*kf->T[6];

        float x0[3]; x0[0] = o[0]+z[0]-u[0]-v[0]; x0[1] = o[1]+z[1]-u[1]-v[1]; x0[2] = o[2]+z[2]-u[2]-v[2];
        float x1[3]; x1[0] = o[0]+z[0]+u[0]-v[0]; x1[1] = o[1]+z[1]+u[1]-v[1]; x1[2] = o[2]+z[2]+u[2]-v[2];
        float x2[3]; x2[0] = o[0]+z[0]+u[0]+v[0]; x2[1] = o[1]+z[1]+u[1]+v[1]; x2[2] = o[2]+z[2]+u[2]+v[2];
        float x3[3]; x3[0] = o[0]+z[0]-u[0]+v[0]; x3[1] = o[1]+z[1]-u[1]+v[1]; x3[2] = o[2]+z[2]-u[2]+v[2];

        glColor3f(r,g,b);

        glVertex3f(x0[0],x0[1],x0[2]); glVertex3f(x1[0],x1[1],x1[2]);
        glVertex3f(x1[0],x1[1],x1[2]); glVertex3f(x2[0],x2[1],x2[2]);
        glVertex3f(x2[0],x2[1],x2[2]); glVertex3f(x3[0],x3[1],x3[2]);
        glVertex3f(x3[0],x3[1],x3[2]); glVertex3f(x0[0],x0[1],x0[2]);

        glVertex3f(o[0],o[1],o[2]); glVertex3f(x0[0],x0[1],x0[2]);
        glVertex3f(o[0],o[1],o[2]); glVertex3f(x1[0],x1[1],x1[2]);
        glVertex3f(o[0],o[1],o[2]); glVertex3f(x2[0],x2[1],x2[2]);
        glVertex3f(o[0],o[1],o[2]); glVertex3f(x3[0],x3[1],x3[2]);
    }

    glEnd();
    glPopAttrib();

}

void KeyFrameModel::renderCameraFrames(std::vector<KeyFrame *> *activeKeys, float len, float r, float g, float b) {

    float maxWeight = 0;
    std::vector<KeyFrame *>::iterator ki;
    if (activeKeys != NULL) {
        // find max weight
        for (ki = activeKeys->begin(); ki != activeKeys->end(); ki++) {
            KeyFrame *activeKey = *ki;
            if (activeKey->weight > maxWeight)  { maxWeight = activeKey->weight; }
        }
    }

    int cnt = getKeyFrameCount();
    for (int i = 0; i < cnt; i++) {
        KeyFrame *kf = getKeyFrame(i);
        if (kf && kf->visible) {
            bool isActive = false;
            float activeR = 0, activeG = 1.0f, activeB = 0.0f;
            if (activeKeys != NULL) {
                // find active key
                for (ki = activeKeys->begin(); ki != activeKeys->end(); ki++) {
                    KeyFrame *activeKey = *ki;
                    if (activeKey == kf)  {
                        isActive = true;
                        activeR = 1.0f;//*= activeKey->weight/maxWeight;
                        activeG = 0;//*= activeKey->weight/maxWeight;
                        activeB = 0;//*= activeKey->weight/maxWeight;
                        break; }
                }
            }
            if (isActive) {
                renderPose(kf,activeR,activeG,activeB,len);
            } else {
                renderPose(kf,r,g,b,len);
            }
        }
    }
}

float integrateTrajectoryLength(float *cameraTrajectory,int firstFrame, int lastFrame) {
    float len = 0.0f;
    for (int i = firstFrame; i < lastFrame-1; i++) {
        float x1 = cameraTrajectory[i*16+3];
        float y1 = cameraTrajectory[i*16+7];
        float z1 = cameraTrajectory[i*16+11];
        float x2 = cameraTrajectory[(i+1)*16+3];
        float y2 = cameraTrajectory[(i+1)*16+7];
        float z2 = cameraTrajectory[(i+1)*16+11];
        float dx = x2-x1, dy = y2-y1, dz = z2-z1;
        len += sqrt(dx*dx+dy*dy+dz*dz+1e-5f);
    }
    return len;
}


void KeyFrameModel::writeTrajectoryDifferenceTxt(float *ref, float *cur,int numFrames, const char *outFile) {
    FILE *f = fopen(outFile,"wb");
    fprintf(f,"frame trans_error angle_error\n");
    for (int i = 0; i < numFrames; i++) {
        float relativeT[16];
        float invCur[16];
        float mRef[16],mCur[16];
        //transpose4x4(&ref[16*i],&mRef[0]);
        //transpose4x4(&cur[16*i],&mCur[0]);
        memcpy(&mRef[0],&ref[16*i],sizeof(float)*16);
        memcpy(&mCur[0],&cur[16*i],sizeof(float)*16);

        invertRT4(&mCur[0],&invCur[0]);
        matrixMult4x4(&invCur[0], &mRef[0], &relativeT[0]);
        float dist,angle;
        poseDistance(&relativeT[0], &dist,&angle);
        fprintf(f,"%d %e %e\n",i, dist,angle);
    }
    fclose(f);
}

void baselineTransform(Mat &depthImageL,Mat &depthImageR,float *KL, float *TLR, float *KR) {
    int width  = depthImageL.cols;
    int height = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    float *zptrDst = (float*)depthImageR.ptr();
    int *counterImage = new int[width*height];
    memset(counterImage,0,sizeof(int)*width*height);
    memset(zptrDst,0,sizeof(float)*width*height);

    float fx = KL[0];
    float fy = KL[4];
    float cx = KL[2];
    float cy = KL[5];

    int offset = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++,offset++) {
            float z = zptrSrc[offset];
            float p3[3],r3[3],p2[3];
            p3[0] = -(float(i) - cx) * z / fx;
            p3[1] = -(float(j) - cy) * z / fy;
            p3[2] = -z;
            transformRT3(TLR, p3, r3);
            matrixMultVec3(KR, r3, p2); p2[0] /= p2[2]; p2[1] /= p2[2];
            int xi = (int)p2[0];
            int yi = (int)p2[1];

            if (xi >= 0 && yi >= 0 && xi < width && yi < height) {
                int offset = xi + yi * width;
                zptrDst[offset] += fabs(r3[2]);
                counterImage[offset]++;
            }
        }
    }

    offset = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++,offset++) {
            if (counterImage[offset] > 0) zptrDst[offset] /= (float)counterImage[offset];
        }
    }
}

/*
void KeyFrameModel::adjustPoses(Calibration *calib, int pixelSelectionAmount) {
    std::vector<KeyFrame*>::iterator ki;
    int keyIndex = 0;
    char buf[512];
    KeyFrame *refKey = NULL;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++,keyIndex++) {
        KeyFrame *keyFrame = (KeyFrame*)*ki;
        // generate pixel selection mask for sift descriptors
        keyFrame->selectPixels(keyFrame->grayImage,keyFrame->depthRGB,keyFrame->baseMask,keyFrame->pixelSelectionMask,pixelSelectionAmount);
        calib->setupCalibDataBuffer(320,240);
        //if (keyIndex % 4 == 0) { keyFrame->initializeFeatures(100,keyIndex,calib); refKey = keyFrame; }
    //    else refKey->searchFeatures(keyFrame,keyIndex,calib,16,300.0f);
    }
}*/

void KeyFrameModel::integrateDepthMaps(const char *datasetPath, Calibration *calib, float *cameraTrajectory) {
    std::vector<KeyFrame*>::iterator ki;
    std::vector<int>::iterator ni;
    int keyIndex = 0;
    char buf[512];

    ZConv zconv; Mat depthMap(240,320,CV_32FC1);
    calib->setupCalibDataBuffer(320,240);
    Mat distortedRgbImageSmall(240,320,CV_8UC3);
    Mat rgbImageSmall(240,320,CV_8UC3);
    Mat rgbImage(480,640,CV_8UC3);

    float *TbRGB = &calib->getCalibData()[TLR_OFFSET];
    float T[16];
    float Kir[9];  memcpy(&Kir[0],&calib->getCalibData()[KL_OFFSET],sizeof(float)*9);
    float Krgb[9]; memcpy(&Krgb[0],&calib->getCalibData()[KR_OFFSET],sizeof(float)*9);

    float *kcRGB = &calib->getCalibData()[KcR_OFFSET];

    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++,keyIndex++) {
        KeyFrame *kCur = (KeyFrame*)*ki;
        int neighborIndex = *(kCur->neighborIndices.begin());
        int referenceFrame = kCur->id;
        Configuration fusionSet;
        // opengl->standard, current IR -> world
        transpose4x4(&kCur->Tbase[0],&T[0]); matrixMult4x4(&T[0],TbRGB,&T[0]);
        // initialize reference view for structural filtering
        fusionSet.init(&T[0],&kCur->depthCPU,&TbRGB[0],&Kir[0],&Krgb[0],kcRGB,256,&kCur->distortedRgbImageCPU);
        int i = 0;
        for (ni = kCur->neighborIndices.begin(); ni != kCur->neighborIndices.end(); ni++,i++) {
            neighborIndex = *ni;
            // if the rest of the measurements are more than 2 second apart, forget them
            if ((abs(referenceFrame-neighborIndex) > 60) || referenceFrame == neighborIndex ) continue;
            loadImage(datasetPath,neighborIndex, distortedRgbImageSmall);
            loadDepthMap(datasetPath, neighborIndex, depthMap, calib);
            // current IR -> world
            matrixMult4x4(&cameraTrajectory[(neighborIndex-1)*16],TbRGB,&T[0]);
            fusionSet.warpToReference(&T[0],&depthMap,&TbRGB[0],&Kir[0],&Krgb[0],kcRGB,&distortedRgbImageSmall,15);
        }
        fusionSet.filterDepth(&kCur->depthCPU,&Kir[0],&Krgb[0],kcRGB,300.0f,4);

        // generate depth map for rgb view (more coarse but ok for feature z init)
        calib->setupCalibDataBuffer(320,240);
        baselineTransform(kCur->depthCPU, kCur->depthRGB, &calib->getCalibData()[KL_OFFSET],&calib->getCalibData()[TLR_OFFSET],&calib->getCalibData()[KR_OFFSET]);
    }

}

void KeyFrameModel::loadImage(const char *datasetPath, int index, cv::Mat &rgbImageOut) {
    char buf[512];
    Mat rgbImage(480,640,CV_8UC3);
    sprintf(buf,"%s/bayer_rgbimage%04d.ppm",datasetPath,index);
//       printf("loading %s...\n",buf);

    Mat bayerImage = imread(buf,-1); //if (bayerImage.cols != width) printf("warning: input size: %dx%d, target reso: %dx%d\n",bayerImage.cols,bayerImage.rows,width,height);
    if (bayerImage.data == NULL) {
        sprintf(buf,"%s/%04d.ppm",datasetPath,index);
        Mat rgbHeader = imread(buf,-1);
        if (rgbHeader.data == NULL) printf("file %s not valid!\n",buf);
        else {
            if (rgbHeader.cols != rgbImageOut.cols || rgbHeader.rows != rgbImageOut.rows) { printf("loaded image does not match %dx%d resolution! abort!\n",rgbImageOut.cols,rgbImageOut.rows); return; }
            cvtColor(rgbHeader,rgbImageOut,CV_RGB2BGR);
        }
        return;
    }
    if (bayerImage.cols != 640 || bayerImage.rows != 480) { printf("bayer image not 640x480! abort!\n"); return; }
    cvtColor(bayerImage,rgbImage,CV_BayerGB2BGR,3);
   // printf("pyrdown: %d x %d -> %d x %d\n",rgbImage.cols,rgbImage.rows,rgbImageSmall.cols,rgbImageSmall.rows); fflush(stdin); fflush(stdout);
    if (rgbImage.cols == rgbImageOut.cols && rgbImage.rows == rgbImageOut.rows) {
        rgbImage.copyTo(rgbImageOut);
    } else {
        pyrDown(rgbImage,rgbImageOut);
    }
}

void KeyFrameModel::writeZMap(const char *fn, Mat &depthImage) {
    FILE *f = fopen(fn,"wb");
    if (f == NULL) { printf("could not create file %s\n",fn); return; }
    fwrite(depthImage.ptr(),depthImage.cols*depthImage.rows*sizeof(float),1,f);
    fclose(f);
}

bool KeyFrameModel::readZMap(const char *fn, Mat &depthImage) {
    FILE *f = fopen(fn,"rb");
    if (f == NULL) { printf("could not read file %s\n",fn); return false; }
    int ret = fread(depthImage.ptr(),depthImage.cols*depthImage.rows*sizeof(float),1,f);
    fclose(f);
    return true;
}

void KeyFrameModel::loadDepthMap(const char *datasetPath, int frameIndex, cv::Mat &depthMap, Calibration *calib) {
    ZConv zconv;
    char buf[512];
    sprintf(buf,"%s/rawdepth%04d.ppm",datasetPath,frameIndex);
    //printf("loading %s\n",buf);
    Mat dispImage = imread(buf,-1);
    if (dispImage.data == NULL ) {
        sprintf(buf,"%s/depth%04d.dat",datasetPath,frameIndex);
      //  printf("try loading %s\n",buf);
        bool loadOk = readZMap(buf,depthMap);
        if (!loadOk) {
            assert(0);
        }
        zconv.dumpDepthRange((float*)depthMap.ptr(),depthMap.cols,depthMap.rows);
        return;
    }
    if (dispImage.cols != 640 || dispImage.rows != 480) { printf("non-standard disparity map size!\n"); assert(0);}

    Mat uDispImage(480,640,CV_32FC1);
    zconv.undistortDisparityMap((unsigned short*)dispImage.ptr(),(float*)uDispImage.ptr(),640,480,calib);

    zconv.d2zHdr((float*)uDispImage.ptr(),dispImage.cols,dispImage.rows,(float*)depthMap.ptr(),320,240,calib);
    //zconv.increaseDynamics((float*)depthMap.ptr(),320,240,2.0f);
    zconv.setRange((float*)depthMap.ptr(),320*240,0,500.0f,0.0f);
}


void KeyFrameModel::generateGPUKeyframes(int pixelSelectionAmount, bool renderableVBuffer, bool undistortRGB, Calibration *calib) {
    std::vector<KeyFrame*>::iterator ki;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++) {
        KeyFrame *keyFrame = *ki;
        keyFrame->generateGPUData(pixelSelectionAmount,renderableVBuffer,m_optimizeGPUMemory,undistortRGB,m_rgbVisualization,calib, imRGBDev, imDepthDev, depth1C, vbufferTemp, nMultiResolutionLayers);
    }
    printFreeDeviceMemory();
}

int KeyFrameModel::getFrameCount() {
    return numFrames;
}

char *KeyFrameModel::getBundleFileName() {
    return &bundleFile[0];
}

char *KeyFrameModel::getSmoothFileName() {
    return &smoothFile[0];
}

void KeyFrameModel::setupCameraTrajectory(float *cameraTrajectory, int firstFrame, int lastFrame) {
    this->trajectory = new LineBuffer(5000*2);
    for (int i = firstFrame+1; i <= lastFrame; i++) {
        float r = 255, g = 0, b = 0;
        // also generate list of neighbors for depth map integration
        float m[16];
        if (!isKeyFrame(i)) {
            transpose4x4(&cameraTrajectory[(i-1)*16],m);
            KeyFrame *nearKey = findSimilarKeyFrame(m, float(distTol),float(angleTol));
            if (nearKey != NULL) {
                nearKey->neighborIndices.push_back(i);
                std::vector<KeyFrame*>::iterator ki;
                for (ki = nearKey->neighborKeys.begin(); ki != nearKey->neighborKeys.end(); ki++) {
                    KeyFrame *neighborKey = (KeyFrame*)*ki;
                    neighborKey->neighborIndices.push_back(i);
                }
        /*        r = 255.0;//nearKey->colorR;
                g = 0.0f;//nearKey->colorG;
                b = 0.0f;//nearKey->colorB;*/
            }
        }
        float x1 = cameraTrajectory[(i-1)*16+3];
        float y1 = cameraTrajectory[(i-1)*16+7];
        float z1 = cameraTrajectory[(i-1)*16+11];
        float x2 = cameraTrajectory[i*16+3];
        float y2 = cameraTrajectory[i*16+7];
        float z2 = cameraTrajectory[i*16+11];
        this->trajectory->addLine(x1,y1,z1,x2,y2,z2,r,g,b);
    }
    trajectory->upload();
}

void KeyFrameModel::generateGridObject(float *gridObj,int nStepsX, int nStepsY, int nLayers, float size, float viewDistanceMin, float zRange, float fovAngleX) {
    float stepX = (2*size)/nStepsX;
    float stepY = (2*size)/nStepsY;
    float stepZ = 0.0f;//zRange/nLayers;
    assert(nLayers > 0);
    if (nLayers == 1) viewDistanceMin += zRange/2.0f;
    else stepZ = zRange/float(nLayers-1);

    int nPoints = 0;
    for (int k = 0; k < nLayers; k++) {

        float xyScale = 1.0f;
        if (k > 0) {
            float zFirst = -viewDistanceMin;
            float zCur = -viewDistanceMin - stepZ*k;
            xyScale = (zCur*tan(fovAngleX*3.141592653f/180.0f))/(zFirst*tan(fovAngleX*3.141592653f/180.0f));
        }
        for (int j = 0; j < nStepsY; j++) {
            for (int i = 0; i < nStepsX; i++) {
                float p[3],pr[3],r[3];
                gridObj[nPoints*3+0] = (-size + i*stepX)*xyScale;
                gridObj[nPoints*3+1] = (-size + j*stepY)*xyScale;
                gridObj[nPoints*3+2] = -viewDistanceMin - stepZ*k;
                nPoints++;
            }
        }
    }
}


int KeyFrameModel::getKeyFrameIndex(KeyFrame *keyFrame) {
    std::vector<KeyFrame*>::iterator ki;
    int index = 0;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++,index++) {
        KeyFrame *kCur = (KeyFrame*)*ki;
        if (kCur == keyFrame) return index;
    }
    return -1;
}


bool KeyFrameModel::isKeyFrame(int frameIndex) {
    std::vector<KeyFrame*>::iterator ki;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++) {
        KeyFrame *kCur = (KeyFrame*)*ki;
        if (kCur->id == frameIndex) return true;
    }
    return false;
}

void KeyFrameModel::setupNeighborKeys(float maxDist, float maxAngle) {
    std::vector<KeyFrame*>::iterator ki;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++) {
        KeyFrame *key = (KeyFrame*)*ki;
        enumerateNearPoses(key->Tbase, maxDist, maxAngle, key->neighborKeys,key);
//        printf("xxx: %d\n",key->neighborKeys.size());
    }
}

void KeyFrameModel::addKeyFrame(int frameIndex, Mat &rgbImageSmall, Mat &depthMap, Calibration *calib, float *T) {
    // still room for additional keyframe?
    if (keyFrames.size() >= maxNumKeyFrames) return;
    KeyFrame *kf = createKeyFrame(frameIndex,rgbImageSmall,depthMap,calib,T); kf->setIterationCounts(&nIterations[0]);
    keyFrames.push_back(kf);
    printf("keyframe %d added2 (frame %d).\n",(int)keyFrames.size(),kf->id);
    fflush(stdin);
    fflush(stdout);
}

KeyFrame *KeyFrameModel::createKeyFrame(int frameIndex, cv::Mat &rgbImageSmall,cv::Mat &depthMap,Calibration *calib,float *T) {
    // create keyframe
    KeyFrame *keyFrame = new KeyFrame(width,height);

    keyFrame->distortedRgbImageCPU = rgbImageSmall.clone();
    //keyFrame->rgbImageCPU = rgbImageSmall.clone();
    keyFrame->depthCPU = depthMap.clone();
    keyFrame->depthRGB = depthMap.clone();

    // generate depth map for rgb view (more coarse but ok for many purposes)
    calib->setupCalibDataBuffer(320,240);
    baselineTransform(keyFrame->depthCPU, keyFrame->depthRGB, &calib->getCalibData()[KL_OFFSET],&calib->getCalibData()[TLR_OFFSET],&calib->getCalibData()[KR_OFFSET]);

    // setup calibration device pointer into shared gpu memory
    keyFrame->setCalibDevPtr(calibDataDev,calib->getCalibData()[KR_OFFSET]/160.0f,calib->getCalibData()[KR_OFFSET+4]/120.0f);

    Mat distortedGray(height,width,CV_8UC1);
    Mat grayImage(height,width,CV_8UC1);
    Mat grayImageHDR(height,width,CV_32FC1);

    cvtColor(keyFrame->distortedRgbImageCPU,distortedGray,CV_RGB2GRAY);
    // precompute keyframe median brightness for post brightness normalization
    fastImageMedian(distortedGray, &keyFrame->medianVal);

    OMPFunctions *multicore = getMultiCoreDevice();
    multicore->undistort(distortedGray,grayImage,&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
    keyFrame->grayImage = grayImage.clone();
    keyFrame->grayImage.convertTo(grayImageHDR,CV_32FC1);
    keyFrame->grayImageHDR = grayImageHDR.clone();
    keyFrame->pixelSelectionMask = grayImage.clone();
    keyFrame->baseMask = grayImage.clone();
    generateBaseMask(keyFrame->grayImage,0.03f,keyFrame->baseMask);
    keyFrame->updateTransform(&T[0]);
    keyFrame->id = frameIndex;
    if (keyFrames.size()%2 == 0) {
        keyFrame->colorR = 0;
        keyFrame->colorG = 128;
        keyFrame->colorB = 0;
    } else {
        keyFrame->colorR = 255;
        keyFrame->colorG = 69;
        keyFrame->colorB = 0;
    }
    keyFrame->updateFrustum(calib->getCalibData()[MIND_OFFSET],calib->getCalibData()[MAXD_OFFSET]);
    return keyFrame;
}

void tile(cv::Mat &rgbImage,cv::Mat &rgbTexture,int tileX,int tileY) {
    unsigned char *dst = rgbTexture.ptr();
    unsigned char *src = rgbImage.ptr();

    int tileWidth  = rgbImage.cols;    int tileHeight = rgbImage.rows;
    int texWidth   = rgbTexture.cols;  int texHeight  = rgbTexture.rows;

    int texOffset = tileX*tileWidth*3 + tileY*tileHeight*texWidth*3;

    // tile coordinates outside texture?
    if (texOffset >= texWidth*3*texHeight) return;

    for (int j = 0; j < tileHeight; j++) {

        for (int i = 0; i < tileWidth; i++) {
            dst[texOffset + i*3 + j*texWidth*3+0] = src[i*3+j*3*tileWidth+0];
            dst[texOffset + i*3 + j*texWidth*3+1] = src[i*3+j*3*tileWidth+1];
            dst[texOffset + i*3 + j*texWidth*3+2] = src[i*3+j*3*tileWidth+2];
        }
    }
}

void KeyFrameModel::generateTexture() {
    Mat rgbImage(240,320,CV_8UC3);
    Mat rgbImageLarge(480,640,CV_8UC3);

    std::vector<KeyFrame*>::iterator ki;
    int index = 0;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++,index++) {
        KeyFrame *key = (KeyFrame*)*ki;
        cvtColor(key->distortedRgbImageCPU,rgbImage,CV_RGB2BGR);
        loadImage(datasetPath,key->id, rgbImageLarge);
        cvtColor(rgbImageLarge,rgbImageLarge,CV_RGB2BGR);

        int tileX = index%6;
        int tileY = index/6;
        tile(rgbImage,rgbTexture,tileX,tileY);
        tile(rgbImageLarge,rgbTextureLarge,tileX,tileY);
    }
}


void KeyFrameModel::saveKeyImages() {
    char buf[512];
    generateTexture();
    sprintf(buf,"%s/texture.png",scratchDir);
    savePNG(rgbTexture.ptr(),rgbTexture.cols, rgbTexture.rows, 3, rgbTexture.cols*3, buf);
    sprintf(buf,"%s/textureLarge.png",scratchDir);
    savePNG(rgbTextureLarge.ptr(),rgbTextureLarge.cols, rgbTextureLarge.rows, 3, rgbTextureLarge.cols*3, buf);


    Mat rgbImage(240,320,CV_8UC3);
    std::vector<KeyFrame*>::iterator ki;
    int index = 0;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++,index++) {
        KeyFrame *key = (KeyFrame*)*ki;
        sprintf(buf,"%s/%04d.ppm",scratchDir,index+1);
        cvtColor(key->distortedRgbImageCPU,rgbImage,CV_RGB2BGR);
        imwrite(buf,rgbImage);
        sprintf(buf,"%s/depth%04d.dat",scratchDir,index+1);
        writeZMap(buf,key->depthCPU);
    }
}

void KeyFrameModel::cpuPointClouds(Calibration *calib) {
    int saveIndex = 1;
    std::vector<KeyFrame*>::iterator ki;
    for (ki = keyFrames.begin(); ki != keyFrames.end(); ki++) {
        KeyFrame *key = (KeyFrame*)*ki;
        key->generatePoints(calib);
    }
}


char *KeyFrameModel::getScratchDirName() {
    return &scratchDir[0];
}

void KeyFrameModel::reconstruct3dPoint(int srcKeyFrame, int dstKeyFrame, float px, float py, float *p3) {
    KeyFrame *srcKey = getKeyFrame(srcKeyFrame);
    KeyFrame *dstKey = getKeyFrame(dstKeyFrame);

    int xi = (int)px;
    int yi = (int)py;
    int offset = xi+yi*srcKey->depthCPU.cols;

    float *zdata = (float*)srcKey->depthCPU.ptr();
    float zf = zdata[offset];
    if (zf <= 0.0f) return;

    m_calib.setupCalibDataBuffer(320,240);

    float *TLR = &m_calib.getCalibData()[TLR_OFFSET];
    float iKir[9]; inverse3x3(&m_calib.getCalibData()[KL_OFFSET],&iKir[0]);

    float v[3],w[3];
    get3DPoint(float(xi),float(yi),zf,iKir, &v[0], &v[1], &v[2]);
    transformRT3(TLR,v,w);

    float srcT[16],dstT[16],T[16];
    transpose4x4(&srcKey->Tbase[0],&srcT[0]);
    transpose4x4(&dstKey->Tbase[0],&dstT[0]);
    relativeTransform(srcT, dstT, &T[0]);
    transformRT3(T,w,p3);
}


KeyFrame *KeyFrameModel::createDummyCPUKeyframe(int frameIndex, Calibration *calib, float *T, KeyFrame *previousKey) {
    KeyFrame *keyFrame = new KeyFrame(width,height);
    calib->setupCalibDataBuffer(320,240);
    // setup calibration device pointer into shared gpu memory
    keyFrame->setCalibDevPtr(calibDataDev,calib->getCalibData()[KR_OFFSET]/160.0f,calib->getCalibData()[KR_OFFSET+4]/120.0f);


    keyFrame->updateTransformGL(&T[0]);

    if (previousKey != NULL)
        keyFrame->setBaseTransform(previousKey->getNextBaseDev());

    keyFrame->id = frameIndex;
    if (keyFrames.size()%2 == 0) {
        keyFrame->colorR = 0;
        keyFrame->colorG = 128;
        keyFrame->colorB = 0;
    } else {
        keyFrame->colorR = 255;
        keyFrame->colorG = 69;
        keyFrame->colorB = 0;
    }
    keyFrame->updateFrustum(calib->getCalibData()[MIND_OFFSET],calib->getCalibData()[MAXD_OFFSET]);
    return keyFrame;
}

KeyFrame *KeyFrameModel::extendMap(int id, ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbuffer, float *imDepthDevIR, int pixelSelectionAmount, float *T, KeyFrame *previousKey) {
    // still room for additional keyframe?
    if (keyFrames.size() >= maxNumKeyFrames) {
        printf("maximum number of keyframes reached, map extension failed! (%d)\n",maxNumKeyFrames);
        fflush(stdin);
        fflush(stdout);
        return NULL;
    }
    Calibration *calib = &m_calib;
    KeyFrame *kf = createDummyCPUKeyframe(id,calib,T,previousKey); kf->setIterationCounts(&nIterations[0]);
    //dumpMatrix("new map pose",T,4,4);
    kf->gpuCopyKeyframe(id, frame1C, frame3C, vbuffer, imDepthDevIR, pixelSelectionAmount,vbufferTemp,3,m_renderableFlag);
    keyFrames.push_back(kf);
//    printf("keyframe %d added (frame %d).\n",(int)keyFrames.size(),kf->id);
//    fflush(stdin);
//    fflush(stdout);
    return kf;
}

// note! this should be called only once, otherwise it does not free memories
void KeyFrameModel::loadKeyFrames(const char *path, Calibration *calib, int pixelSelectionAmount, int interpolatedKeyPointAmount, bool undistortRGB) {
    char buf[512];
//    printf("%s/keyframes.txt",path); fflush(stdin); fflush(stdout);
    sprintf(buf,"%s/keyframes.txt",path);

    parseKeyFramesTxt(buf,&firstFrame,&lastFrame,&distTol,&angleTol,&maxNumKeyFrames,&poseFile[0],datasetPath);
    sprintf(&posePath[0],"%s/%s",path,poseFile);

    printf("first frame: %d\n",firstFrame);
    printf("last frame: %d\n",lastFrame);
    printf("max num keyframes: %d\n",maxNumKeyFrames);
    printf("distTol : %d, angleTol: %d\n",distTol,angleTol);
    printf("poseFile: %s\n",&posePath[0]);
    printf("dataSet: %s\n",datasetPath);

    sprintf(buf,"%s/calib/calib.yml",datasetPath);
    printf("calibDir: %s\n",buf);
    calib->init(buf);
    double minDist = calib->getMinDist();
    double maxDist = calib->getMaxDist();
    setDimensions();

    numFrames = countFrames(datasetPath);
    printf("%d bayer frames found!\n",numFrames);

    if (numFrames == 0) { printf("no images found! aborting!\n"); exit(1); }

   Mat rgbImageSmall(height,width,CV_8UC3);
   Mat depthMap(height,width,CV_32FC1);
   // täskit
   // 1) noise cleanup tool (etualan paskat pois keyframemallista)

   sprintf(scratchDir,"%s/%d-%d-%d",path,maxNumKeyFrames,distTol,angleTol);
   mkdir(scratchDir,0700);
   sprintf(&bundleFile[0],"%s/cameraMatrixEstBA.txt",scratchDir);
   printf("bundleFile: %s\n",bundleFile);
   sprintf(&smoothFile[0],"%s/cameraMatrixEst.txt",scratchDir);
   printf("smoothFile: %s\n",smoothFile);

   bool keyFramesGenerated = false;//9fileExists(bundleFile);

   if (!keyFramesGenerated) {
       // load canonized trajectory, first frame <-> identity matrix
       int numRows;
       float *cameraTrajectory = loadCameraMatrices(posePath,&numRows);
       if (numRows>0) {
           // if trajectory is too long vs available images, cut it ( recording phase often produces +1 trajectory matrices)
           if (numRows > numFrames) numRows = numFrames;
           assert(numRows <= numFrames);
           canonizeTrajectory(cameraTrajectory,numRows);
           printf("camera trajectory loaded: %d poses\n",numRows);
           if (numRows < numFrames) numFrames = numRows;
           if (lastFrame > numFrames-1) lastFrame = numFrames-1;
       } else {
           printf("abort! trajectory is empty!\n");
           if (cameraTrajectory != NULL) delete[] cameraTrajectory;
           exit(1);
       }
       float trajLen = integrateTrajectoryLength(cameraTrajectory,firstFrame,lastFrame);
       printf("trajectory length: %3.3fm\n",trajLen/1000.0f);

       // overriding keyframe selections (temporary functionality):
       std::vector<int> keyNumbers;
       FILE *fuu = fopen("scratch/keylog.txt","rb");
       for (int i = 0; i < 2500; i++) {
           int i0,i1;
           fscanf(fuu,"%d %d\n",&i0,&i1); i1++;
           bool alreadyThere = false;
           for (size_t j = 0; j < keyNumbers.size(); j++) {
               if (keyNumbers[j] == i1) {alreadyThere = true; break;}
           }
           if (!alreadyThere) keyNumbers.push_back(i1);
       }
       for (size_t j = 0; j < keyNumbers.size(); j++) {
           printf("refkey: %d\n",keyNumbers[j]);
           loadImage(datasetPath,keyNumbers[j], rgbImageSmall);
           loadDepthMap(datasetPath,keyNumbers[j],depthMap,calib);
           addKeyFrame(keyNumbers[j],rgbImageSmall,depthMap,calib,&cameraTrajectory[(keyNumbers[j]-1)*16]);
       }
       fclose(fuu);
/*       for (int frameIndex = firstFrame; frameIndex <= lastFrame; frameIndex++) {
           KeyFrame *nearKey = NULL;
           if (keyFrameOccupied(&cameraTrajectory[(frameIndex-1)*16],float(distTol),float(angleTol),&nearKey)) {
               continue;
           }
           loadImage(datasetPath,frameIndex, rgbImageSmall);
           loadDepthMap(datasetPath,frameIndex,depthMap,calib);
           addKeyFrame(frameIndex,rgbImageSmall,depthMap,calib,&cameraTrajectory[(frameIndex-1)*16]);
       }*/
       printf("keys added.\n");
       setupNeighborKeys(float(distTol)*1.1f,float(angleTol)*1.1f);
       printf("neighbors set.\n");
       setupCameraTrajectory(cameraTrajectory,firstFrame,lastFrame);
       printf("trajectory set.\n");
       //integrateDepthMaps(datasetPath,calib,cameraTrajectory);
       printf("depths integrated.\n"); fflush(stdin); fflush(stdout);
       delete[] cameraTrajectory;
   } else {
       printf("loading BA keyframes.\n");
       int numRows=0;
       float *cameraTrajectory = loadCameraMatrices(bundleFile,&numRows);
       for (int frameIndex = 1; frameIndex <= numRows; frameIndex++) {
           loadImage(scratchDir,frameIndex, rgbImageSmall);
           loadDepthMap(scratchDir,frameIndex,depthMap,calib);
           addKeyFrame(frameIndex,rgbImageSmall,depthMap,calib,&cameraTrajectory[(frameIndex-1)*16]);
       }
       delete[] cameraTrajectory;
   }


   cpuPointClouds(calib);
   printf("cpuPointClouds generated.\n"); fflush(stdin); fflush(stdout);
   generateGPUKeyframes(pixelSelectionAmount, m_renderableFlag, undistortRGB, calib);
   printf("gpu keyframes generated.\n"); fflush(stdin); fflush(stdout);
/*
   // also initialize interpolated keyframe (experimental)
   float m[16];
   identity4x4(m);
   interpolatedKey = createKeyFrame(0,keyFrames[0]->distortedRgbImageCPU,keyFrames[0]->depthCPU,calib,&m[0]); //printf("allocated interpolated key: %d\n",interpolatedKey != NULL); fflush(stdin); fflush(stdout);
   interpolatedKey->vbuffer.init(interpolatedKeyPointAmount,true);
   interpolatedKey->vbuffer.setElementsCount(interpolatedKeyPointAmount);*/
}

KeyFrame *KeyFrameModel::getInterpolatedKey() {
    return interpolatedKey;
}

/*
void KeyFrameModel::interpolateKeyAlongTrajectory(float t) {
    int n = keyFrames.size();
    int indexRange = n-1;
    float maxT = 1.0f-1e-5f;
    // check bounds
    if (t < 0.0f) t = 0;
    if (t > maxT) t = maxT;
    // generate floating point index
    float findex = float(indexRange)*t;
    int i0 = (int)findex; float frac = findex - i0;
    int i1 = i0+1;

    std::vector<KeyFrame*> nearKeys;
    std::vector<float> weights;
    nearKeys.push_back(keyFrames[i0]); weights.push_back(1.0f-frac);
    nearKeys.push_back(keyFrames[i1]); weights.push_back(frac);

    interpolateKey(nearKeys,weights);
}

void KeyFrameModel::interpolateKeyAlongTrajectory4(float t) {
    int n = keyFrames.size();
    int indexRange = n-1;
    float maxT = 1.0f-1e-5f;
    // check bounds
    if (t < 0.0f) t = 0;
    if (t > maxT) t = maxT;
    // generate floating point index
    float findex = float(indexRange)*t;

    std::vector<KeyFrame*> nearKeys;
    std::vector<float> weights;

    float scale = 1.0f;

    int i0 = (int)findex; nearKeys.push_back(keyFrames[i0]); weights.push_back(expf(-(findex-i0)*(findex-i0)*scale));
    int i1 = i0+1;        nearKeys.push_back(keyFrames[i1]); weights.push_back(expf(-(findex-i1)*(findex-i1)*scale));

    //printf("%f %f\n",weights[0],weights[1]);
    //fflush(stdin); fflush(stdout);
    int i2 = i0+2;
    if (i2 < n) {
        nearKeys.push_back(keyFrames[i2]); weights.push_back(expf(-(findex-i2)*(findex-i2)*scale));
    }
    int ip = i0-1;
    if (ip >= 0) {
        nearKeys.push_back(keyFrames[ip]); weights.push_back(expf(-(findex-ip)*(findex-ip)*scale));
    }

    interpolateKey(nearKeys,weights);
}

void KeyFrameModel::interpolateKeyFromDatabase(float *T, float maxDist, float maxAngle) {
    std::vector<KeyFrame*> nearKeys;

    findSimilarKeyFrames(T, maxDist, maxAngle, nearKeys, NULL,0);
    std::vector<float> weights; weights.resize(nearKeys.size());

    for (size_t i = 0; i < nearKeys.size(); i++) {
        weights[i] = nearKeys[i]->weight;
    }

    interpolateKey(nearKeys,weights);
}
*/
float KeyFrameModel::getInterpolationTime() {
    return interpolationTimeMillis;
}

void KeyFrameModel::sparseCollection(std::vector<KeyFrame*> &nearKeys,std::vector<float> &weights, bool evenSelection, KeyFrame *interpolatedKey) {
    if (interpolatedKey == NULL) return;
    interpolatedKey->vbuffer.setElementsCount(0);
    interpolatedKey->vbuffer.lock();
    interpolatedKey->vbuffer.lockIndex();

    size_t n = nearKeys.size();
    int totalPoints256 = nearKeys[0]->vbuffer.getElementsCount()/256;
    int pointsRemaining256 = totalPoints256;

    for (int i = 0; i < n; i++) {
        KeyFrame *kf = nearKeys[i];

        int collectedPoints256 = 0;
        if (!evenSelection) {
            collectedPoints256 = int(weights[i]*totalPoints256);
            // the last set must contain the remaining points
            if (i == (n-1)) collectedPoints256 = pointsRemaining256;
            if (collectedPoints256 < 1) continue;
        } else {
            collectedPoints256 = totalPoints256/n;
        }

        kf->vbuffer.lock();
        kf->vbuffer.lockIndex();
        collectPointsCuda(&kf->vbuffer,kf->getAbsBaseDev(), collectedPoints256 ,&interpolatedKey->vbuffer,interpolatedKey->getAbsBaseDev());
        kf->vbuffer.unlockIndex();
        kf->vbuffer.unlock();

        pointsRemaining256 -= collectedPoints256;
    }
    interpolatedKey->vbuffer.unlockIndex();
    interpolatedKey->vbuffer.unlock();
}
/*
void KeyFrameModel::denseBlend(std::vector<KeyFrame*> &nearKeys,std::vector<float> &weights, KeyFrame *interpolatedKey) {
    size_t n = nearKeys.size();
    if (interpolatedKey == NULL) return;
#define BLEND_PERFORMANCE_TEST

#if defined(BLEND_PERFORMANCE_TEST)
    float delay = 0.0f;
    float delays[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaThreadSynchronize();

    cudaEventRecord(start,0);
#endif

    interpolatedKey->vbuffer.lock();
    interpolatedKey->vbuffer.lockIndex();

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[0] += delay; cudaEventRecord(start,0);
#endif


    for (int i = 0; i < n; i++) {
        nearKeys[i]->vbuffer.lock();
        nearKeys[i]->vbuffer.lockIndex();
        nearKeys[i]->grayPyramid.lock();
    }

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[1] += delay; cudaEventRecord(start,0);
#endif


    int totalPoints = nearKeys[0]->vbuffer.getElementsCount();
    int totalPoints256 = totalPoints/256;

    for (int i = 0; i < n; i++) {
        KeyFrame *kf = nearKeys[i];
        collectPointsCuda2(&kf->vbuffer,kf->getAbsBaseDev(), totalPoints256 ,vertexFusionLayersDev+i*(VERTEXBUFFER_STRIDE*totalPoints),interpolatedKey->getAbsBaseDev());
    }

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[2] += delay; cudaEventRecord(start,0);
#endif


    int interpolatedPoints = m_interpolatedKeyPointCount;
    if (interpolatedPoints > totalPoints*n) interpolatedPoints = totalPoints*n;

    interpolatedKey->selectPixelsGPUCompressed2(vertexFusionLayersDev,vertexFusionIndicesDev,totalPoints*n,interpolatedPoints,VERTEXBUFFER_STRIDE);

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[3] += delay; cudaEventRecord(start,0);
#endif


    // replace intensity by average over the keyframes using IIR filter
    setPointIntensityCuda(&interpolatedKey->vbuffer,interpolatedKey->getAbsBaseDev(),nearKeys[0]->getAbsBaseDev(),&nearKeys[0]->grayPyramid);
    //for (int j = 1; j < n; j++) {
    //    KeyFrame *kf = nearKeys[j];
    //    updatePointIntensityCuda(interpolatedKey->vbuffer,interpolatedKey->getAbsBaseDev(),kf->getAbsBaseDev(),&kf->grayPyramid);
    //}
#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[4] += delay; cudaEventRecord(start,0);
#endif

    for (int i = n-1; i >= 0; i--) {
        nearKeys[i]->grayPyramid.unlock();
        nearKeys[i]->vbuffer.unlockIndex();
        nearKeys[i]->vbuffer.unlock();
    }

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[5] += delay; cudaEventRecord(start,0);
#endif


    interpolatedKey->vbuffer.unlockIndex();
    interpolatedKey->vbuffer.unlock();

#if defined(BLEND_PERFORMANCE_TEST)
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[6] += delay; cudaEventRecord(start,0);
#endif

#if defined(BLEND_PERFORMANCE_TEST)
    printf("locki: %3.1f, lockk: %3.1f, collect: %3.1f, select: %3.1f, interp: %3.1f, unlockk: %3.1f, unlocki: %3.1f\n",delays[0],delays[1],delays[2],delays[3],delays[4],delays[5],delays[6]);
    fflush(stdin); fflush(stdout);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

}
*/

void KeyFrameModel::sortKeys(std::vector<KeyFrame*> &nearKeys, std::vector<float> &weights) {

    int n = (int)nearKeys.size();

    std::vector<int> indices; indices.resize(n);
    std::vector<float> weightsTemp; weightsTemp.resize(n);
    std::vector<KeyFrame*> nearKeysTemp; nearKeysTemp.resize(n);

    // setup identity mapping
    for (int index = 0; index < n; index++) {
        indices[index] = index;
        nearKeysTemp[index] = nearKeys[index];
        weightsTemp[index] = weights[index];
    }

    bubbleSort(weights,indices);

    // store sorted vectors
    for (int index = 0; index < n; index++) {
        weights[index] = weightsTemp[indices[n-1-index]];
        nearKeys[index] = nearKeysTemp[indices[n-1-index]];
    }
}
/*
void KeyFrameModel::interpolateKey(std::vector<KeyFrame*> &nearKeys,std::vector<float> &weights) {
    if (interpolatedKey == NULL) return;
    size_t n = nearKeys.size();
    size_t nn = weights.size();
    if (n != nn) {
        printf("interpolateKeys error: mismatch in vector sizes!\n");
        return;
    }

    interpolationNeighbors.clear();

    sortKeys(nearKeys,weights);

    // make sure poses fit into allocated arrays
    const int maxKeys = VERTEX_FUSION_LAYER_COUNT;
    if (n > maxKeys) n = maxKeys;

    float weightSum = 0.0f;
    for (size_t i = 0; i < n; i++) {
//        printf("weight %d : %f\n",i,weights[i]);
        weightSum += weights[i];
    }

    float qarr[16*maxKeys];
    float tarr[3*maxKeys];


    // convert data into temp arrays
    for (size_t i = 0; i < n; i++) {
        KeyFrame *kf = nearKeys[i];
        float pose[16];
        transpose4x4(&kf->Tbase[0],&pose[0]);
        rot2Quaternion(&pose[0],4,&qarr[i*4]);
        tarr[i*3+0] = pose[3];
        tarr[i*3+1] = pose[7];
        tarr[i*3+2] = pose[11];
        // also normalize weights
        weights[i] /= weightSum;
        nearKeys[i]->weight = weights[i];
        interpolationNeighbors.push_back(nearKeys[i]);

    }

    float interQ[4] = {0,0,0,0}; float interT[3] = {0,0,0};
    for (size_t i = 0; i < n; i++) {
        interT[0]  += tarr[i*3+0]*weights[i];
        interT[1]  += tarr[i*3+1]*weights[i];
        interT[2]  += tarr[i*3+2]*weights[i];
        interQ[0]  += qarr[i*4+0]*weights[i];
        interQ[1]  += qarr[i*4+1]*weights[i];
        interQ[2]  += qarr[i*4+2]*weights[i];
        interQ[3]  += qarr[i*4+3]*weights[i];
    }
    float lenQ = sqrtf(interQ[0]*interQ[0]+interQ[1]*interQ[1]+interQ[2]*interQ[2]+interQ[3]*interQ[3]);
    interQ[0] /= lenQ; interQ[1] /= lenQ; interQ[2] /= lenQ; interQ[3] /= lenQ;

    float interPose[16]; identity4x4(interPose);
    quaternion2Rot(interQ,interPose);
    interPose[3]  = interT[0];
    interPose[7]  = interT[1];
    interPose[11] = interT[2];

    interpolatedKey->updateTransform(interPose);

    float interPoseInv[16];
    invertRT4(interPose,interPoseInv);

    //printf("%d\n",nearKeys[0]->vbuffer.getElementsCount());
    //fflush(stdin);
    //fflush(stdout);

    float delay = 0.0f;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // sparseCollection merely picks n points from each set as is
//    sparseCollection(nearKeys,weights, true, interpolatedKey);

    // denseBlend integrates point cloud data smoothly
    //denseBlend(nearKeys,weights, interpolatedKey);

    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); interpolationTimeMillis = delay;

}*/

std::vector<KeyFrame*> *KeyFrameModel::getInterpolationNeighbors() {
    return &interpolationNeighbors;
}
/*
void KeyFrameModel::setInterpolatedKey(float *m) {

}
*/

float KeyFrameModel::projectionDistance(float *relT, float *testObj, int nVertices) {
    float *K = &m_calib.getCalibData()[KR_OFFSET];
    float error = 0.0f;
    for ( int i = 0; i < nVertices; i++) {
        float p1[3],p2[3],r1[3],r2[3];
        p1[0] = testObj[i*3+0];
        p1[1] = testObj[i*3+1];
        p1[2] = testObj[i*3+2];

        transformRT3(relT, p1, p2);

        /*
        float dX = p1[0]-p2[0];
        float dY = p1[1]-p2[1];
        float dZ = p1[2]-p2[2];

        error += dX*dX;
        error += dY*dY;
        error += dZ*dZ;*/


        p1[0] /= p1[2]; p1[1] /= p1[2]; p1[2] = 1;
        p2[0] /= p2[2]; p2[1] /= p2[2]; p2[2] = 1;

        matrixMultVec3(K,p1,r1);
        matrixMultVec3(K,p2,r2);

        float dX = r1[0]-r2[0];
        float dY = r1[1]-r2[1];
        error += dX*dX+dY*dY;
    }
    return error / float(nVertices);
}

bool KeyFrameModel::keyFrameOccupied(float *Tt, float maxDist, float maxAngle, KeyFrame **keyFrame) {
    *keyFrame = NULL;
    if (Tt == NULL) return NULL;
    float T[16];
    transpose4x4(&Tt[0],&T[0]);
    std::vector<KeyFrame*>::iterator li;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kCur = (KeyFrame*)*li;
        if (!kCur->visible) continue;
        float relativeT[16];
        kCur->getRelativeTransform(T, &relativeT[0]);
        float dist,angle;
        poseDistance(&relativeT[0], &dist,&angle);
        if (dist < maxDist && angle < maxAngle) { *keyFrame = kCur; return true; }
    }
    return false;
}


void KeyFrameModel::enumerateNearPoses(float *T, float maxDist, float maxAngle, std::vector<KeyFrame*> &closePose, KeyFrame *rejectedKey) {
    std::vector<KeyFrame*>::iterator li;
    //printf("enumerating near poses! (%d candidates)\n",int(keyFrames.size()));
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kCur = (KeyFrame*)*li;
        if (!kCur->visible) continue;
        float relativeT[16];
        kCur->getRelativeTransform(T, &relativeT[0]);
        //dumpMatrix("candidate pose",&kCur->Tbase[0],4,4);
        float dist,angle;
        poseDistance(&relativeT[0], &dist,&angle);
        if (dist < maxDist && angle < maxAngle && kCur != rejectedKey) {closePose.push_back(kCur); /*printf("%d : %f %f added to potential set\n",index,dist,angle);*/ }
    }
    //fflush(stdin); fflush(stdout);
}

KeyFrame *KeyFrameModel::findSimilarKeyFrame(float *T, float maxDist, float maxAngle, float *extTestObj, int nVerticesExt) {
    if (T == NULL) return NULL;

    float *testObj = extTestObj;
    int nVertices = nVerticesExt;

    if (testObj == NULL) {
        testObj = &gridObject[0];
        nVertices = nGridSize;
    }
    if (nVertices < 1) return NULL;

    std::vector<KeyFrame*> closePose;
    int index = 0;

    enumerateNearPoses(T, maxDist, maxAngle, closePose);

//    printf("%d nearest keys found!\n",closePose.size());

    float minScore = FLT_MAX; KeyFrame *bestKey = NULL;
    std::vector<KeyFrame*>::iterator ki;
    for (ki = closePose.begin(); ki != closePose.end(); ki++) {
        KeyFrame *kCur = *ki;
        if (!kCur->visible) continue;
        float relativeT[16];
        kCur->getRelativeTransform(T, &relativeT[0]);
        float score = projectionDistance(&relativeT[0],testObj,nVertices);
        //float score = poseDistance(T,kCur->T,40.0f);
        if (score < minScore) {
            minScore = score;
            bestKey = kCur;
        }
    }

    return bestKey;
}


void KeyFrameModel::bubbleSort(std::vector<float> &vals, std::vector<int> &indices)
{
    size_t arraySize = vals.size();
    if (arraySize < 2) return;

    for (size_t i = (arraySize - 1); i > 0; i--) {
        for (size_t j = 1; j <= i; j++) {
            if (vals[indices[j-1]] > vals[indices[j]]) {
                int temp = indices[j-1];
                indices[j-1] = indices[j];
                indices[j] = temp;
            }
        }
    }
}

void KeyFrameModel::findSimilarKeyFrames(float *T, std::vector<KeyFrame*> &similarKeys, float *testObj, int nVertices) {
    findSimilarKeyFrames(T, distTol, angleTol, similarKeys, testObj, nVertices);
}

void KeyFrameModel::findSimilarKeyFramesLarge(float *T, std::vector<KeyFrame*> &similarKeys, float *testObj, int nVertices) {
    findSimilarKeyFrames(T, distTol*3, angleTol*3, similarKeys, testObj, nVertices);
}


void KeyFrameModel::findSimilarKeyFrames(float *T, float maxDist, float maxAngle, std::vector<KeyFrame*> &similarKeys, float *extTestObj, int nVerticesExt) {
    if (T == NULL) return;
//    printf("distTol: %f, angleTol: %f\n",maxDist,maxAngle);

    float *testObj = extTestObj;
    int nVertices = nVerticesExt;

    if (testObj == NULL) {
        testObj = &gridObject[0];
        nVertices = nGridSize;
    }
    if (nVertices < 1) return;

    std::vector<KeyFrame*> closePose;

    similarKeys.clear();
    enumerateNearPoses(T, maxDist, maxAngle, closePose);

    // nothing found?
    if (closePose.size() < 1) return;

    std::vector<float> weights; weights.resize(closePose.size()); //weights.clear();
    std::vector<int> indices; indices.resize(closePose.size());   //indices.clear();
    similarKeys.resize(closePose.size());

    int index = 0; //float weightSum = 0.0f;
    float wMin = FLT_MAX; float wMax = FLT_MIN;
    std::vector<KeyFrame*>::iterator ki;
    for (ki = closePose.begin(); ki != closePose.end(); ki++,index++) {
        KeyFrame *kCur = *ki;
        float relativeT[16];
        kCur->getRelativeTransform(T, &relativeT[0]);
        float w = projectionDistance(&relativeT[0],testObj,nVertices);
        if (w < wMin) wMin = w;
        if (w > wMax) wMax = w;
        weights[index] = w;
        indices[index] = index;
//        printf("weight[%d]:%f\n",index,w);
    }

    bubbleSort(weights,indices);

//    float invWeightSum = 0.0f;
    index = 0;
    for (ki = similarKeys.begin(); ki != similarKeys.end(); ki++,index++) {
        float w = (weights[indices[index]]-wMin)/(wMax-wMin);
        KeyFrame *kf = closePose[indices[index]]; kf->weight = 1.0f - w; //invWeightSum += kf->weight;
        similarKeys[index] = kf;
    }
}

void KeyFrameModel::resetTransforms() {
    std::vector<KeyFrame*>::iterator li;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kf = (KeyFrame*)*li;
        kf->setupCPUTransform();
    }
}


KeyFrame *KeyFrameModel::getKeyFrame(int index) {
    std::vector<KeyFrame*>::iterator li;
    int i = 0;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kf = (KeyFrame*)*li;
        if (i == index) return kf;
        i++;
    }
    return NULL;
}

KeyFrame *KeyFrameModel::getKeyFrameID(int index) {
    std::vector<KeyFrame*>::iterator li;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kf = (KeyFrame*)*li;
        if (kf->id == index) return kf;
    }
    return NULL;
}


void KeyFrameModel::disableKeyFrame(int id) {
    KeyFrame *kf = getKeyFrameID(id);
    if (kf) kf->visible = false;
}

int KeyFrameModel::getKeyFrameCount() {
    return keyFrames.size();
}


void KeyFrameModel::removeKeyframes() {
    int index = 0;
    std::vector<KeyFrame*>::iterator li;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++,index++) {
        KeyFrame *kf = (KeyFrame*)(*li);
        printf("removing key %d\n",index); fflush(stdin); fflush(stdout);
        kf->release(); delete kf;
    }
    keyFrames.clear();
}

void KeyFrameModel::release() {
    removeKeyframes();

    if (interpolatedKey != NULL) {
        interpolatedKey->release(); delete interpolatedKey; interpolatedKey = NULL;
    }

    if (vbufferTemp != NULL) {
        vbufferTemp->release();
        delete vbufferTemp;
        vbufferTemp = NULL;
    }
    if (vertexFusionLayersDev != NULL) cudaFree(vertexFusionLayersDev); vertexFusionLayersDev = NULL;
    if (vertexFusionIndicesDev != NULL) cudaFree(vertexFusionIndicesDev); vertexFusionIndicesDev = NULL;

    if (calibDataDev != NULL) cudaFree(calibDataDev); calibDataDev = NULL;

    if (imRGBDev != NULL) cudaFree(imRGBDev); imRGBDev = NULL;
    if (imDispDev != NULL) cudaFree(imDispDev); imDispDev = NULL;
    if (imDepthDev != NULL) cudaFree(imDepthDev); imDepthDev = NULL;
    if (trajectory!=NULL) delete trajectory; trajectory = NULL;
    if (depth1C.data != NULL) depth1C.releaseData();
    if (savedTransforms != NULL) delete[] savedTransforms;
}

void KeyFrameModel::pushTransforms() {
    if (savedTransforms != NULL) delete[] savedTransforms;

    int cnt = getKeyFrameCount();
    savedTransforms = new float[16*cnt];

    for (int i = 0; i < cnt; i++) {
        KeyFrame *kf = getKeyFrame(i);
        memcpy(&savedTransforms[i*16],&kf->T[0],sizeof(float)*16);
    }

}

void KeyFrameModel::popTransforms() {
    assert(savedTransforms != NULL);

    int cnt = getKeyFrameCount();
    for (int i = 0; i < cnt; i++) {
        KeyFrame *kf = getKeyFrame(i);
        memcpy(&kf->T[0],&savedTransforms[i*16],sizeof(float)*16);
        //memcpy(&kf->Tbase[0],&savedTransforms[i*16],sizeof(float)*16);
    }
}


void KeyFrameModel::setIterationCounts(int *nIter) {
    for (int i = 0; i < 3; i++) nIterations[i] = nIter[i];
    int cnt = getKeyFrameCount();
    for (int i = 0; i < cnt; i++) {
        KeyFrame *kf = getKeyFrame(i);
        kf->setIterationCounts(nIterations);
    }
}


KeyFrameModel::~KeyFrameModel() {
}

int KeyFrameModel::getWidth()
{
        return width;
}

int KeyFrameModel::getHeight()
{
        return height;
}

int KeyFrameModel::getDepthWidth()
{
        return width;

}

int KeyFrameModel::getDepthHeight()
{
        return height;
}

void KeyFrameModel::extractFeatures() {
    std::vector<KeyFrame*>::iterator li;
    for (li = keyFrames.begin(); li != keyFrames.end(); li++) {
        KeyFrame *kf = (KeyFrame*)*li;
        kf->extractFeatures();
    }
}

unsigned int KeyFrameModel::reconstructGlobalPointCloud(float **points, float **normals) {

    *points = NULL;
    *normals = NULL;
    if (keyFrames.size() < 1) return 0;

    int useNKeys = keyFrames.size(); //useNKeys = 2;

    unsigned int cnt = keyFrames[0]->xyzImage.cols * keyFrames[0]->xyzImage.rows*useNKeys;
    *points  = new float[cnt*3];
    *normals = new float[cnt*3];
    unsigned int pindex = 0;
    float *pptr = *points;
    float *nptr = *normals;

    for (int ki = 0; ki < useNKeys; ki++) {
        KeyFrame *kf = keyFrames[ki];
        float *xyz    = (float*)kf->xyzImage.ptr();
        float *normal = (float*)kf->normalImage.ptr();
        float T[16],N[16],T0[16],Tc[16];
        transpose4x4(&kf->T[0],&Tc[0]); //dumpMatrix("Tc",Tc,4,4);
        transpose4x4(&keyFrames[0]->T[0],&T0[0]); //dumpMatrix("T0",T0,4,4);
        relativeTransform(&Tc[0],&T0[0],&T[0]); //dumpMatrix("Tr",T,4,4);
        memcpy(&N[0],&T[0],sizeof(float)*16); N[3] = 0; N[7] = 0; N[11] = 0;
        int offset = 0;
        float minDistance = 300.0f*300.0f;
        float maxDistance = 8000.0f*8000.0f;
        for (int j = 0; j < kf->xyzImage.rows; j++) {
            for (int i = 0; i < kf->xyzImage.cols; i++,offset+=3,pindex+=3) {
                transformRT3(&T[0],&xyz[offset],&pptr[pindex]);
                float z2 = xyz[offset+2]*xyz[offset+2];
                if (z2 < minDistance) z2 = maxDistance;
                if (z2 > maxDistance) z2 = maxDistance;
                float confidence = 1.0f;//1.0f - (3.0f*z2/maxDistance); if (confidence < 0.0f) confidence = 0.0f;
                transformRT3(&N[0],&normal[offset],&nptr[pindex]); nptr[pindex+0] *= confidence; nptr[pindex+1] *= confidence; nptr[pindex+2] *= confidence;
            }
        }
    }

    printf("reconstructed point cloud with %d points\n",cnt); fflush(stdin); fflush(stdout);
    return cnt;
}

bool pointInBox(float *p, float *boxParams) {

    float b[3],t[3];
    b[0] = p[0] - boxParams[0];
    b[1] = p[1] - boxParams[1];
    b[2] = p[2] - boxParams[2];

    float mx[16],my[16];
    matrix_from_euler4(-deg2rad(boxParams[6]),0,0,&mx[0]);
    matrix_from_euler4(0,-deg2rad(boxParams[7]),0,&my[0]);

    transformRT3(mx,b,t);
    transformRT3(my,t,b);

    float hx = boxParams[3]/2.0f;
    float hy = boxParams[4]/2.0f;
    float hz = boxParams[5]/2.0f;

    if (b[0] < -hx) return false;
    if (b[1] < -hy) return false;
    if (b[2] < -hz) return false;

    if (b[0] > hx) return false;
    if (b[1] > hy) return false;
    if (b[2] > hz) return false;

    return true;
}

int clipCloud(float *points,float *normals,int cnt,float *boxParams) {

    int dstOffset = 0;
    for (int i = 0; i < cnt; i++) {
        if (pointInBox(&points[i*3],boxParams)) {
            points[dstOffset*3+0] = points[i*3+0];
            points[dstOffset*3+1] = points[i*3+1];
            points[dstOffset*3+2] = points[i*3+2];
            normals[dstOffset*3+0] = normals[i*3+0];
            normals[dstOffset*3+1] = normals[i*3+1];
            normals[dstOffset*3+2] = normals[i*3+2];
            dstOffset++;
        }
    }
    return dstOffset;
}


void generateVisualization(float *points, float *normals, int cnt, VertexBuffer2 **vbuf) {
    VertexBuffer2 *buf = *vbuf;
    if (buf == NULL) {
        *vbuf = new VertexBuffer2(cnt,true,VERTEXBUFFER_STRIDE,"visualization buffer"); buf = *vbuf;
    } else {
        buf->release();
        delete buf;
        *vbuf = new VertexBuffer2(cnt,true,VERTEXBUFFER_STRIDE,"visualization buffer"); buf = *vbuf;
    }
    for (int i = 0; i < cnt; i++) {
        buf->addVertex(points[i*3+0],points[i*3+1],points[i*3+2],1,0,0);
    }
    buf->upload();
}

void KeyFrameModel::savePolygonModel(char *filename, int numThreads, int depth, bool useClipBox, float *boxParams, VertexBuffer2 **vbuf, TriangleBuffer2 **tribuf) {
    pr::Octree<2> tree;
    tree.threads = numThreads;
    int maxSolveTreeDepth = depth; // max voxel grid size := 2^n
    int minSolveTreeDepth = depth-3; // min octree subdivision level
    // The depth at which a block Gauss-Seidel solver is used
    int treeDepth = depth-1;
    // isoDivideDepth specifies the depth at which a block iso-surface extractor should be used to extract the iso-surface.
    // Using this parameter helps reduce the memory overhead at the cost of a small increase in extraction time.
    // In practice, we have found that for reconstructions of depth 9 or higher a subdivide depth of 7 or 8 can greatly reduce the memory usage.
    int isoDivideDepth = depth-1;
    // This floating point value specifies the minimum number of sample points that should fall within an octree node as the octree construction is adapted to sampling density.
    // For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples, larger values in the range [15.0 - 20.0] may be needed to provide a smoother,
    // noise-reduced, reconstruction. The default value is 1.0.
    Real samplesPerNode = 1;//4;
    // Specifies the factor of the bounding cube that the input samples should fit into.
    float scale = 1.1f;
    // Enabling this flag tells the reconstructor to use the size of the normals as confidence information.
    // When the flag is not enabled, all normals are normalized to have unit-length prior to reconstruction.
    bool confidence=false;//true;
    // This floating point value specifies the importants that interpolation of the point samples is given in the formulation of the screened Poisson equation.
    // The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0.
    // The default value for this parameter is 4.
    float pointWeight = 0.0f;
    // This specifies the exponent scale for the adaptive weighting.
    float adaptiveExponent = 1.0f;
    int minIters = 24;
    float solverAccuracy = float(1e-6);
    int fixedIters = -1;
    XForm4x4<Real> xForm = XForm4x4< Real >::Identity();
    XForm4x4<Real> iXForm = xForm.inverse();
    float mtx4x4[16]; identity4x4(&mtx4x4[0]);

    if( treeDepth < minSolveTreeDepth )
    {
        fprintf( stderr , "[WARNING] subdivision level must be at least as large as %d\n" , minSolveTreeDepth );
        treeDepth = minSolveTreeDepth;
    }
    if( isoDivideDepth < minSolveTreeDepth )
    {
        fprintf( stderr , "[WARNING] isodivision value must be at least as large as %d\n" , isoDivideDepth );
        isoDivideDepth = minSolveTreeDepth;
    }
    TreeOctNode::SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );

    int kernelDepth = depth;

    tree.setBSplineData( depth , 1 );
    if( kernelDepth>depth )
    {
        fprintf( stderr,"[ERROR] kernelDepth can't be greater than %d\n" , maxSolveTreeDepth );
        return;
    }

    float *points = NULL;
    float *normals = NULL;
    int cnt = reconstructGlobalPointCloud(&points,&normals);
    if (cnt && useClipBox) {
        cnt = clipCloud(points,normals,cnt,boxParams);
    }
    // store visualization into given buffer
   // generateVisualization(points,normals,cnt,vbuf);
    int pointCount = tree.setTree( points,normals,cnt , maxSolveTreeDepth , minSolveTreeDepth , kernelDepth , samplesPerNode , scale, confidence, pointWeight, adaptiveExponent,xForm);
    delete[] points; delete[] normals;

    tree.ClipTree();
    tree.finalize( isoDivideDepth );

    printf( "Input Points: %d\n" , pointCount );
    printf( "Leaves/Nodes: %d/%d\n" , tree.tree.leaves() , tree.tree.nodes() );
    printf( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );

    if (pointCount > 1000) {
        int maxMemoryUsage = tree.maxMemoryUsage;
        tree.maxMemoryUsage=0;
        tree.SetLaplacianConstraints();
        printf( "Memory Usage after Laplacian: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        tree.maxMemoryUsage=0;
        tree.LaplacianMatrixIteration( treeDepth, false , minIters, solverAccuracy, maxSolveTreeDepth , fixedIters);
        printf( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        CoredFileMeshData mesh;
        tree.maxMemoryUsage=0;
        Real isoValue = tree.GetIsoValue();
        printf( "Iso-Value: %e\n" , isoValue );

        tree.maxMemoryUsage = 0;
        tree.GetMCIsoTriangles( isoValue , isoDivideDepth , &mesh , 0 , 1 , false , true );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        char buf[512];
        sprintf(buf,"%s/%s",scratchDir,filename);
        writeWaveFront(buf, &mesh, &mtx4x4[0], tribuf);
    } else {
        printf("not enough points to save %s\n",filename);
    }
    fflush(stdin); fflush(stdout);
}

void KeyFrameModel::writeWaveFrontMaterials(const char *filename, char *materialFile) {
    char materialPath[512]; memset(materialPath,'\0',512);
    strcpy(materialPath,filename);
    char *marker = strrchr(materialPath,'/')+1;
    strcpy(marker,materialFile);

    printf("materialpath: %s\n",materialPath);

    FILE *f = fopen(materialPath,"wb");

    fprintf(f,"newmtl material0\n");
    fprintf(f,"Ka 1.000 1.000 1.000\n");
    fprintf(f,"Kd 1.000 1.000 1.000\n");
    fprintf(f,"Ks 0.000 0.000 0.000\n");
    fprintf(f,"Ke 0.000 0.000 0.000\n");
    fprintf(f,"Ns 0.000\n");
    fprintf(f,"d 1.000\n");
    fprintf(f,"illum 2\n");
    // store large texture with 640x480 tiles instead of 320x240
    fprintf(f,"map_Ka textureLarge.png\n");
    fprintf(f,"map_Kd textureLarge.png\n");
    //    fprintf(f,"map_Bump Body_Shirt_DecolleteBump.tga\n");
    fclose(f);
    fflush(stdin); fflush(stdout);
}

float convexArea(float *uv, int nVerts) {
    vector<Point2f> originalPoints;   // Your original points
    Point2f pt;
    for (int i = 0; i < nVerts; i++) {
        pt.x = uv[i*2+0]; pt.y = uv[i*2+1];
        originalPoints.push_back(pt);
    }

    vector<Point2f> ch;  // Convex hull points
    // Calculate convex hull of original points (which points positioned on the boundary)
    cv::convexHull(Mat(originalPoints),ch,false);
    // false parameter is used to organize the points in clockwise direction

    float area = 0;
    // Now calculate the area of convex hull 'ch':
    for (size_t i = 0; i < ch.size(); i++) {
        int next_i = (i+1)%(ch.size());
        float dX   = ch[next_i].x - ch[i].x;
        float avgY = (ch[next_i].y + ch[i].y)/2;
        area += dX*avgY;  // This is the integration step.
    }
    if (area > 0) area = 0.0f;
    return -area;
}
/*
void KeyFrameModel::generateUV(float *vertices, unsigned int *faceIndex, int nFaces, FILE *f)
{
    float *K    = &m_calib.getCalibData()[KR_OFFSET];
    float *kc   = &m_calib.getCalibData()[KcR_OFFSET];
    float *KIR  = &m_calib.getCalibData()[KL_OFFSET];
    float *TRL  = &m_calib.getCalibData()[TRL_OFFSET];

    int width  = m_calib.getWidth();
    int height = m_calib.getHeight();

    float *faceArea  = new float[nFaces];
    float *faceUV    = new float[nFaces*16*2];
    int   *bestKey   = new int[nFaces];

    // reset tables
    for (int face = 0; face < nFaces; face++) {
        faceArea[face] = 0.0f;
        bestKey[face]  = 0;
        for (int vert  = 0; vert < 16; vert++) {
            faceUV[face*16*2+2*vert+0] = 0.0f;
            faceUV[face*16*2+2*vert+1] = 0.0f;
        }
    }

    // associate faces to views with largest screen-space area
    for (size_t i = 0; i < keyFrames.size(); i++) {
        float mtxT[16],mtx[16];
        transpose4x4(&keyFrames[i]->Tbase[0],&mtxT[0]);
        invertRT4(&mtxT[0],&mtx[0]);
        float *zData = (float*)keyFrames[i]->depthCPU.ptr();

        for (int face = 0; face < nFaces; face++) {
            unsigned int nVerts = faceIndex[face*16+0];
            float uv[16*2];
            bool visible = true;
            float avgDepth = 0.0f;
            float avgU = 0.0f;
            float avgV = 0.0f;
            // generate uv coordinates into keyframe view i:
            for (int vert = 0; vert < nVerts; vert++) {
                float *p = &vertices[3*faceIndex[face*16+vert+1]];
                float v[3],w[3];
                transformRT3(mtx,p,v);
                transformRT3(TRL,v,w); // generate point in IR view too for depth inspection
                if (w[2] > -300) visible = false;
                else {
                    avgDepth += w[2];
                    // normalize projective points
                    v[0] /= v[2]; v[1] /= v[2]; v[2] = 1;
                    w[0] /= w[2]; w[1] /= w[2]; w[2] = 1;
                    // little bit ad hoc test to see that lens distortion does not map points outside fov into image domain:
                    if (v[0]*v[0]+v[1]*v[1] < 1.0f) {
                        distortPointCPU(v,kc,K,&uv[vert*2]);                     
                        // use undistorted coordinates to sample depthCPU image (occlusion test)
                        avgU += KIR[0]*w[0]+KIR[2];
                        avgV += KIR[4]*w[1]+KIR[5];
                    } else visible = false;
                }
            }
            // guarantee uv coordinates are in the image domain:
            if (visible) {
                bool inBounds = true;
                // boundary test
                for (int vert = 0; vert < nVerts; vert++) {
                    float u = uv[vert*2+0];
                    float v = uv[vert*2+1];
                    if (u < 1.0f || u > (width-2) || v < 1.0f || v > (height-2)) {
                        inBounds = false;
                        break;
                    }
                }
                // occlusion test
                if (inBounds) {
                    avgU /= nVerts; avgV /= nVerts; avgDepth /= nVerts;
                    int ui = int(avgU);
                    int vi = int(avgV);
                    if (ui > 0 && ui < (width-1) && vi > 1 && vi < (height-1) ) {
                        // depths are positive in stored maps
                        float z =  zData[ui+vi*width];
                        float z2 = zData[(ui+1)+vi*width];
                        float z3 = zData[(ui+1)+(vi+1)*width];
                        float z4 = zData[ui+(vi+1)*width];
                        if (z2 < z) z = z2;
                        if (z3 < z) z = z3;
                        if (z4 < z) z = z4;
                        if (-avgDepth-200.0f > z) {
                            inBounds = false;
                        }
                    }
                }
                // if these uv coordinates are better than anything before, select them for this polygon:
                if (inBounds) {
                    float area = convexArea(uv,nVerts);
//                    if (i > faceArea[face]) {
//                        faceArea[face] = i;
                    if (area > faceArea[face]) {
                        faceArea[face] = area;
                        bestKey[face] = i;
                        memcpy(&faceUV[face*16*2],&uv[0],sizeof(float)*nVerts*2);
                    }
                }
            }
        }
    }

    // finally convert uv coordinates to larger tiled texture space
    int keyCounts[256]; memset(keyCounts,0,sizeof(int)*256);

//    int saveMesh = 0;
//    Mat rgbImage(240,320,CV_8UC3); memcpy(rgbImage.ptr(),keyFrames[saveMesh]->distortedRgbImageCPU.ptr(),320*240*3);

    for (int face = 0; face < nFaces; face++) {
        int key = bestKey[face];
        if (faceArea[face] > 0.0f) keyCounts[key]++;
      //  else continue;

        int nVerts = faceIndex[face*16+0];
        int baseUV = face*16*2;

        // compute tile coords
        int tileX = key%6; int tileY = key/6;
        float textureUBase = tileX*width;
        float textureVBase = tileY*height;

        for (int vert = 0; vert < nVerts; vert++) {
            //convert uv into normalized coordinates in big texture
            float u1 = faceUV[baseUV+vert*2+0];
            float v1 = faceUV[baseUV+vert*2+1];
            float u2 =        (u1+textureUBase)/2048.0f;
            float v2 = 1.0f - (v1+textureVBase)/2048.0f;
            fprintf(f,"vt %f %f\n",u2,v2);
        }
    }
//    imwrite("scratch/kusi.ppm",rgbImage);
    for (size_t i = 0; i < keyFrames.size(); i++) printf("key %d faces : %d\n",int(i),keyCounts[i]);

    delete[] faceArea;
    delete[] bestKey;
    delete[] faceUV;
    fflush(stdin); fflush(stdout);

    return;
}*/

void recurseNeighbors(int face,int maxNeighbourCount,int *faceLinks, std::vector<int> &neighborIndex, int maxIter) {
    if (maxIter < 0) return;

    for (int i = 0; i < maxNeighbourCount; i++)
    if (faceLinks[face*maxNeighbourCount+i] != -1) {
        int ni = faceLinks[face*maxNeighbourCount+i];
        bool alreadyExists = false;
        for (size_t j = 0; j < neighborIndex.size(); j++) {
            if (neighborIndex[j] == ni) {
                alreadyExists = true;
                break;
            }
        }
        if (!alreadyExists) { neighborIndex.push_back(ni); recurseNeighbors(ni,maxNeighbourCount,faceLinks,neighborIndex,maxIter-1); }
    }
}


void reguralizeUVMapping(int nFaces,int *keys, int *faceLinks, const int maxNeighbourCount, bool *validUV, int maxRecursion, int maxIter) {
    // regularize uv mapping in several iterations
    for (int iterations = 0; iterations < maxIter; iterations++)
    {
        for (int face = 0; face < nFaces; face++) {            
            // make array of labels in the neighbourhood:
            std::vector<int> neighborIndex;
            std::vector<int> neighborhoodKeys;
            // add first
            neighborIndex.push_back(face);
            // add neighbors recursively (neglect existing ones)
            recurseNeighbors(face,maxNeighbourCount,faceLinks,neighborIndex,maxRecursion);

            for (size_t ni = 0; ni < neighborIndex.size(); ni++) {
                int key = keys[neighborIndex[ni]];
                if (key >= 0 && validUV[key*nFaces+face]) {
                    neighborhoodKeys.push_back(key);
                }
            }
            if (neighborhoodKeys.size() > 0) {
                std::vector<int> histogram;
                histogram.resize(neighborhoodKeys.size());

                int idx = (int)neighborhoodKeys.size();
                // make statistics of labels
                for (int j = 0; j < idx; j++) {
                    histogram[j] = 1;
                    for (int i = j+1; i < idx; i++) {
                        //if ( i == j ) continue;
                        if (neighborhoodKeys[i] == neighborhoodKeys[j]) histogram[j] = histogram[j]+1;
                    }
                }

                // find the most dominant label, which has valid UV mapping for the current face:
                int bestIndex = 0, bestScore = 0;
                for (int j = 0; j < idx; j++) {
                    if (histogram[j] > bestScore) {
                        bestScore = histogram[j];
                        bestIndex = j;
                    }
                }
                // re-label the pivot face using the dominant label:

                keys[face] = neighborhoodKeys[bestIndex];
            }
        }
    }
}
/*
void KeyFrameModel::initializeUVImages(std::map<int,Mat*> &uvImages) {
    for (size_t i = 0; i < keyFrames.size(); i++) {
        Mat *rgbImage = new cv::Mat(240,320,CV_8UC3);
        memcpy(rgbImage->ptr(),keyFrames[i]->distortedRgbImageCPU.ptr(),320*240*3);
        uvImages[i] = rgbImage;
    }
}

void KeyFrameModel::releaseUVImages(std::map<int,Mat*> &uvImages) {
    char buf[512];
    for (size_t i = 0; i < keyFrames.size(); i++) {
        Mat *rgbImage = uvImages[i];
        sprintf(buf,"scratch/uvmap%04d.ppm",int(i));
        imwrite(buf,*rgbImage);
        rgbImage->release(); delete rgbImage; uvImages[i] = NULL;
    }
}*/


void KeyFrameModel::generateUV3(float *vertices, unsigned int *vertexIndex3, int nFaces, FILE *f)
{
    if (nFaces <= 0 || keyFrames.size() < 1) return;
    float *K    = &m_calib.getCalibData()[KR_OFFSET];
    float *kc   = &m_calib.getCalibData()[KcR_OFFSET];
    float *KIR  = &m_calib.getCalibData()[KL_OFFSET];
    float *TRL  = &m_calib.getCalibData()[TRL_OFFSET];

    int width  = m_calib.getWidth();
    int height = m_calib.getHeight();
    const int maxNeighbourCount = 32;
    float *faceArea      = new float[nFaces];
    float *faceUV        = new float[nFaces*3*2];
    int   *bestKey       = new int[nFaces]; memset(bestKey,0,sizeof(int)*nFaces);
    int   *vertFaces     = new int[nFaces*maxNeighbourCount]; for (int i = 0; i < nFaces*maxNeighbourCount; i++) vertFaces[i] = -1;
    int   *faceLinks     = new int[nFaces*maxNeighbourCount]; for (int i = 0; i < nFaces*maxNeighbourCount; i++) faceLinks[i] = -1;

    // uvMap contains the mesh uv mapping to all views
    float *uvMap = new float[6*nFaces*keyFrames.size()];
    bool *validUV = new bool[nFaces*keyFrames.size()]; for (int i = 0; i < nFaces*keyFrames.size(); i++) validUV[i] = false;

    // reset tables
    for (int face = 0; face < nFaces; face++) {
        faceArea[face] = 0.0f;//FLT_MAX;//0.0f;
        bestKey[face]  = 0;
        for (int vert  = 0; vert < 3; vert++) {
            faceUV[face*3*2+2*vert+0] = 0.0f;
            faceUV[face*3*2+2*vert+1] = 0.0f;
        }
        int i0 = vertexIndex3[face*3+0];
        int i1 = vertexIndex3[face*3+1];
        int i2 = vertexIndex3[face*3+2];
        // link vertex to face array which share it (max 3 entries)
        int j = 0;
        j = 0; while (vertFaces[i0*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) j++; vertFaces[i0*maxNeighbourCount+j] = face;
        j = 0; while (vertFaces[i1*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) j++; vertFaces[i1*maxNeighbourCount+j] = face;
        j = 0; while (vertFaces[i2*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) j++; vertFaces[i2*maxNeighbourCount+j] = face;
    }

    int neighbourCount = 0;
    // link faces to neighbours which share same vertices:
    for (int face = 0; face < nFaces; face++) {
        int i0 = vertexIndex3[face*3+0];
        int i1 = vertexIndex3[face*3+1];
        int i2 = vertexIndex3[face*3+2];

        int j = 0; int idx = 0;
        int faceIndices[maxNeighbourCount]; for (j = 0; j < maxNeighbourCount; j++) faceIndices[j] = -1;

        // collect array of neighbouring faces
        j = 0; while (vertFaces[i0*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) { faceIndices[idx] = vertFaces[i0*maxNeighbourCount+j]; j++; idx++; }
        j = 0; while (vertFaces[i1*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) { faceIndices[idx] = vertFaces[i1*maxNeighbourCount+j]; j++; idx++; }
        j = 0; while (vertFaces[i2*maxNeighbourCount+j] != -1 && j < maxNeighbourCount) { faceIndices[idx] = vertFaces[i2*maxNeighbourCount+j]; j++; idx++; }

        int idx2 = 0;
        for (j = 0; j < idx; j++) {
            if (faceIndices[j] != face) {
                bool alreadyExists = false;
                for (int i = 0; i < idx2; i++) {
                    if (faceLinks[face*maxNeighbourCount+i] == faceIndices[j]) {
                        alreadyExists = true; break;
                    }
                }
                if (!alreadyExists) {
                    faceLinks[face*maxNeighbourCount+idx2] = faceIndices[j];
                    idx2++;
                }
            }
        }
        if (idx2 > neighbourCount) neighbourCount = idx2;
    }
    printf("max neighbor count: %d\n",neighbourCount);
    // associate faces to views with largest screen-space area
    for (size_t i = 0; i < keyFrames.size(); i++) {
        float mtxT[16],mtx[16];
        transpose4x4(&keyFrames[i]->T[0],&mtxT[0]);
        invertRT4(&mtxT[0],&mtx[0]);
        float *zData = (float*)keyFrames[i]->depthCPU.ptr();

        for (int face = 0; face < nFaces; face++) {
            float *uv = &uvMap[(nFaces*i+face)*6];
            bool visible = true;
            float avgDepth = 0.0f;
            float avgU = 0.0f;
            float avgV = 0.0f;
            // generate uv coordinates into keyframe view i:
            for (int vert = 0; vert < 3; vert++) {
                float *p = &vertices[3*vertexIndex3[face*3+vert]];
                float v[3],w[3];
                transformRT3(mtx,p,v);
                transformRT3(TRL,v,w); // generate point in IR view too for depth inspection
                if (w[2] > -300) visible = false;
                else {
                    avgDepth += w[2];
                    // normalize projective points
                    v[0] /= v[2]; v[1] /= v[2]; v[2] = 1;
                    w[0] /= w[2]; w[1] /= w[2]; w[2] = 1;
                    // little bit ad hoc test to see that lens distortion does not map points outside fov into image domain:
                    if (v[0]*v[0]+v[1]*v[1] < 1.0f) {
                        distortPointCPU(v,kc,K,&uv[vert*2]);
                        // use undistorted coordinates to sample depthCPU image (occlusion test)
                        avgU += KIR[0]*w[0]+KIR[2];
                        avgV += KIR[4]*w[1]+KIR[5];
                    } else visible = false;
                }
            }
            // guarantee uv coordinates are in the image domain:
            if (visible) {
                bool inBounds = true;
                // boundary test
                for (int vert = 0; vert < 3; vert++) {
                    float u = uv[vert*2+0];
                    float v = uv[vert*2+1];
                    if (u < 1.0f || u > (width-2) || v < 1.0f || v > (height-2)) {
                        inBounds = false;
                        break;
                    }
                }
                // occlusion test
                if (inBounds) {
                    avgU /= 3; avgV /= 3; avgDepth /= 3;
                    int ui = int(avgU);
                    int vi = int(avgV);
                    if (ui > 0 && ui < (width-1) && vi > 1 && vi < (height-1) ) {
                        // depths are positive in stored maps
                        float z  = zData[ui+vi*width];
                        float z2 = zData[(ui+1)+vi*width];
                        float z3 = zData[(ui+1)+(vi+1)*width];
                        float z4 = zData[ui+(vi+1)*width];
                        if (z2 < z) z = z2;
                        if (z3 < z) z = z3;
                        if (z4 < z) z = z4;
                        if (-avgDepth > z+100.0f || avgDepth > 0) {
                            inBounds = false;
                        }
                    }
                }
                // if these uv coordinates are better than anything before, select them for this polygon:
                if (inBounds) {
                    validUV[nFaces*i+face] = true;
                    float area = convexArea(uv,3);
//                    if (i < faceArea[face]) {
//                        faceArea[face] = i;
                    if (area > faceArea[face]) {
                        faceArea[face] = area;
                        bestKey[face] = i;
                    }
                }
            }
        }
    }

    // regularize uv mapping by selecting the most dominant label from face neighborhood in n iterations:
    reguralizeUVMapping(nFaces,&bestKey[0],faceLinks,maxNeighbourCount,&validUV[0],20,3);

    // use the best UV mapping for each face
    for (int face = 0; face < nFaces; face++) {
        int theBestView = bestKey[face];
        float *uv = &uvMap[(nFaces*theBestView+face)*6];
        memcpy(&faceUV[face*6],&uv[0],sizeof(float)*6);
    }

    // finally convert uv coordinates to larger tiled texture space
    int keyCounts[256]; memset(keyCounts,0,sizeof(int)*256);

    Mat uvMapping = rgbTexture.clone();

    for (int face = 0; face < nFaces; face++) {
        int key = bestKey[face];
        if (faceArea[face] > 0.0f) keyCounts[key]++;

        int baseUV = face*3*2;

        // compute tile coords
        int tileX = key%6; int tileY = key/6;
        float textureUBase = tileX*width;
        float textureVBase = tileY*height;

        Point pts[3];

        bool strangeZero = false;
        for (int vert = 0; vert < 3; vert++) {
            //convert uv into normalized coordinates in big texture
            float u1 = faceUV[baseUV+vert*2+0];
            float v1 = faceUV[baseUV+vert*2+1];
            float u2 =        (u1+textureUBase)/2048.0f;
            float v2 = 1.0f - (v1+textureVBase)/2048.0f;
            pts[vert] = Point((u1+textureUBase), (v1+textureVBase));
            fprintf(f,"vt %f %f\n",u2,v2);
            if (u1 < 1.0f || v1 < 1.0f) strangeZero = true;
        }

        if (key >= 0 && key < keyFrames.size() && !strangeZero) {
            const Point* polyPts[1] = { pts };
            int n[] = { 3 };
//            Mat *rgbImage = uvImages[key];
            for (int j = 0; j < 3; j++) {
                cv::line(uvMapping,pts[j],pts[(j+1)%3],cv::Scalar(0,0,255),1);
            }
//            cv::fillPoly(uvMapping,polyPts,n,1,cv::Scalar(0,0,255));
        }
    }
    for (size_t i = 0; i < keyFrames.size(); i++) printf("key %d faces : %d\n",int(i),keyCounts[i]);

    char buf[512];
    sprintf(buf,"%s/textureUV.png",scratchDir);
    savePNG(uvMapping.ptr(),rgbTexture.cols, rgbTexture.rows, 3, rgbTexture.cols*3, buf);

    delete[] faceArea;
    delete[] bestKey;
    delete[] faceUV;
    delete[] vertFaces;
    delete[] faceLinks;
    delete[] uvMap;
    delete[] validUV;
    fflush(stdin); fflush(stdout);

    return;
}


void KeyFrameModel::writeWaveFront(const char *filename, CoredFileMeshData *mesh, float *mtx4x4, TriangleBuffer2 **tribuf) {

    printf("saving 3d model into %s...\n",filename);

    char materialFile[] = "materials.mtl";

    // create tiled texture and store it:
    saveKeyImages(); // save keyframe textures
    writeWaveFrontMaterials(filename,materialFile);

    FILE *f = fopen(filename,"wb");
    fprintf(f,"mtllib %s\n",materialFile);
    fprintf(f,"g part0\n");
    fprintf(f,"usemtl material0\n");

    mesh->resetIterator();

    int totalVerts = int(mesh->inCorePoints.size())+mesh->outOfCorePointCount();
    float *vertices = new float[totalVerts*3];

    Point3D< float > p; int vi = 0;
    // incore points
    for( size_t i=0 ; i < mesh->inCorePoints.size(); i++ ) {
        p = mesh->inCorePoints[i];
        vertices[vi*3+0] = p[0];
        vertices[vi*3+1] = p[1];
        vertices[vi*3+2] = p[2];
        vi++;
    }
    // out-of-core points
    for( int i=0 ; i < mesh->outOfCorePointCount(); i++ ) {
        mesh->nextOutOfCorePoint(p);
        vertices[vi*3+0] = p[0];
        vertices[vi*3+1] = p[1];
        vertices[vi*3+2] = p[2];
        vi++;
    }

    double midPoint[] = {0.0f,0.0f,0.0f};
    for (int i = 0; i < totalVerts; i++) {
        midPoint[0] += vertices[i*3+0];
        midPoint[1] += vertices[i*3+1];
        midPoint[2] += vertices[i*3+2];
    }
    midPoint[0] /= totalVerts; midPoint[1] /= totalVerts; midPoint[2] /= totalVerts;

    printf("midpoint: %f %f %f\n",midPoint[0],midPoint[1],midPoint[2]);
    for (int i = 0; i < totalVerts; i++) {
        fprintf(f,"v %f %f %f\n",(vertices[i*3+0]-midPoint[0])*1e-3f,(vertices[i*3+1]-midPoint[1])*1e-3f,(vertices[i*3+2]-midPoint[2])*1e-3f);
    }

    int nFaces=mesh->polygonCount();

    const int maxVertsPerPolygon = 16;
    unsigned int *faceIndex = new unsigned int[nFaces*maxVertsPerPolygon]; // max 15 vertices per face
    std::vector< CoredVertexIndex > polygon;
    int fi = 0;
    // store all polygon data with valid index, also make sure polygons have < 16 vertex indices! >= 16 is never occurring in marching cubes
    for(int  i=0 ; i < nFaces ; i++ ) {
        mesh->nextPolygon( polygon );
        int verts = int( polygon.size() );
        // valid range: 3-15 vertices per face
        if (verts >= maxVertsPerPolygon || verts < 3) continue;
        faceIndex[fi*maxVertsPerPolygon+0] = verts;
        for( int j = 0; j < verts; j++ ) {
            if ( polygon[j].inCore) faceIndex[fi*maxVertsPerPolygon+j+1] = polygon[j].idx;
            else                    faceIndex[fi*maxVertsPerPolygon+j+1] = polygon[j].idx + int( mesh->inCorePoints.size());
        }
        fi++;
    }    
    nFaces = fi;

    printf("totalverts: %d, polygon count:%d \n",totalVerts,nFaces); fflush(stdin); fflush(stdout);

    // how many triangles after tesselation?
    int nFaces3 = nFaces*(maxVertsPerPolygon-2);
    unsigned int *faceIndex3 = new unsigned int[nFaces3*3]; // max 3 vertices per face
    fi = 0;
    for (int i = 0; i < nFaces; i++ ) {
        int verts = faceIndex[i*maxVertsPerPolygon+0];
        // store the first triangle as is:
        faceIndex3[fi*3+0] = faceIndex[i*maxVertsPerPolygon+1];
        faceIndex3[fi*3+1] = faceIndex[i*maxVertsPerPolygon+2];
        faceIndex3[fi*3+2] = faceIndex[i*maxVertsPerPolygon+3];
        fi++;
        // tesselate the rest:
        for (int j = 3; j < verts; j++,fi++) {
            // store the first triangle as is:
            faceIndex3[fi*3+0] = faceIndex3[(fi-1)*3+0];
            faceIndex3[fi*3+1] = faceIndex3[(fi-1)*3+2];
            faceIndex3[fi*3+2] = faceIndex[i*maxVertsPerPolygon+1+j];
        }
    }
    nFaces3 = fi;
    delete[] faceIndex;

    printf("tesselated triangle count:%d\n",nFaces3);

    // setup calibration data for 320x240 resolution
    m_calib.setupCalibDataBuffer(keyFrames[0]->distortedRgbImageCPU.cols,keyFrames[0]->distortedRgbImageCPU.rows);

    generateUV3(vertices,&faceIndex3[0],nFaces3,f);
    int uvIndex = 1;
    for (int i = 0; i < nFaces3; i++) {
        int verts = 3;
        fprintf(f,"f");
        for( int j = 0; j < verts; j++,uvIndex++) {
            fprintf(f," %d/%d",faceIndex3[i*3+j]+1,uvIndex);
        }
        fprintf(f,"\n");
    }
    fclose(f);

    TriangleBuffer2 *tbuf = *tribuf;
    if (tbuf != NULL) {
        tbuf->release();
        delete tbuf;
    }

    tbuf = new TriangleBuffer2(vertices,nFaces3,faceIndex3,"triangle buffer");
    *tribuf = tbuf;

    printf("nindices: %d\n",uvIndex); fflush(stdin); fflush(stdout);
    delete[] faceIndex3;
    delete[] vertices;
}
