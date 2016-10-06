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

#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <rendering/VertexBuffer2.h>
#include <rendering/BaseBuffer2.h>
#include <opencv2/opencv.hpp>
#include <list>
#include <vector>
using namespace cv;
using namespace std;

const int maxResidualSize = 320*240;
class Calibration;

typedef struct {
    float x0[3],x1[3],x2[3],x3[3]; // front quad vertices
    float y0[3],y1[3],y2[3],y3[3]; // back  quad vertices
} FRUSTUM;

typedef struct {
    float o[3];
    float dir[3];
    float tmin,tmax;
} RAY;

class TrackFrame {
private:
    void selectPixels(Mat &img, Mat &depth, Mat &baseMask, Mat &pixelSelectionMask, float percent);
    void allocateJacobian(int pixelSelectionAmount);
    void precomputeJacobian();
    void precomputeJacobian(VertexBuffer2 *vbufferExt);

    float *TrelDev;
    float *TabsDev;
    float *TnextDev;
    float *TidentityDev;
    float *TtmpDev;
    float *calibDataDevEXT;
    float *jacobianTDev[3];
    VertexBuffer2 *baseBufferEXT;
    float fovAngleX;
    float fovAngleY;
    FRUSTUM frustum;
    int nIterations[3];
public:
    TrackFrame(int width, int height);
    ~TrackFrame() {};
    void setIterationCounts(int *nIterations);
   // void initializeFeatures(int nFeatures, int keyIndex, Calibration *calib);
    //void searchFeatures(KeyFrame *curFrame, int keyIndex, Calibration *calib, int searchRegion, float depthTolerance);
    void release();
    void normalizeDepthMap(cv::Mat &depthMap, cv::Mat &normalizedMap, float depthMin, float depthMax);
    void generateGPUData(int pixelSelectionAmount, bool renderableVBuffer,bool optimizeGPUMemory, bool undistortRGB, bool rgbVisualization,Calibration *calib, unsigned char *imRGBDev, float *imDepthDev, Image2 &depth1C, VertexBuffer2 *vbufferTemp, int nMultiResolutionLayers);
    void resetRelativeDev();
    void getScenePoint(int i, int j, Calibration &calib, float *x, float *y, float *z);
    void selectPixels(float pixelSelectionPercent);
    void selectPixels(Mat &img, Mat &depth, Mat &baseMask, Mat &pixelSelectionMask, int amount);    
    void selectPixelsGPUCompressed(VertexBuffer2 &vbufferExt, int pixelSelectionAmount,Image2 *imDepth, bool rgbVisualization);
    //void selectPixelsGPUCompressed2(float *verticesExt, int *indicesExt, int vertexCount, int pixelSelectionAmount, int stride);
    void selectPixelsGPUCompressedLock(VertexBuffer2 &vbufferExt, int pixelSelectionAmount,Image2 *imDepth, ImagePyramid2 *grayPyramidExt);
    void reallocJacobian(int pixelSelectionAmount);
    // experimental feature extraction for 3d model refinement
    void extractFeatures();
    void gpuCopyKeyframe(int id, ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbuffer, float *imDepthDevIR, int pixelSelectionAmount, VertexBuffer2 *vbufferTemp, int nMultiResolutionLayers=3, bool renderableVBuffer=false);
    void generatePointsWithoutNormals(Calibration *calib);
    void generatePoints(Calibration *calib);
    void optimizePose(ImagePyramid2 &grayPyramid, VertexBuffer2 *vbufferCur, bool filterDepthFlag=false);
    void setCalibDevPtr(float *calibDataDev, float normalizedK11, float normalizedK22);
    void setBaseBuffer(VertexBuffer2 *vbuf) { baseBufferEXT = vbuf; }
    void resetTransform();
    void updateBase(BaseBuffer2 &base, ImagePyramid2 &grayPyramid);
    // inputT in standard form, is transposed into OpenGL convention
    void updateTransform(float *T);
    // inputT in OpenGL compatible form already
    void updateTransformGL(float *T);
    void setupCPUTransform();
    void setupRelativeCPUTransform(float *T);
    void setupRelativeGPUTransform(float *TDev);
    void getRelativeTransform(float *T, float *relativeT);
    float *getAbsBaseDev() { return TabsDev; }
    float *getRelBaseDev() { return TrelDev; }
    float *getNextBaseDev() { return TnextDev; }
    void setBaseTransform(float *absBaseDev);
    float getFovX() { return fovAngleX; }
    float getFovY() { return fovAngleY; }
    void updateFrustum(float viewDistanceMin, float viewDistanceMax);
    FRUSTUM *getFrustum() { return &frustum; };
    void getRay(int i, int j, int nX, int nY, RAY *ray);
    void lock();
    void unlock();
    std::vector<int> neighborIndices;
    std::vector<TrackFrame*> neighborKeys;
    Image2 rgbImage;
    ImagePyramid2 grayPyramid;
    Mat distortedRgbImageCPU;
    Mat grayImageHDR;
    Mat grayImage;
    Mat baseMask;
    Mat pixelSelectionMask;
    Mat depthCPU;
    Mat depthRGB;    
    Mat xyzImage,normalImage; // CPU point cloud
    int medianVal;
    VertexBuffer2 vbuffer;
    float T[16];
    float invT[16];
    float Tbase[16];
    bool visible;
    int id;
    vector<KeyPoint> keyPoints;
    cv::Mat imageDescriptors;
    float colorR,colorG,colorB;
    // keyframe weight is temporary variable containing the estimation weight which is based on the similarity metric
    float weight;
};


void getSearchBound(TrackFrame *key0, TrackFrame *key1, float x, float y, float searchBound, Calibration &calib, float *p1, float *p2);
void getSearchBound(TrackFrame *key0, TrackFrame *key1, float *p3d, float searchBound, Calibration &calib, float *p1, float *p2);
void getProjection(TrackFrame *key0, TrackFrame *key1, float *p3d, Calibration &calib, float *p );
