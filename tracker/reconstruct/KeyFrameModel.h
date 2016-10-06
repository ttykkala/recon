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
#ifndef KEYFRAMEMODEL_H
#define KEYFRAMEMODEL_H

#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <rendering/VertexBuffer2.h>
#include <rendering/LineBuffer.h>
#include <calib/calib.h>
#include <opencv2/opencv.hpp>
#include <reconstruct/KeyFrame.h>
#include <list>
#include <map>
#include <vector>

using namespace cv;
using namespace std;

class ProjectionData{
public:
    ProjectionData() { px = 0; py = 0; z = 0; }
    ProjectionData(float x, float y, float zztop) { px = x; py = y; z = zztop; }
    ProjectionData &operator = (const ProjectionData &projData) {
        px = projData.px;
        py = projData.py;
        z = projData.z;
    }
    float px,py;
    float z;
};

class TriangleBuffer2;

// grid object for image based pose comparison
#define nGridX  30
#define nGridY  30
#define nGridZ  5
#define nGridSize (nGridX*nGridY*nGridZ)

#define VERTEX_FUSION_LAYER_COUNT 4

class CoredFileMeshData;

class KeyFrameModel
{
public:
        // flags:
        // optimizeGPUMemory : full dense point clouds vs sparse point clouds
        // undistortRGB      : undistort RGB images only for visualization purposes (algorithm uses distorted images directly)
        // renderableVBuffer : flag to tell whether this buffer will be rendered, disabling rendering avoids slow cuda interop completely
        // rgbVisualization  : flag to tell whether compressed gray1,gray2,gray3 values should be replaced by r,g,b (only for visualization!)
        KeyFrameModel(const char *keyFramePath, const int nLayers, int pixelSelectionAmount, int interpolatedKeyPointAmount, bool optimizeGPUMemory = true, bool undistortRGB=false, bool renderableVBuffers = true, bool rgbVisualization=false);
        // this special initialization is used when empty keyframe model is generated for incremental reconstruction:
        KeyFrameModel(int slamDistTol, int slamAngleTol, int nMaxKeys, Calibration *extCalib, const int nLayers, int pixelSelectionAmount, int interpolatedKeyPointAmount, bool undistortRGB=false, bool renderableVBuffers = true, bool rgbVisualization=false);
        ~KeyFrameModel();
        int getWidth();
        int getHeight();
        int getDepthWidth();
        int getDepthHeight();
        KeyFrame *getKeyFrame(int index);
        KeyFrame *getKeyFrameID(int index);
        int getKeyFrameIndex(KeyFrame *keyFrame);
        int getKeyFrameCount();
        void disableKeyFrame(int id);
        void renderTrajectory(int frame = -1);
        void renderCameraFrames(std::vector<KeyFrame *> *activeKeys = NULL, float len = 350.0f, float r=1.0f, float g=1.0f, float b=1.0f);
        void findSimilarKeyFrames(float *T, std::vector<KeyFrame*> &similarKeys, float *testObj=NULL, int nVertices=0);
        void findSimilarKeyFramesLarge(float *T, std::vector<KeyFrame*> &similarKeys, float *testObj=NULL, int nVertices=0);
        void setIterationCounts(int *nIterations);

//        KeyFrame *findSimilarKeyFrame2(KeyFrame *kRef);
        //void interpolateKeyAlongTrajectory(float t);
        //void interpolateKeyAlongTrajectory4(float t);
        //void interpolateKeyFromDatabase(float *T, float maxDist, float maxAngle);
        //void interpolateKey(std::vector<KeyFrame*> &nearKeys, std::vector<float> &weights);
        void renderInterpolatedPose(float r, float g, float b, float len);
        std::vector<KeyFrame*> *getInterpolationNeighbors();
        void release();
        void resetTransforms();
        void pushTransforms();
        void popTransforms();
        void removeKeyframes();
        int getFrameCount();
        char *getBundleFileName();
        char *getSmoothFileName();
        char *getScratchDirName();
        void saveKeyImages();
        void generateTexture();
        Calibration *getCalib() { return &m_calib; }
        KeyFrame *getInterpolatedKey();
        float getInterpolationTime();
        void reconstruct3dPoint(int srcKeyFrame, int dstKeyFrame, float px, float py, float *p3);
        KeyFrame *extendMap(int id, ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbuffer, float *imDepthDevIR, int pixelSelectionAmount, float *T, KeyFrame *previousKey);
        void extractFeatures();     
        void savePolygonModel(char *filename, int numThreads, int depth, bool clipUsingBox=false, float *boxParams=NULL,VertexBuffer2 **vbuf = NULL, TriangleBuffer2 **tribuf = NULL);
        void renderPose(KeyFrame *kf, float r, float g, float b, float len);
private:
        int width;
        int height;
        int nIterations[3];
        bool m_renderableFlag;
        bool m_rgbVisualization;
        const int nMultiResolutionLayers;
        void uploadCalibData();
        void loadKeyFrames(const char *path, Calibration *calib, int pixelSelectionAmount, int interpolatedKeyPointAmount, bool undistortRGB=false);
        void enumerateNearPoses(float *T, float maxDist, float maxAngle, std::vector<KeyFrame*> &closePose, KeyFrame *rejectedKey = NULL);
        void setDimensions();
        KeyFrame *findSimilarKeyFrame(KeyFrame *key, float minDist, float maxDist, int M, int N);
        KeyFrame *findSimilarKeyFrame(float *m, float maxDist, float maxAngle, float *testObj=NULL, int nVertices=0);
        void findSimilarKeyFrames(float *T, float maxDist, float maxAngle, std::vector<KeyFrame*> &similarKeys, float *testObj=NULL, int nVertices=0);

        std::vector<KeyFrame*> keyFrames;
        KeyFrame *interpolatedKey;
        float *parseBundleData(const char *bundleFile, std::map<int, std::list<ProjectionData> > &projMap);
        void fastImageMedian(Mat &src, int *medianVal);
        void writeWaveFront(const char *filename, CoredFileMeshData *mesh, float *mtx4x4, TriangleBuffer2 **tribuf);
        void writeWaveFrontMaterials(const char *filename, char *materialFile);
        void generateUV3(float *vertices, unsigned int *faceIndex, int nFaces, FILE *f);
        //void initializeUVImages(std::map<int,Mat*> &uvImages);
        //void releaseUVImages(std::map<int,Mat*> &uvImages);
//        void generateUV(float *vertices, unsigned int *faceIndex, int numFaces, FILE *f);
        // two methods for fusing point clouds
        void sparseCollection(std::vector<KeyFrame*> &nearKeys,std::vector<float> &weights, bool evenSelection, KeyFrame *interpolatedKey);
        //void denseBlend(std::vector<KeyFrame*> &nearKeys,std::vector<float> &weights, KeyFrame *interpolatedKey);
        void sortKeys(std::vector<KeyFrame*> &nearKeys, std::vector<float> &weights);
        void cpuPointClouds(Calibration *calib);
        unsigned int reconstructGlobalPointCloud(float **points, float **normals);
  //      float poseDistance(float *T1, float *T2, float lambda);
 //       void poseDistance(float *relativeT, float *dist, float *angle);

        void setupNeighborKeys(float maxDist, float maxAngle);
        float projectionDistance(float *relT, float *testObj, int nVertices);
        bool keyFrameOccupied(float *T, float maxDist, float maxAngle, KeyFrame **keyframe);
        void writeTrajectoryDifferenceTxt(float *ref, float *cur,int numFrames, const char *outFile);
        LineBuffer *trajectory;
        Calibration m_calib;
        float *calibDataDev;
        Image2 depth1C;
        std::vector<KeyFrame*> interpolationNeighbors;
        float interpolationTimeMillis;
        // temporary vertex buffer for full point clouds generated by full zmaps
        VertexBuffer2 *vbufferTemp;
        void integrateDepthMaps(const char *datasetPath,Calibration *calib, float *cameraTrajectory);
        void adjustPoses(Calibration *calib, int pixelSelectionAmount);
        void loadImage(const char *datasetPath, int index, cv::Mat &rgbImageSmall);
        void loadDepthMap(const char *datasetPath, int frameIndex, cv::Mat &depthMap, Calibration *calib);
        bool readZMap(const char *fn, Mat &depthImage);
        KeyFrame *createKeyFrame(int frameIndex, cv::Mat &rgbImageSmall,cv::Mat &depthMap,Calibration *calib,float *T);
        // for on the fly GPU map extension:
        KeyFrame *createDummyCPUKeyframe(int frameIndex, Calibration *calib, float *T, KeyFrame *previousKey);
        void addKeyFrame(int frameIndex,cv::Mat &rgbImageSmall,cv::Mat &depthMap,Calibration *calib,float *T);
        void generateGPUKeyframes(int pixelSelectionAmount, bool renderableVBuffers, bool undistortRGB, Calibration *calib);
        void setupCameraTrajectory(float *cameraTrajectory, int firstFrame, int lastFrame);
        bool isKeyFrame(int frameIndex);
        void generateGridObject(float *gridObj,int nStepsX, int nStepsY, int nLayers, float size, float viewDistanceMin, float zRange, float fovAngleX);
        void writeZMap(const char *fn, Mat &depthImage);
        void bubbleSort(std::vector<float> &vals, std::vector<int> &indices);
        char poseFile[512];
        char posePath[512];
        char datasetPath[512];
        char bundleFile[512];
        char smoothFile[512];
        char scratchDir[512];
        int numFrames, maxNumKeyFrames, firstFrame, lastFrame;
        int distTol,angleTol;
        float gridObject[nGridSize*3];
        float projectedGridObject[nGridSize*2];
        float *vertexFusionLayersDev;
        int *vertexFusionIndicesDev;
        int m_interpolatedKeyPointCount;
        bool m_optimizeGPUMemory;
        Mat rgbTexture;
        Mat rgbTextureLarge;
        float *savedTransforms;
};


#endif // KEYFRAMEMODEL_H
