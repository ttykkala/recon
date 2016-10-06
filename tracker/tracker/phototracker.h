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

#include <barrier_sync.h>
#include <opencv2/opencv.hpp>
#include <image2/ImagePyramid2.h>
#include <rendering/VertexBuffer2.h>
#include <tracker/shapetracker.h>
#include <cuda_runtime.h>

struct cudaArray;
struct float4;
class KeyFrame;
class Calibration;

enum TrackMode { GEOMETRIC, PHOTOMETRIC, BIOBJECTIVE };

class PhotoTracker {
public:
    PhotoTracker(int cudaDeviceID, barrierSync *barrier, int resoX, int resoY, Calibration *calib);
    ~PhotoTracker();
    int run();
    void release();
    int getDeviceID();
    void setThreadActive(bool flag);
    bool isActive();
    void sync();
    double *getGlobalPose();
    float *getRelativePose();
    void resetTracking(TrackMode mode=GEOMETRIC);
    void setCurrentImage(cudaArray *rgbdFrame);
    bool setReferencePoints(float4 *newRefPoints,float4 *newRefPointNormals, cudaArray *rgbdFrame, TrackMode suggestedMethod, float lambda=0.0f);
    float4 *getDebugImage();
    float *getDebugImage1C(int layer);
    float *getRefImage1C(int layer);
    void allocMemory();
    void freeMemory();
    void prepareCurrent();
    void prepareReference();
    bool isReferencePrepared();
    void setIterationCounts(int *nIter);
    void lock();
    void unlock();
    bool readyToTrack();
    void listSelected(int layer, float **selected, float **refSelected, float **color, int *nSelect);
    void signalCalibrationUpdate(float *calibDataExt);
    int numReferenceUpdates();
    bool poseDifferenceExceeded(float *T, float translationThreshold, float rotationThreshold);
    ShapeTracker *getShapeTracker();
    int getLayers();
    void applyIncrement();
    void addShapeTrackerIncrement(float *Tinc);
    double *getIncrement();
    void setMode(TrackMode mode);
    TrackMode getMode();
    void setReferenceUpdating(bool flag);
    void setFilteredReference(bool flag);
    bool getFilteredReference();
    void clearPose();
    void resetToTransform(double *Tinc);
    void gpuPreProcess();
    void gpuEstimateIncrement();
    void getIterationCounts(int *niter);
    void setLambda(float lambda);
    float getLambda();
    void updateErrorImage();
    void setIncrementalMode(bool flag);
private:
    bool m_incrementalMode;
    cudaStream_t stream;
    TrackMode mode;
    bool referenceUpdatingFlag;
    bool referenceOutdated();
    KeyFrame *createKeyFrame(int frameIndex, cv::Mat &rgbImageSmall, cv::Mat &depthMap, Calibration *calib,float *T);
    void selectPixels(int pixelSelectionAmount);
    bool updateLinearSystem(int layer);
    void estimateIncrement(int layer);
    void updatePhotometricReference(float4 *newRefPointsDev, cudaArray *newRefFrame);
    int deviceID;
    bool threadActiveFlag;
    barrierSync *barrier;
    double globalPose[16]; // pose at frame n-1
//    cudaArray *rgbdFrame;
    bool estimateOnCPU;
    cudaArray *refFrame;
    float4 *debugImage;
    float *debugImage1C;
    float *debugRefImage1C;
    int texWidth,texHeight;
    bool referenceExists;
    bool currentExists;
    bool referencePrepared;
    ShapeTracker *shapeTracker;
    int refFrameNumber;
    float *calibData;
    float *T;
    double *Tinc;
//    float TbasePhoto[16];
    float *calibDataDev;
    int nLayers;
    int nIterations[4];
    int nIterationsBiObjective[4];
    float *TidentityDev;
    float *TrelDev;
    bool calibrationUpdateSignaled;
    float poseDistanceThreshold;
    float poseAngleThreshold;
    float lambda;
    int currentFrame;
    char eigenFileName[512];

    float4 *refPointsDev;
    ImagePyramid2 grayPyramidRef;
    ImagePyramid2 grayPyramidCur;
    VertexBuffer2 vbuffer;
    float *histogramFloatDev;
    float *gradientScratchDev;
    float *d_hist;
    float *jacobianTDev[4];
    float *residualDev;
    float *weightedResidualDev;
    float *weightsDev;
    float *zWeightsDev;
    float *weightedJacobianTDev;
    float *JtJDev;
    float *zReferenceDev;
    float *zCurrentDev;
    float *residual6Dev;
    float *selectionPointsDev;
    float *selectionColorsDev;
    float *refSelectionPointsDev;
    float *refSelectionColorsDev;
    float *selectionPoints;
    float *selectionColors;
    float *refSelectionPoints;
    float *refSelectionColors;
    int nSelected;
    bool saveCovariancesToDisk;
};

PhotoTracker *initializePhotoTracker(int deviceID, int texWidth, int texHeight, Calibration *calib);
//void releasePhotoTracker();
bool trackingThreadAlive();
