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
#include <cuda_runtime.h>

struct cudaArray;
struct float4;
class KeyFrame;
class Calibration;

class ShapeTracker {
public:
    ShapeTracker(int cudaDeviceID, int resoX, int resoY, Calibration *calib, float *TrelExtHost, float *TrelDevExt, float optScaleIn, float optScaleOut, cudaStream_t stream=0);
    ~ShapeTracker();
    void release();
    int getDeviceID();
   // float *getEstimatedPose();
    void resetTracking();
    void setCurrentImage(float *zCurrentDev);
    void updateErrorImage();
    bool referenceOutdated();
    void setReferencePoints(float4 *newRefPointsDev, float4 *newRefPointNormalsDev, cudaArray *newRefFrame);
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
    void listSelected(int layer, float **selected, float **refSelected,float **color, int *nSelect);
    void signalCalibrationUpdate(float *calibDataExt);
    int numReferenceUpdates();
    bool poseDifferenceExceeded(float *T, float translationThreshold, float rotationThreshold);
    void updateCalibration(float *calibData);
  //  void setPhotoRefToCurrent(float *iT1Dev);
   // void addIncrement(float *Tinc);
  //  float *getIncrement();
    void setReferenceUpdating(bool flag);
    void setFilteredReference(bool flag);
    bool getFilteredReference();
   // void clearPose();
//    void resetToTransform(float *Tinc);
    bool updateLinearSystem(int layer, int iteration);
    void getIterationCounts(int *niter);
    void estimateIncrement();
    void linearFuse(float *JtJDev, float *residual6Dev, int nPixels, int layer, float lambda);
    void copyLinearSystemTo(float *JtJDevExt, float *residual6DevExt);
    int getRecentSelected(int layer);
    void setIncrementalMode(bool flag);
private:
    bool m_incrementalMode;
    bool referenceUpdatingFlag;
    KeyFrame *createKeyFrame(int frameIndex, cv::Mat &rgbImageSmall, cv::Mat &depthMap, Calibration *calib,float *T);
    void generatePyramids(float **xyzMap, int **selectionMasks, int nLayers);
    void updateRefImages(float **xyzMapDev, int nLayers);
//    void updateDebugDiffImages(float **xyzMapDev1,float *T1, float *T2, float **xyzMapDev2, int nLayers);
    void updateDebugDiffImages(float **xyzMapDev1,int **select1Dev, float *Tdev, float *calibDataDev, float **xyzMapDev2, int **select2Dev, int nLayers);
    void clearDebugImages();
    int deviceID;
    float estimatedPose[16]; // pose at frame n-1
//    cudaArray *rgbdFrame;
  //  cudaArray *refFrame;
    float optScaleIn,optScaleOut;
    float4 *debugImage;
    float *debugImage1C[4];
    float *refImage1C[4];
    int texWidth,texHeight;
    bool referenceExists;
    bool currentExists;
    bool referencePrepared;
    int geometryResidualSize;
    cudaStream_t stream;

    float *calibData;
    float *Text;
//    float T[16],Tinc[16];
  //  float Tbase[16];
    float *calibDataDev;
    int nLayers;
    int nIterations[4];
    float *TidentityDev;
    float *TDev,*iTDev;
    bool calibrationUpdateSignaled;
    bool useRawMeasurementsForPoseEstimation;
    float poseDistanceThreshold;
    float poseAngleThreshold;

    float  *refPointsDev[4];
    float  *curPointsDev[4];
    int    *selectionMaskCurDev[4];
    int    *selectionMaskRefDev[4];
    int    *selectionIndexDev[4];
    float  *weightsDev[4];
    int    nSelected[4];
    float  *jacobianDev;
    unsigned char *normalStatusDev;
    float *fullResidualDev;
    int   *residualMaskDev;
    float *residualDev;
    float *residual2Dev;
    float *JtJDev;
    float *zReferenceDev;
    float *zCurrentDevExt;
    float *residual6Dev;
    float *selectionPointsDev;
    float *refSelectionPointsDev;
    float *selectionColorsDev;
    float *selectionPoints;
    float *refSelectionPoints;
    float *selectionColors;
    float *partialSumDev;
    int   *partialSumIntDev;
    int    referenceUpdateCount;
};
