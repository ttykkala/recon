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

#include <opencv2/opencv.hpp>
#include <vector>
#include <calib/calib.h>
#include <tracker/basic_math.h>
#include <Eigen/Geometry>

class OMPFunctions
{
public:
    OMPFunctions();
    ~OMPFunctions();
    void convert2Gray(cv::Mat &rgbImage, cv::Mat &grayImage);
    void undistort(cv::Mat &distorted, cv::Mat &undistorted, float *K, float *iK, float *kc);
    void undistortLookup(cv::Mat &distorted, cv::Mat &undistorted);
    void undistortF(cv::Mat &distorted, cv::Mat &undistorted, float *K, float *iK, float *kc);
    void undistortLookupF(cv::Mat &distorted, cv::Mat &undistorted);
    void downSampleDepth(cv::Mat &depthImage, cv::Mat &depthImageSmall);
    void d2Z(cv::Mat &dispImage, cv::Mat &depthImage, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);
    void d2ZHdr(cv::Mat &dispImage, cv::Mat &depthImage, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);
    void d2ZLow(cv::Mat &dispImage, cv::Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);
    void d2ZLowHdr(cv::Mat &dispImage, cv::Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);
    void d2ZLowGPU(cv::Mat &dispImage, cv::Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff); // gpu version has also normalization
    void solveZMap(cv::Mat &dispImage, cv::Mat &depthImage, float c0, float c1, float minDist, float maxDist);
    void z2Pts(cv::Mat &depthMap, float *K, float *pts3);
    void baselineTransform(cv::Mat &depthImageL,cv::Mat &depthImageR,float *KL, float *TLR, float *KR);
    void convert2Float(cv::Mat &dispImage, cv::Mat &hdrImage);
    void replaceUShortRange(cv::Mat &dispImage, unsigned short valueStart, unsigned short valueEnd, unsigned short newValue);
    void downsampleDisparityMap(cv::Mat &dispImage, cv::Mat &dispImageLow);
    void baselineWarp(cv::Mat &depthImageL,cv::Mat &grayImageR,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet);
    void baselineWarpRGB(cv::Mat &depthImageL,cv::Mat &rgbImageR,Eigen::Matrix3f &KL, Eigen::Matrix4f &TLR, Eigen::Matrix3f &KR, float *kc, ProjectData *fullPointSet);
    //void baselineWarpRGB(cv::Mat &depthImageL,cv::Mat &rgbImageR,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet);
    void generateDepthMap(ProjectData *fullPointSet, cv::Mat &depthImageR);
    void undistortDisparityMap(cv::Mat &dispImage, cv::Mat &uDispImage, float alpha0, float alpha1, float *beta);
    void optimizePhotometrically(float *zmap, unsigned char *rgbMask, unsigned char *rgbReference, int w, int h, float *stdevImage, int nsamples, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage);
    void generateZArray(float *zmap,unsigned char *rgbMask,int width, int height,float *stdevImage,int nsamples,float *zArray);
    void generateCostVolume(float *zArray, int width, int height, int nsamples, unsigned char *mask, unsigned char *rgbReference, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage, float *costArray);
    void normalizeCosts(float *costArray, float *countArray, unsigned char *mask, int width, int height, int nsamples);
    void argMinCost(float *zArray, float *costArray, int width, int height, int nsamples, unsigned char *mask, float *zmap);
    double residualICP(cv::Mat &xyzRef, cv::Mat &maskRef, float *K, float *T, cv::Mat &xyzCur, float *residual, float *jacobian, float scaleIn, float depthThreshold, int stride);
    double residualPhotometric(cv::Mat &xyz,cv::Mat &selection, int nPoints, float *kc, float *KR, float *TLR, float *T, cv::Mat &grayRef, float* residual, float *jacobian, float *wjacobian, int layer, float intensityThreshold, int stride);
    void refineDepthMap(cv::Mat &xyzCur,cv::Mat &weightsCur, float *K,  float *T, cv::Mat &xyzRef,cv::Mat &weightsRef, int stride, float depthThreshold=50.0f, float rayThreshold = 20.0f);
    void downSamplePointCloud(cv::Mat &hiresXYZ, cv::Mat &lowresXYZ, int stride);
    void generateNormals(cv::Mat &xyzImage, int stride);
    void downSampleMask(cv::Mat &hiresMask, cv::Mat &lowresMask);
    void downSampleHdrImage(cv::Mat &hiresImage, cv::Mat &lowresImage);
    void Jtresidual(float *jacobian, float *residual, int cnt, int rows, double *b);
    void AtA6(float *jacobian, int cnt, double *A);
    void generateOrientedPoints(cv::Mat &depthCPU, cv::Mat &xyzImage, float *KL, cv::Mat &normalStatus, float *kc, float *KR, float *TLR, cv::Mat &grayImage, int stride);
    void precomputePhotoJacobians(cv::Mat &xyzCur,float *kc, float *K, float *TLR, cv::Mat &gray, int nPoints, cv::Mat &photometricSelection, int stride, cv::Mat &photoJacobian, int layer, float scaleIn);
private:
    void init();
    int NCPU,NTHR;
    float *dxTable;
    bool mappingPrecomputed;
};

extern OMPFunctions *getMultiCoreDevice();
