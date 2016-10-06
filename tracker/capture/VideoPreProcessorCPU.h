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
#include "videoSource.h"
#include "../calib/calib.h"
#include "../multicore/multicore.h"

using namespace cv;

class VertexBuffer2;

class VideoPreProcessorCPU
{
public:
    VideoPreProcessorCPU(VideoSource *source, const int nLayers, Calibration &localCalib, int targetResoX, int targetResoY);
    ~VideoPreProcessorCPU();
    int preprocess(bool normalizeDepth=false, bool calcPlanarityIndex=false, bool calcTexturing = false);
    Mat &getRGBImage();
    Mat &getDistortedRGBImage();
    Mat &getGrayImage();//int index);
    int getWidth();
    int getHeight();
    int getDepthWidth();
    int getDepthHeight();
    bool isPlaying();
    void setVideoSource(VideoSource *kinect, int width, int height);
    VideoSource *getVideoSource();
    void setBrightnessNormalization(bool flag);
    void updateCalibration();
    void setPixelSelectionAmount(int pixelAmount);
    float *getSelected2dPoints() { return selectedPoints2d; }
    float *getSelected3dPoints() { return selectedPoints3d; }
    void getPlane(float *mean,float *normal);
    Mat &getDepthImageR();
    Mat &getDepthImageL();
    cv::Mat depthMapR;
    void release();
    int getFrame();
    void setFrame(int frame);
    void pause();
    void reset();
    bool isPaused();
    float getPlanarityIndex() { return m_planarityIndex; };
    float getTexturingIndex() { return m_texturingIndex; };
    ProjectData *getPointCloud();
private:
    int frameIndex;
    int frameInc;
    bool loopFlag;
    float calcPlanarityIndex();  // note: does this actually work?
    float calcTexturingIndex();
    VideoSource *device;
    int width, dwidth;
    int height, dheight;
    bool brightnessNormalizationFlag;
    const int nMultiResolutionLayers;
    void cpuPreProcess(unsigned char *rgb, float *depth,bool normalizeDepth, bool calcPlanarityFlag, bool calcTexturing);
    void downSample2( Mat  &img, Mat &img2 );
    void normalizeBrightness( Mat &src, Mat &dst );
    void planeRegressionPhotometric(ProjectData *fullPointSet, int count, float *mean, float *normal);
    float planeRegressionGeometric(ProjectData *fullPointSet, int count, float *mean, float *normal,float *eigenU,float *eigenV);
    void fastImageMedian(Mat &src, int *medianVal);
    Calibration localCalib;
    cv::Mat depthMapL;
    float m_planarityIndex;
    float m_texturingIndex;
    ProjectData *fullPointSet;
    int pixelSelectionAmount;
    float *selectedPoints3d;
    float *selectedPoints2d;
    float planeMean[3];
    float planeNormal[3];
    bool planeComputed;
};
