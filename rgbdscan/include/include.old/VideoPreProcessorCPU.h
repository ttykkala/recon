/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/
#pragma once

#include <opencv2/opencv.hpp>
#include "videoSource.h"
#include <calib.h>
#include "basic_math.h"

using namespace cv;

class VertexBuffer2;

class VideoPreProcessorCPU
{
public:
    VideoPreProcessorCPU(VideoSource *source, const int nLayers, Calibration &extCalib, int targetResoX, int targetResoY);
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
    void setFrameSkip(int skip) { frameInc = skip; }
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
