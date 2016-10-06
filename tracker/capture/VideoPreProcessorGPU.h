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
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <capture/videoSource.h>
#include <calib/calib.h>

using namespace cv;

class VertexBuffer2;

class VideoPreProcessorGPU
{
public:
    VideoPreProcessorGPU(VideoSource *source, const int nLayers, Calibration *calib);
    ~VideoPreProcessorGPU();
    int preprocess(ImagePyramid2 &imBW, Image2 &imRGB, VertexBuffer2 *vbuffer,  float *imDepthDevIR, Image2 &imDepth, bool keyframeMode=false);
    int getWidth();
    int getHeight();
    int getDepthWidth();
    int getDepthHeight();
    bool isPlaying();
    void setVideoSource(VideoSource *kinect);
    VideoSource *getVideoSource();
    void updateCalibration();
    void release();

    int getFrame();
    void setFrame(int frame);
    void pause();
    void reset();
    bool isPaused();
private:
    int frameIndex;
    int frameInc;
    bool loopFlag;

    VideoSource *device;
    int width, dwidth;
    int height, dheight;
    const int nMultiResolutionLayers;
    void gpuPreProcess(unsigned char *rgbDev, unsigned short *disparityDev,ImagePyramid2 &imBW, Image2 &imRGB, Image2 &imDepthIR, Image2 &imDepth, VertexBuffer2 *vbuffer, bool keyframeMode);
    void uploadCalibData();
    Calibration *calib;
};
