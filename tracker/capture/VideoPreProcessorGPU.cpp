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
#include <cudakernels/hostUtils.h>
#include <VideoPreProcessorGPU.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <inttypes.h>
#include "timer/performanceCounter.h"
#include <reconstruct/basic_math.h>
#include <rendering/VertexBuffer2.h>

using namespace std;
using namespace cv;
//using namespace omp_functions;

static float *calibDataDev = NULL;
static float *uDisparityDev = NULL;

//static unsigned int histogramCPU[HISTOGRAM256_BIN_COUNT];

PerformanceCounter timer;

// CUDA mapped memories with streams for passing data to GPU
static unsigned char *mappedRGBPtr = NULL;
static unsigned char *mappedRGBDevPtr = NULL;
static cudaStream_t driverStreamRGB;
static cudaStream_t driverStreamRGBASync;

static unsigned short *mappedDepthPtr = NULL;
static unsigned short *mappedDepthDevPtr = NULL;
static cudaStream_t driverStreamDepth;
static cudaStream_t driverStreamDepthASync;

//#define PERFORMANCE_TEST

void VideoPreProcessorGPU::uploadCalibData() {
    if (calibDataDev == NULL) cudaMalloc( (void **)&calibDataDev, sizeof(float)*(CALIB_SIZE));
    cudaMemcpyAsync(calibDataDev, calib->getCalibData(), sizeof(float)*CALIB_SIZE, cudaMemcpyHostToDevice,0);
}


bool VideoPreProcessorGPU::isPlaying() {
    if (device == NULL) return false;
    if (device->isPaused()) return false;
    return true;
}

int VideoPreProcessorGPU::getFrame() {
    return frameIndex;
}
void VideoPreProcessorGPU::setFrame(int frame) {
    frameIndex = frame;
}
void VideoPreProcessorGPU::pause() {
    frameInc = !frameInc;
    if (device) device->reset();
}
void VideoPreProcessorGPU::reset() {
    frameIndex = 0; frameInc = 0;
    if (device) device->reset();
}
bool VideoPreProcessorGPU::isPaused() {
    if (frameInc == 0) return true; else return false;
}


VideoPreProcessorGPU::VideoPreProcessorGPU(VideoSource *kinect, const int nLayers, Calibration *calib) : nMultiResolutionLayers(nLayers)
{
    assert(kinect != NULL);
    this->calib = calib;

    setVideoSource(kinect);
    cudaMalloc((void **)&uDisparityDev,kinect->getDisparityWidth()*kinect->getDisparityHeight()*sizeof(float));

    checkCudaError("cudaMalloc error");
};

VideoPreProcessorGPU::~VideoPreProcessorGPU() {
}

void VideoPreProcessorGPU::release() {
    // release mapped memories and streams if allocated:
    if (mappedRGBPtr != NULL) {
        cudaFreeHost(mappedRGBPtr);
        cudaFreeHost(mappedDepthPtr);
        mappedRGBPtr = NULL;
        mappedRGBDevPtr = NULL;
        mappedDepthPtr = NULL;
        mappedDepthDevPtr = NULL;

        cudaStreamDestroy(driverStreamRGB);
        cudaStreamDestroy(driverStreamDepth);
        cudaStreamDestroy(driverStreamRGBASync);
        cudaStreamDestroy(driverStreamDepthASync);
    }
    if (calibDataDev != NULL) cudaFree(calibDataDev); calibDataDev = NULL;
    if (uDisparityDev != NULL) cudaFree(uDisparityDev); uDisparityDev = NULL;    
}


void VideoPreProcessorGPU::gpuPreProcess(unsigned char *rgbDev, unsigned short *disparityDev,ImagePyramid2 &imBW, Image2 &imRGB, Image2 &imDepthIR, Image2 &imDepth, VertexBuffer2 *vbuffer, bool keyframeMode)
{
    float delay = 0.0f;
    float delays[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    static float delaysCumulated[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    static int numExec = 0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   Image2 rgbInput(rgbDev,width,height,width*3,3,false); rgbInput.setStream(driverStreamRGB);

    #if defined(PERFORMANCE_TEST)
         cudaEventRecord(start,0);
    #endif

    cudaMemsetAsync(imDepth.devPtr,0,sizeof(float)*imDepth.width*imDepth.height);
    // process all raw data
    undistortDisparityCuda(disparityDev, uDisparityDev, calibDataDev, 640,480, imDepthIR.cudaStream);

    if (!calib->isOffsetXY()) {
        d2ZCudaHdr(uDisparityDev,&imDepthIR,calibDataDev,0,0);
    } else {
        d2ZCudaHdr(uDisparityDev,&imDepthIR,calibDataDev,-4,-3);
    }
    //d2ZCuda(disparityDev,&imDepthIR,calibDataDev);
    #if defined(PERFORMANCE_TEST)
        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[0] += delay; cudaEventRecord(start,0);
    #endif
 //   undistortRGBCuda(&rgbInput,&imRGB,calibDataDev);
    convertToHDRCuda(&rgbInput,&imRGB);
    rgb2GrayCuda(&imRGB,imBW.getImagePtr(0));

#if defined(PERFORMANCE_TEST)
        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[1] += delay; cudaEventRecord(start,0);
    #endif

    imBW.updateLayers(); /*	gradientCuda(imBW,gradX,gradY,1);*/
    #if defined(PERFORMANCE_TEST)
        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[2] += delay; cudaEventRecord(start,0);
    #endif
    //7ms before this point
    vbuffer->setStream(driverStreamRGB);
    z2CloudCuda(&imDepthIR, calibDataDev, vbuffer, &imRGB, &imBW,&imDepth,!keyframeMode); //5-6ms

    #if defined(PERFORMANCE_TEST)
        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[3] += delay;
    #endif

    #if defined(PERFORMANCE_TEST)
        for (int i = 0; i < 4; i++)
            delaysCumulated[i] = (delays[i] + numExec*delaysCumulated[i])/float(1+numExec);
         printf("d2z: %3.1f, rgb preprocess: %3.1f, layers: %3.1f, reconstruct: %3.1f\n",delaysCumulated[0],delaysCumulated[1],delaysCumulated[2],delaysCumulated[3]);
         numExec++;
    #endif

   //cudaThreadSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //	checkCudaError("GetAndFillFrames error");
}

int VideoPreProcessorGPU::preprocess( ImagePyramid2 &imBW, Image2 &imRGB, VertexBuffer2 *vbuffer, float *imDepthDevIR, Image2 &imDepth, bool keyframeMode)
{
    assert(device != NULL);


    imRGB.setStream(driverStreamRGB);
    imBW.setStream(driverStreamRGB);

    // rgbDev will hold 320x240 rgb image, disparityDev will be 640x480 disparity image
    unsigned char *rgbCPU = NULL; unsigned short *disparityCPU = NULL;
    int ret = device->fetchRawImages(&rgbCPU, &disparityCPU, frameIndex);
    if (ret) {
        frameIndex+=frameInc;
    } else {
        reset();
        return 0;
    }

    // upload to gpu
    cudaMemcpyAsync(mappedRGBPtr,rgbCPU,kinectRgbSizeSmall,cudaMemcpyHostToHost);
    cudaMemcpyAsync(mappedDepthPtr,disparityCPU,kinectDepthSize,cudaMemcpyHostToHost);
    //memcpy(mappedRGBPtr,rgbCPU,kinectRgbSizeSmall);
    //memcpy(mappedDepthPtr,disparityCPU,kinectDepthSize);

    if (ret) {
        Image2 imDepthIR(imDepthDevIR,width,height,width*sizeof(float),1,true); imDepthIR.setStream(driverStreamDepth);
        gpuPreProcess(mappedRGBPtr,mappedDepthPtr,imBW,imRGB,imDepthIR,imDepth,vbuffer,keyframeMode);
    }
    return ret;
}

int VideoPreProcessorGPU::getWidth()
{
	return width;
}

int VideoPreProcessorGPU::getHeight()
{
	return height;
}

void VideoPreProcessorGPU::updateCalibration() {
    uploadCalibData();
}

int VideoPreProcessorGPU::getDepthWidth()
{
	return width;

}

int VideoPreProcessorGPU::getDepthHeight()
{
	return height;
}

void VideoPreProcessorGPU::setVideoSource( VideoSource *kinect )
{
    width   = kinect->getWidth();
    height  = kinect->getHeight();
    dwidth  = kinect->getDisparityWidth();
    dheight = kinect->getDisparityHeight();

    frameInc = 1;
    loopFlag = false;
    frameIndex = 0;

    printf("preprocessor expects %d x %d rgb input and %d x %d disparity input\n",width,height,dwidth,dheight);

    device  = kinect;
    calib->setupCalibDataBuffer(width,height);
    uploadCalibData();
    //	printf("video source set!\n");

    if (mappedRGBPtr == NULL)  {
        // allocated mapped memory for gpu->cpu transfers
        cudaHostAlloc((void**)&mappedRGBPtr,kinectRgbSizeSmall,cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&mappedRGBDevPtr,(void*)mappedRGBPtr,0);
        cudaHostAlloc((void**)&mappedDepthPtr,kinectDepthSize,cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&mappedDepthDevPtr,(void*)mappedDepthPtr,0);
        // also determine cuda streams to use with rgb and depth images
        cudaStreamCreate(&driverStreamRGB);
        cudaStreamCreate(&driverStreamRGBASync);
        cudaStreamCreate(&driverStreamDepth);
        cudaStreamCreate(&driverStreamDepthASync);
    }
}

VideoSource *VideoPreProcessorGPU::getVideoSource() {
    return device;
}
