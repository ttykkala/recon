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
//#include <GL/gl.h>	// OpenGL32 Library
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>
//#include <libfreenect.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include "KinectDisparityCompressor.h"
#include "fileSource.h"
#include <calib/calib.h>
#include <reconstruct/zconv.h>
#include <multicore/multicore.h>

using namespace std;
using namespace cv;


#define RGB_WIDTH 640
#define RGB_HEIGHT 480
#define RGB_WIDTH_SMALL 320
#define RGB_HEIGHT_SMALL 240
#define DISPARITY_WIDTH 640
#define DISPARITY_HEIGHT 480
/*#define COMPRESSED_DISPARITY_WIDTH 320
#define COMPRESSED_DISPARITY_HEIGHT 240
const int kinectRgbSize   = RGB_WIDTH*RGB_HEIGHT*3*sizeof(unsigned char);
const int kinectRgbSizeSmall   = RGB_WIDTH_SMALL*RGB_HEIGHT_SMALL*3*sizeof(unsigned char);
const int kinectBayerSize = RGB_WIDTH*RGB_HEIGHT*sizeof(unsigned char);
const int compressedKinectDepthSize = COMPRESSED_DISPARITY_WIDTH*COMPRESSED_DISPARITY_HEIGHT*sizeof(unsigned short);
const int kinectDepthSize = DISPARITY_WIDTH*DISPARITY_HEIGHT*sizeof(unsigned short);
*/

static Mat rgbMat(RGB_HEIGHT,RGB_WIDTH,CV_8UC3);
static Mat rgbMatSmall(RGB_HEIGHT_SMALL,RGB_WIDTH_SMALL,CV_8UC3);
static Mat depthMat(DISPARITY_HEIGHT,DISPARITY_WIDTH,CV_32FC1);
static Mat depthMat640(480,640,CV_32FC1);
static Mat debugImage(424,512,CV_8UC3);
static Mat debugImage640(480,640,CV_8UC3);
static Mat dispMat(DISPARITY_HEIGHT,DISPARITY_WIDTH,CV_16UC1);
static Mat k2DepthMat(424,512,CV_16UC1);

FileSource::FileSource(const char *baseDir, bool flipY)
{
	loadingPathStr = baseDir;
    prevLoadIndex = -1;
	printf("loading path set to %s\n",loadingPathStr.c_str());
}

FileSource::~FileSource()
{

}

void FileSource::reset() {
    prevLoadIndex = -1;
}
/*
int FileSource::getWidth()
{
	return RGB_WIDTH_SMALL;
}

int FileSource::getHeight()
{
	return RGB_HEIGHT_SMALL;
}

int FileSource::getDisparityWidth() {
        return DISPARITY_WIDTH;
}

int FileSource::getDisparityHeight() {
    return DISPARITY_HEIGHT;
}
*/

bool loadRaw(char *buf,int width, int height,unsigned short *dst) {
    FILE *f = fopen(buf,"rb");
    if (f == NULL) return false;
    printf("loaded file %s!\n",buf);
    fread(dst,width*height*2,1,f);
    fclose(f);
    return true;
}

void convertK2RawToDepth640(unsigned short* rawDepthInMillimeters,int srcWidth, int srcHeight,int srcCx,int srcCy,float *zmap,int dstWidth, int dstHeight, int dstCx, int dstCy) {
    memset(zmap,0,sizeof(float)*dstWidth*dstHeight);
    int off1 = 0;
    for (int j = 0; j < srcHeight; j++) {
        for (int i = 0; i < srcWidth; i++,off1++) {
            int i2 = (i-srcCx)+dstCx;
            int j2 = (j-srcCy)+dstCy;
            if (i2 < 0 || i2 > dstWidth-1) continue;
            if (j2 < 0 || j2 > dstHeight-1) continue;
            int off2 = i2 + j2*dstWidth;
            zmap[off2] = (float)rawDepthInMillimeters[off1];
        }
    }
}

void downSampleDepthRaw(float *srcPtr, int srcWidth, int srcHeight, float *dstPtr, int dstWidth, int dstHeight) {
    int width = dstWidth;
    int height = dstHeight;
    int offset = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++,offset++) {
            int offset2 = i*2+j*2*width*2;
            float z1 = srcPtr[offset2];
            float z2 = srcPtr[offset2+1];
            float z3 = srcPtr[offset2+width*2];
            float z4 = srcPtr[offset2+width*2+1];
            dstPtr[offset] = MIN(MIN(MIN(z1,z2),z3),z4);
        }
    }
}

void genDebugImage(cv::Mat &k2DepthMat, cv::Mat &debugImage) {
    int w = k2DepthMat.cols;
    int h = k2DepthMat.rows;
    unsigned short *ptr = (unsigned short*)k2DepthMat.ptr();
    unsigned char *dst = (unsigned char*)debugImage.ptr();
    int n = w*h;
    for (int i = 0; i < n; i++) {
        int val = (ptr[i]>0)*255;
        dst[i*3+0] = val;
        dst[i*3+1] = val;
        dst[i*3+2] = val;
    }
}

void genDebugImageF(cv::Mat &k2DepthMat, float minDist, float maxDist, cv::Mat &debugImage) {
    int w = k2DepthMat.cols;
    int h = k2DepthMat.rows;
    float *ptr = (float*)k2DepthMat.ptr();
    unsigned char *dst = (unsigned char*)debugImage.ptr();
    int n = w*h;
    for (int i = 0; i < n; i++) {
        int val = 0;
        if (ptr[i] >= minDist && ptr[i] <= maxDist) {
            val = (ptr[i]/4500.0f)*255;
        }
        dst[i*3+0] = val;
        dst[i*3+1] = val;
        dst[i*3+2] = val;
    }
}


// produces rgb images both in 640,320 resolutions and returns the one which is desired
int FileSource::fetchRawImages(unsigned char **rgbCPU, float **depthCPU, int frameIndex, int rgbTargetResoX, Calibration *calib)
{
    int loadIndex = frameIndex+1;
    char buf[512];
    // 640?
    int targetResoX=0,targetResoY=0;
    if (rgbTargetResoX == rgbMat.cols) {
        *rgbCPU = rgbMat.ptr();
        targetResoX = rgbMat.cols;
        targetResoY = rgbMat.rows;
    } else { // 320
        *rgbCPU = rgbMatSmall.ptr();
        targetResoX = rgbMatSmall.cols;
        targetResoY = rgbMatSmall.rows;
    }
    *depthCPU = (float*)depthMat.ptr();

//    if (prevLoadIndex == loadIndex) return 1;
//    prevLoadIndex = loadIndex;

    bool loadOk = true;

    sprintf(buf,"%s/bayer_rgbimage%04d.ppm",loadingPathStr.c_str(),loadIndex);
    Mat bayerHeader = imread(buf,0);
    if (bayerHeader.data!=NULL) {
        cvtColor(bayerHeader,rgbMat,cv::COLOR_BayerGB2RGB);
        cv::pyrDown(rgbMat,rgbMatSmall);//,rgbMatSmall.size());
    } else {
        sprintf(buf,"%s/%04d.ppm",loadingPathStr.c_str(),loadIndex);
        Mat rgbHeader = imread(buf,-1);
        if (rgbHeader.data == NULL) { /*printf("file %s not valid!\n",buf);*/ loadOk =false; }
        else {
            //printf("loaded image (%dx%d): %s\n",rgbHeader.cols,rgbHeader.rows,buf);
            if (rgbHeader.cols == 320 && rgbHeader.rows == 240) {
                cvtColor(rgbHeader,rgbMatSmall,cv::COLOR_RGB2BGR);
                cv::pyrUp(rgbMatSmall,rgbMat);
            } else if (rgbHeader.cols == 640 && rgbHeader.rows == 480) {
                cvtColor(rgbHeader,rgbMat,cv::COLOR_RGB2BGR);
                cv::pyrDown(rgbMat,rgbMatSmall);
            } else {
                printf("file %s not valid!\n",buf); loadOk = false;
            }
        }
    }

    sprintf(buf,"%s/rawdepth%04d.ppm",loadingPathStr.c_str(),loadIndex);
    Mat dispHeader = imread(buf,-1);
//    if (dispHeader.data == NULL) { /*printf("file %s not valid!\n",buf);*/ loadOk = false; }
    if (dispHeader.data != NULL) {
        ZConv zconv;
        dispHeader.copyTo(dispMat);
        zconv.d2z((unsigned short*)dispMat.ptr(),dispMat.cols,dispMat.rows,(float*)depthMat.ptr(),targetResoX,targetResoY,calib);
        return loadOk;
    }
    sprintf(buf,"%s/rawdepth-kinect2-%04d.raw",loadingPathStr.c_str(),loadIndex);
    if (loadRaw(buf,512,424,(unsigned short*)k2DepthMat.ptr())) {
      /*  genDebugImage(k2DepthMat,debugImage);
        char buf[512];
        sprintf(buf,"debug%04d.ppm",loadIndex);
        imwrite(buf,debugImage);*/
        // convert principal point on resolution 320 or 640 into 640x480 format
        float cxDst = 640.0f*calib->getCalibData()[KL_OFFSET+2]/float(targetResoX);
        float cyDst = 480.0f*calib->getCalibData()[KL_OFFSET+5]/float(targetResoY);
        // crop borders to match raw image at 512x424 resolution
        float cxSrc = cxDst-64;
        float cySrc = cyDst-28;
       // printf("principal point: (%f,%f)\n",cxDst,cyDst);
        convertK2RawToDepth640((unsigned short*)k2DepthMat.ptr(),512,424,cxSrc,cySrc,(float*)depthMat640.ptr(),640,480,cxDst,cyDst);

        if (targetResoX == 640) {
            depthMat640.copyTo(depthMat);
        } else if (targetResoX == 320) {
            downSampleDepthRaw((float*)depthMat640.ptr(), 640,480, (float *)depthMat.ptr(), 320,240);
        } else {
            printf("invalid reso for depth: (%d,%d)\n",targetResoX,targetResoY);
            loadOk = false;
        }
     /*   genDebugImageF(depthMat,calib->getMinDist(),calib->getMaxDist(),debugImage640);
        char buf[512];
        sprintf(buf,"debug%04d.ppm",loadIndex);
        imwrite(buf,debugImage640);
*/
        //zconv.d2z((unsigned short*)dispMat.ptr(),dispMat.cols,dispMat.rows,(float*)depthMat.ptr(),targetResoX,targetResoY,calib);
        //return loadOk;
    } else {
        printf("file %s not valid!\n",buf); loadOk = false;
    }

    return loadOk;
}

