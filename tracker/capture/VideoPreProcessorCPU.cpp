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
#include "VideoPreProcessorCPU.h"
#include <opencv2/opencv.hpp>
//#include <libfreenect.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <inttypes.h>
//#include "timer/performanceCounter.h"
#include <tracker/basic_math.h>
//#include <rendering/VertexBuffer2.h>
//#include <multicore/multicore.h>
#include "../reconstruct/zconv.h"

using namespace std;
using namespace cv;
//using namespace omp_functions;

static float *calibDataDev = NULL;
static float *uDisparityDev = NULL;

//static unsigned int histogramCPU[HISTOGRAM256_BIN_COUNT];

// gray pyramid
static cv::Mat distortedGray;
static cv::Mat distortedGrayHDR;
static cv::Mat normalizedGray;
static cv::Mat grayImageHDR[16];
static cv::Mat rgbImage;
static cv::Mat distortedRGBImage;


//#define PERFORMANCE_TEST

bool VideoPreProcessorCPU::isPlaying() {
    if (device == NULL) return false;
    if (device->isPaused()) return false;
    return true;
}

int VideoPreProcessorCPU::getFrame() {
    return frameIndex;
}
void VideoPreProcessorCPU::setFrame(int frame) {
    frameIndex = frame;
}
void VideoPreProcessorCPU::pause() {
    frameInc = !frameInc;
    if (device) device->reset();
}
void VideoPreProcessorCPU::reset() {
    frameIndex = 0; frameInc = 0;
    if (device) device->reset();
}
bool VideoPreProcessorCPU::isPaused() {
    if (frameInc == 0) return true; else return false;
}


VideoPreProcessorCPU::VideoPreProcessorCPU(VideoSource *kinect, const int nLayers, Calibration &calib, int targetResoX, int targetResoY) : nMultiResolutionLayers(nLayers)
{
    assert(kinect != NULL);
    this->localCalib.copyCalib(calib);

    setVideoSource(kinect,targetResoX,targetResoY);

    brightnessNormalizationFlag = false;

    depthMapR = Mat::zeros(height,width,CV_32FC1);
    depthMapL = Mat::zeros(height,width,CV_32FC1);

    fullPointSet = new ProjectData[width*height];
    memset(fullPointSet,0,sizeof(ProjectData)*width*height);
    pixelSelectionAmount = 0;
    m_planarityIndex = 0;
    m_texturingIndex = 0;
    selectedPoints3d = NULL;
    selectedPoints2d = NULL;
    planeComputed = false;
};

VideoPreProcessorCPU::~VideoPreProcessorCPU() {
}

void VideoPreProcessorCPU::release() {
    depthMapR.release();
    depthMapL.release();
    delete[] fullPointSet;
    if (selectedPoints3d != NULL) delete[] selectedPoints3d;
    if (selectedPoints2d != NULL) delete[] selectedPoints2d;
}

void VideoPreProcessorCPU::downSample2( Mat  &img, Mat &img2 )
{
        if (img.channels() != 1) assert(0);

        unsigned int newWidth  = (unsigned int)img.cols/2;
        unsigned int newHeight = (unsigned int)img.rows/2;

        if (img.type() == CV_8UC1) {
            unsigned char *srcPtr = img.ptr();
            unsigned char *dstPtr = img2.ptr();

            unsigned int offset = 0;
            for (unsigned int j = 0; j < newHeight; j++) {
                    for (unsigned int i = 0; i < newWidth; i++,offset++) {
                        int offset2 = (i<<1)+(j<<1)*img.cols;
                        dstPtr[offset] = (srcPtr[offset2] + srcPtr[offset2+1] + srcPtr[offset2+img.cols] + srcPtr[offset2+1+img.cols])>>2;
                }
            }
        } else if (img.type() == CV_32FC1) {
            float *srcPtr = (float*)img.ptr();
            float *dstPtr = (float*)img2.ptr();

            unsigned int offset = 0;
            for (unsigned int j = 0; j < newHeight; j++) {
                    for (unsigned int i = 0; i < newWidth; i++,offset++) {
                        int offset2 = (i<<1)+(j<<1)*img.cols;
                        dstPtr[offset] = (srcPtr[offset2] + srcPtr[offset2+1] + srcPtr[offset2+img.cols] + srcPtr[offset2+1+img.cols])/4.0f;
                }
            }
        }
}

void VideoPreProcessorCPU::setPixelSelectionAmount(int pixelAmount) {
    pixelSelectionAmount = pixelAmount;
    if (selectedPoints3d != NULL) delete[] selectedPoints3d;
    if (selectedPoints2d != NULL) delete[] selectedPoints2d;

    selectedPoints3d = new float[pixelAmount*3];
    selectedPoints2d = new float[pixelAmount*2];
}

void VideoPreProcessorCPU::fastImageMedian(Mat &src, int *medianVal) {
    unsigned char *srcData = src.ptr();

    unsigned int hist[256]; memset(hist,0,256*sizeof(int));
    unsigned int mass = 0;
    int offset = 0;
    for (int j  = 0; j < src.rows; j++) {
        for (int i  = 0; i < src.cols; i++,offset++) {
            unsigned char v0 = srcData[offset];
            hist[v0]++;
            mass++;
        }
    }
    // seek median value
    int desiredMass = mass/2;
    int currentMass = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++) {
        currentMass += hist[i];
        if (currentMass >= desiredMass) { threshold = i; break;}
    }
    *medianVal = threshold;
}

void VideoPreProcessorCPU::normalizeBrightness( Mat &src, Mat &dst )
{
    int medianVal = 0;
    fastImageMedian(src,&medianVal);

    unsigned char *srcData = src.ptr();
    unsigned char *dstData = dst.ptr();

    // set median to 128, this normalizes global brightness variations
    int offset = 0;
    for (int j  = 0; j < src.rows; j++) {
        for (int i  = 0; i < src.cols; i++,offset++) {
            int v0 = srcData[offset];
            int v1 = MIN(MAX(v0 + (128 - medianVal),0),255);
            dstData[offset] = (unsigned char)v1;
        }
    }
}


Mat &VideoPreProcessorCPU::getGrayImage() {
    return distortedGrayHDR;//grayImageHDR[index];
}

Mat &VideoPreProcessorCPU::getRGBImage() {
    return rgbImage;
}

Mat &VideoPreProcessorCPU::getDistortedRGBImage() {
    return distortedRGBImage;
}

Mat &VideoPreProcessorCPU::getDepthImageR() {
    return depthMapR;
}

Mat &VideoPreProcessorCPU::getDepthImageL() {
    return depthMapL;
}

void VideoPreProcessorCPU::planeRegressionPhotometric(ProjectData *fullPointSet, int count, float *mean, float *normal) {
    mean[0] = 0; mean[1] = 0; mean[2] = 0;
    float realcnt = 0;
    for (int i = 0; i < count; i++) {
        if (fullPointSet[i].magGrad > 10) {
            mean[0] += fullPointSet[i].rx;
            mean[1] += fullPointSet[i].ry;
            mean[2] += fullPointSet[i].rz;
            realcnt++;
        }
    }
    mean[0] /= realcnt; mean[1] /= realcnt; mean[2] /= realcnt;

    float mtx[9] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < count; i++) {
        if (fullPointSet[i].magGrad > 10) {
            float n[3] = {0,0,0};
            n[0] = fullPointSet[i].rx-mean[0];
            n[1] = fullPointSet[i].ry-mean[1];
            n[2] = fullPointSet[i].rz-mean[2];
            mtx[0] += n[0]*n[0]; mtx[1] += n[0]*n[1]; mtx[2] += n[0]*n[2];
            mtx[3] += n[0]*n[1]; mtx[4] += n[1]*n[1]; mtx[5] += n[1]*n[2];
            mtx[6] += n[0]*n[2]; mtx[7] += n[1]*n[2]; mtx[8] += n[2]*n[2];
        }
    }
    cv::Mat E, V;
    cv::Mat M(3,3,CV_32FC1,mtx);
    cv::eigen(M,E,V);
    normal[0] = V.at<float>(2,0);
    normal[1] = V.at<float>(2,1);
    normal[2] = V.at<float>(2,2);
    normalize(&normal[0]);
//    printf("mean: %f %f %f\n",mean[0],mean[1],mean[2]);
//    printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
}


float VideoPreProcessorCPU::planeRegressionGeometric(ProjectData *fullPointSet, int count, float *mean, float *normal, float *eigenU, float *eigenV) {
    mean[0] = 0; mean[1] = 0; mean[2] = 0;
    float realcnt = 0;
    for (int i = 0; i < count; i++) {
        if (fabs(fullPointSet[i].magGrad) > 0) {
            mean[0] += fullPointSet[i].rx;
            mean[1] += fullPointSet[i].ry;
            mean[2] += fullPointSet[i].rz;
            realcnt++;
        }
    }
    mean[0] /= realcnt; mean[1] /= realcnt; mean[2] /= realcnt;

    float mtx[9] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < count; i++) {
        if (fabs(fullPointSet[i].magGrad) > 0) {
            float n[3] = {0,0,0};
            n[0] = fullPointSet[i].rx-mean[0];
            n[1] = fullPointSet[i].ry-mean[1];
            n[2] = fullPointSet[i].rz-mean[2];
            mtx[0] += n[0]*n[0]; mtx[1] += n[0]*n[1]; mtx[2] += n[0]*n[2];
            mtx[3] += n[0]*n[1]; mtx[4] += n[1]*n[1]; mtx[5] += n[1]*n[2];
            mtx[6] += n[0]*n[2]; mtx[7] += n[1]*n[2]; mtx[8] += n[2]*n[2];
        }
    }
    for (int i = 0; i < 9; i++) mtx[i] /= float(realcnt);
    cv::Mat E, V;
    cv::Mat M(3,3,CV_32FC1,mtx);
    cv::eigen(M,E,V);
    normal[0] = V.at<float>(2,0);
    normal[1] = V.at<float>(2,1);
    normal[2] = V.at<float>(2,2);
    eigenU[0] = V.at<float>(0,0);
    eigenU[1] = V.at<float>(0,1);
    eigenU[2] = V.at<float>(0,2);
    eigenV[0] = V.at<float>(1,0);
    eigenV[1] = V.at<float>(1,1);
    eigenV[2] = V.at<float>(1,2);
     normalize(&normal[0]);
    normalize(&eigenU[0]);
    normalize(&eigenV[0]);
    return E.at<float>(0,0);
//    for (int i = 0; i < 3; i++) printf("E[%d]:%e\n",i,E.at<float>(i,i));
//    printf("mean: %f %f %f\n",mean[0],mean[1],mean[2]);
//    printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
}


void VideoPreProcessorCPU::getPlane(float *mean, float *normal) {
    memcpy(mean,&planeMean[0],sizeof(float)*3);
    memcpy(normal,&planeNormal[0],sizeof(float)*3);
}

  // note: does this actually work?
float VideoPreProcessorCPU::calcPlanarityIndex() {
    // plane regression variables:
    float geoMean[3],geoNormal[3],eigenU[3],eigenV[3];
    planeRegressionGeometric(fullPointSet,width*height,&geoMean[0],&geoNormal[0],&eigenU[0],&eigenV[0]);

    int cnt = width*height;
//    double largestVar = 0.0f;
    int nSamples = 0;
    int nTotal   = 0;
    for (int i = 0; i < cnt; i++) {
        if (fabs(fullPointSet[i].magGrad) > 0) {
            float v[3] = {0,0,0};
            v[0] = fullPointSet[i].rx-geoMean[0];
            v[1] = fullPointSet[i].ry-geoMean[1];
            v[2] = fullPointSet[i].rz-geoMean[2];
            float dotU = v[0]*eigenU[0]+v[1]*eigenU[1]+v[2]*eigenU[2];

            v[0] -= dotU*eigenU[0];
            v[1] -= dotU*eigenU[1];
            v[2] -= dotU*eigenU[2];

            float dotV = v[0]*eigenV[0]+v[1]*eigenV[1]+v[2]*eigenV[2];
            v[0] -= dotV*eigenV[0];
            v[1] -= dotV*eigenV[1];
            v[2] -= dotV*eigenV[2];

            float dotN = v[0]*geoNormal[0]+v[1]*geoNormal[1]+v[2]*geoNormal[2];


            float delta = fabs(dotN);//v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
            if (delta < 200.0f) nSamples++;
            nTotal++;
/*            error += delta*delta;
            delta = v[0]*geoEigen[0]+v[1]*geoEigen[1]+v[2]*geoEigen[2];
            largestVar += delta*delta;*/
        }
//        nSamples++;
    }
    // normalize sample amount out:
    //error /= nSamples; error = sqrtf(error+1e-7f);
//    largestVar /= nSamples; largestVar = sqrtf(largestVar+1e-7f);
    // normalize dataset out
//    return 1.0f-(error/largestVar);
    //return max(1.0f-float(error/250.0f),0.0f);
    return float(nSamples)/float(nTotal);
}

static Mat debugImage640(480,640,CV_8UC3);

void genDebugImageFF(cv::Mat &k2DepthMat, float minDist, float maxDist, cv::Mat &debugImage) {
    int w = k2DepthMat.cols;
    int h = k2DepthMat.rows;
    float *ptr = (float*)k2DepthMat.ptr();
    unsigned char *dst = (unsigned char*)debugImage.ptr();
    int n = w*h;
    for (int i = 0; i < n; i++) {
        int val = 0;
        if (ptr[i] >= 0 && ptr[i] <= 1.0f) {
            val = ptr[i]*255;
        }
        dst[i*3+0] = val;
        dst[i*3+1] = val;
        dst[i*3+2] = val;
    }
}


float VideoPreProcessorCPU::calcTexturingIndex() {
    float *ptr = (float*)distortedGrayHDR.ptr();
    int nSamples = 0;
    double texturing = 0.0f;
    for (int j = 1; j < height-1; j++) {
        for (int i = 1; i < width-1; i++)
        {
            int offset = i+j*width;
            float u0 = ptr[offset-1];
            float u1 = ptr[offset+1];
            float v0 = ptr[offset-width];
            float v1 = ptr[offset+width];
            float grad = fabs(v1-v0)+fabs(u1-u0);
            texturing += (grad > 32);
            nSamples++;
        }
    }
    // normalize sample amount out:
    texturing /= nSamples; texturing *= 4;
    if (texturing > 1.0f) texturing = 1.0f;
    return texturing;
}

ProjectData *VideoPreProcessorCPU::getPointCloud() {
    return fullPointSet;
}

// generate hdr grayscale pyramid with normalized brightness
void VideoPreProcessorCPU::cpuPreProcess(unsigned char *rgbCPU, float *depthMapCPU, bool normalizeDepth, bool calcPlanarity, bool calcTexturing) {
    ZConv zconv;
    localCalib.setupCalibDataBuffer(width,height);

    float minDepth = localCalib.getMinDist();
    float maxDepth = localCalib.getMaxDist();

    Mat inputDepthMapL(height,width,CV_32FC1,depthMapCPU);
    inputDepthMapL.copyTo(depthMapL);

    zconv.baselineTransform((float*)depthMapL.ptr(),(float*)depthMapR.ptr(),width,height,&localCalib);
    Mat rgbHeader(height, width, CV_8UC3, rgbCPU);
    if ( distortedGray.data == NULL) {
        distortedGray = Mat::zeros(height,width,CV_8UC1);
        distortedGrayHDR = Mat::zeros(height,width,CV_32FC1);
        rgbImage = rgbHeader.clone();
        distortedRGBImage = rgbHeader.clone();
    }
    rgbHeader.copyTo(distortedRGBImage);
    cvtColor(rgbHeader,distortedGray,cv::COLOR_RGB2GRAY); // 0.5ms with 1 core
    distortedGray.convertTo(distortedGrayHDR,CV_32FC1);
    OMPFunctions *multicore = getMultiCoreDevice();

    float *ptr = NULL;
    cv::Mat K(3,3,CV_32FC1); ptr = (float*)K.ptr(); for (int i = 0; i < 9; i++) ptr[i] = localCalib.getCalibData()[KR_OFFSET+i];
    cv::Mat kc(5,1,CV_32FC1); ptr = (float*)kc.ptr(); for (int i = 0; i < 5; i++) ptr[i] = localCalib.getCalibData()[KcR_OFFSET+i];
    undistort(rgbHeader,rgbImage,K,kc);
    //multicore->undistortF(rgbHeader,rgbImage,&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
    //rgbHeader.copyTo(rgbImage);

    zconv.baselineWarp((float*)depthMapL.ptr(),distortedGray.ptr(),fullPointSet,width,height,&localCalib);
    if (!planeComputed) {
        //    calib->setupCalibDataBuffer(width,height);
        //    multicore->undistortF(distortedGray,hdrGray,&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
        // plane regression variables:
        planeRegressionPhotometric(fullPointSet,width*height,&planeMean[0],&planeNormal[0]);
        planeComputed = true;
    }
    if (calcPlanarity) {
        m_planarityIndex = calcPlanarityIndex();
    } else {
        m_planarityIndex = 0;
    }
    if (calcTexturing) {
        m_texturingIndex = calcTexturingIndex();
    } else {
        m_texturingIndex = 0;
    }
    if (normalizeDepth) {
        int sz = depthMapL.cols*depthMapL.rows;
        float *dptrL = (float*)depthMapL.ptr();
        float *dptrR = (float*)depthMapR.ptr();
        for (int i = 0; i < sz; i++) {
            dptrL[i] = MAX(MIN(dptrL[i],maxDepth)-minDepth,0)/(maxDepth-minDepth);
            dptrR[i] = MAX(MIN(dptrR[i],maxDepth)-minDepth,0)/(maxDepth-minDepth);
        }
/*
            genDebugImageFF(depthMapL,localCalib.getMinDist(),localCalib.getMaxDist(),debugImage640);
            char buf[512]; static int loadIndex=0;
            sprintf(buf,"zmapR%04d.ppm",loadIndex);
            imwrite(buf,debugImage640);
          //  sprintf(buf,"rgb%04d.ppm",loadIndex);
          //  imwrite(buf,distortedRGBImage);
            loadIndex++;
*/
    }
    // early exit:
    return;
    /*
    multicore->generateDepthMap(fullPointSet, depthMapR);


    if (pixelSelectionAmount <= 0) return;


    //static int counter = 0;
    //char buf[512];
    //sprintf(buf,"scratch/tmpimg%04d.ppm",counter++);
    //imwrite(buf,distortedGray);

    int histogram[256]; int size = width*height;
    memset(histogram,0,sizeof(int)*256);
    for (int i = 0; i < size; i++) {
        histogram[fullPointSet[i].magGrad]++;
    }

    int mass = 0; int thresholdBin = 255;
    for (int i = 255; i >= 0; i--) {
        mass += histogram[i];
        if (mass > pixelSelectionAmount) {
            thresholdBin = i;
            break;
        }
    }

    int numSelected = 0;
    for (int i = 0; i < size; i++) {
        if (fullPointSet[i].magGrad > thresholdBin) {
            selectedPoints3d[numSelected*3+0] = fullPointSet[i].rx;
            selectedPoints3d[numSelected*3+1] = fullPointSet[i].ry;
            selectedPoints3d[numSelected*3+2] = fullPointSet[i].rz;
            selectedPoints2d[numSelected*2+0] = fullPointSet[i].px;
            selectedPoints2d[numSelected*2+1] = fullPointSet[i].py;
            numSelected++;
        }
    }
    if (numSelected == pixelSelectionAmount) return;

    for (int i = 0; i < size; i++) {
        if (fullPointSet[i].magGrad == thresholdBin) {
            selectedPoints3d[numSelected*3+0] = fullPointSet[i].rx;
            selectedPoints3d[numSelected*3+1] = fullPointSet[i].ry;
            selectedPoints3d[numSelected*3+2] = fullPointSet[i].rz;
            selectedPoints2d[numSelected*2+0] = fullPointSet[i].px;
            selectedPoints2d[numSelected*2+1] = fullPointSet[i].py;
            numSelected++;
            if (numSelected == pixelSelectionAmount) return;
        }
    }
*/
    /*
    OMPFunctions *multicore = getMultiCoreDevice();

    for (int i = 0; i < nMultiResolutionLayers; i++) {
        if (grayImageHDR[i].data == NULL) {
            grayImageHDR[i] = Mat::zeros(height>>i,width>>i,CV_32FC1);
            printf("allocated gray %d\n",i);
        }
        if (i == 0) {
            // allocate storage for undistortion
            if ( distortedGray.data == NULL) {
                distortedGray = Mat::zeros(height,width,CV_8UC1);
                undistortedGray = Mat::zeros(height,width,CV_8UC1);
                normalizedGray = Mat::zeros(height,width,CV_8UC1);
            }
            cvtColor(rgbHeader,distortedGray,CV_RGB2GRAY); // 0.5ms with 1 core
            // normalize brightness?
            if (brightnessNormalizationFlag) {
                normalizeBrightness(distortedGray,normalizedGray);
                multicore->undistortF(normalizedGray,grayImageHDR[0],&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
            } else {
               multicore->undistortF(distortedGray,grayImageHDR[0],&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
            }
        } else {
            downSample2(grayImageHDR[i-1],grayImageHDR[i]);
        }
    }
*/
}


int VideoPreProcessorCPU::preprocess(bool normalizeDepth, bool calcPlanarityIndex, bool calcTexturingIndex)
{
    assert(device != NULL);
    localCalib.setupCalibDataBuffer(width,height);
    // rgbDev and disparityDev will hold targetResoX x targetResoY images
    unsigned char *rgbCPU = NULL; float *depthCPU = NULL;
    int ret = device->fetchRawImages(&rgbCPU, &depthCPU,frameIndex,width,&localCalib);
    if (ret) {        
        cpuPreProcess(rgbCPU,depthCPU,normalizeDepth,calcPlanarityIndex,calcTexturingIndex);
        frameIndex+=frameInc;
    } else {
        reset();
    }
    return ret;
}

int VideoPreProcessorCPU::getWidth()
{
	return width;
}

int VideoPreProcessorCPU::getHeight()
{
	return height;
}

int VideoPreProcessorCPU::getDepthWidth()
{
	return width;

}

int VideoPreProcessorCPU::getDepthHeight()
{
	return height;
}

void VideoPreProcessorCPU::setBrightnessNormalization(bool flag) {
    brightnessNormalizationFlag = flag;
}

void VideoPreProcessorCPU::setVideoSource( VideoSource *kinect, int targetWidth, int targetHeight )
{
    width   = targetWidth;
    height  = targetHeight;
    dwidth  = targetWidth;
    dheight = targetHeight;
    frameInc = 1;
    loopFlag = false;
    frameIndex = 0;
    printf("preprocessor expects %d x %d rgb input and %d x %d disparity input\n",width,height,dwidth,dheight);

    device  = kinect;
    localCalib.setupCalibDataBuffer(width,height);
    planeComputed = false;
    m_planarityIndex = 0;
    m_texturingIndex = 0;


    //	printf("video source set!\n");
}

VideoSource *VideoPreProcessorCPU::getVideoSource() {
    return device;
}
