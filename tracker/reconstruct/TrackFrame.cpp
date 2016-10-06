/*
Copyright 2016 Tommi M. Tykk�l�

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
#include <stdio.h>
#include <string>
#include "timer/performanceCounter.h"
#include <hostUtils.h>
#include <cuda_funcs.h>
#include <tracker/basic_math.h>
#include <rendering/VertexBuffer2.h>
#include <opencv2/opencv.hpp>
#include <multicore/multicore.h>
#include <calib/calib.h>
#include <timer/performanceCounter.h>
#include <reconstruct/TrackFrame.h>

//#define PERFORMANCE_TEST

using namespace std;
using namespace cv;

static int keyFrameRefCount = 0;

static float *residualDev = NULL;
static float *weightedResidualDev = NULL;
static float *weightsDev = NULL;
static float *zWeightsDev = NULL;
static float *weightedJacobianTDev = NULL;
static float *JtJDev = NULL;
static float *residual6Dev = NULL;
static unsigned char *selectionMaskDev = NULL; // selection mask for z-averaging after poses have been estimated
//static unsigned int *histogramDev = NULL;
//static unsigned int *partialHistogramsDev = NULL;
static float *histogramFloatDev = NULL;
static float *gradientScratchDev = NULL;
static float *d_hist = NULL;
// these constant scales are only to make sure that parameter vector X contains uniform magnitudes in optimization
static float optScaleIn = 1e-3f;
static float optScaleOut = 1e+3f;

// generate pixel selection mask for sift descriptors
void TrackFrame::selectPixels(Mat &img, Mat &depth, Mat &baseMask, Mat &pixelSelectionMask, int amount) {
    unsigned char *selectionMask = pixelSelectionMask.ptr();
    memset(selectionMask,0,img.cols*img.rows);
    float *zptr = (float*)depth.ptr();

    unsigned int hist[512]; memset(hist,0,512*sizeof(int));
    unsigned char *data = img.ptr();
    unsigned int mass = 0;
    unsigned char *mask = baseMask.ptr();

    // compute selection score per pixel (gradient magnitude)
    for (int j  = 1; j < img.rows-1; j++) {
        for (int i  = 1; i < img.cols-1; i++) {
            int offset = i + j*img.cols;
            if (zptr[offset] <= 0) continue;
            if (mask[offset] == 0) continue;
            unsigned char v0 = data[offset];
            int v1x = data[offset+1];
            int v0x = data[offset-1];
            int v1y = data[offset+img.cols];
            int v0y = data[offset-img.cols];
            int bin = abs(v1x-v0x)+abs(v1y-v0y);
            selectionMask[offset] = bin;
            hist[bin]++;
            mass++;
        }
    }
    // compute mass threshold for % of all pixel values
    int desiredMass = amount;
    int currentMass = 0;
    int threshold = 1;
    for (int i = 511; i >= 1; i--) {
        currentMass += hist[i];
        if (currentMass >= desiredMass) { threshold = i; break;}
    }

//    printf("selection threshold: %d\n",threshold);
    // mark % best pixels into mask
    int offset = 0;
    for (int j  = 0; j < img.rows; j++) {
        for (int i  = 0; i < img.cols; i++,offset++) {
            if (selectionMask[offset] >= threshold) selectionMask[offset] = 255;
            else selectionMask[offset] = 0;
        }
    }
}

void TrackFrame::selectPixels(Mat &img, Mat &depth, Mat &baseMask, Mat &pixelSelectionMask, float percent)
{
    unsigned char *selectionMask = pixelSelectionMask.ptr();
    memset(selectionMask,0,img.cols*img.rows);
    float *zptr = (float*)depth.ptr();

    unsigned int hist[512]; memset(hist,0,512*sizeof(int));
    unsigned char *data = img.ptr();
    unsigned int mass = 0;
    unsigned char *mask = baseMask.ptr();

    // compute selection score per pixel (gradient magnitude)
    for (int j  = 1; j < img.rows-1; j++) {
        for (int i  = 1; i < img.cols-1; i++) {
            int offset = i + j*img.cols;
            if (zptr[offset] <= 0) continue;
            if (mask[offset] == 0) continue;
            unsigned char v0 = data[offset];
            int v1x = data[offset+1];
            int v0x = data[offset-1];
            int v1y = data[offset+img.cols];
            int v0y = data[offset-img.cols];
            int bin = abs(v1x-v0x)+abs(v1y-v0y);
            selectionMask[offset] = bin;
            hist[bin]++;
            mass++;
        }
    }
    // compute mass threshold for % of all pixel values
    int desiredMass = int(percent * float(mass));
    int currentMass = 0;
    int threshold = 1;
    for (int i = 511; i >= 1; i--) {
        currentMass += hist[i];
        if (currentMass >= desiredMass) { threshold = i; break;}
    }

//    printf("selection threshold: %d\n",threshold);
    // mark % best pixels into mask
    int offset = 0;
    for (int j  = 0; j < img.rows; j++) {
        for (int i  = 0; i < img.cols; i++,offset++) {
            if (selectionMask[offset] >= threshold) selectionMask[offset] = 255;
            else selectionMask[offset] = 0;
        }
    }
}


void TrackFrame::updateFrustum(float viewDistanceMin, float viewDistanceMax) {
    float sX,sY,len;
    float z[3],u[3],v[3];
    float o[3];
    o[0] = T[12]; o[1] = T[13];  o[2] = T[14];

    len = viewDistanceMin; if (len < 1.0f) len = 1.0f;
    z[0] =  -len*T[8]; z[1] = -len*T[9]; z[2] = -len*T[10];
    sX = tan(3.141592653f*getFovX()/360.0f)*len;
    sY = tan(3.141592653f*getFovY()/360.0f)*len;
    u[0] = sX*T[0]; u[1] = sX*T[1]; u[2] = sX*T[2];
    v[0] = sY*T[4]; v[1] = sY*T[5]; v[2] = sY*T[6];
    frustum.x0[0] = o[0]+z[0]-u[0]-v[0]; frustum.x0[1] = o[1]+z[1]-u[1]-v[1]; frustum.x0[2] = o[2]+z[2]-u[2]-v[2];
    frustum.x1[0] = o[0]+z[0]+u[0]-v[0]; frustum.x1[1] = o[1]+z[1]+u[1]-v[1]; frustum.x1[2] = o[2]+z[2]+u[2]-v[2];
    frustum.x2[0] = o[0]+z[0]+u[0]+v[0]; frustum.x2[1] = o[1]+z[1]+u[1]+v[1]; frustum.x2[2] = o[2]+z[2]+u[2]+v[2];
    frustum.x3[0] = o[0]+z[0]-u[0]+v[0]; frustum.x3[1] = o[1]+z[1]-u[1]+v[1]; frustum.x3[2] = o[2]+z[2]-u[2]+v[2];

    len = viewDistanceMax; if (len < 1.0f) len = 1.0f;
    z[0] =  -len*T[8]; z[1] = -len*T[9]; z[2] = -len*T[10];
    sX = tan(3.141592653f*getFovX()/360.0f)*len;
    sY = tan(3.141592653f*getFovY()/360.0f)*len;
    u[0] = sX*T[0]; u[1] = sX*T[1]; u[2] = sX*T[2];
    v[0] = sY*T[4]; v[1] = sY*T[5]; v[2] = sY*T[6];
    frustum.y0[0] = o[0]+z[0]-u[0]-v[0]; frustum.y0[1] = o[1]+z[1]-u[1]-v[1]; frustum.y0[2] = o[2]+z[2]-u[2]-v[2];
    frustum.y1[0] = o[0]+z[0]+u[0]-v[0]; frustum.y1[1] = o[1]+z[1]+u[1]-v[1]; frustum.y1[2] = o[2]+z[2]+u[2]-v[2];
    frustum.y2[0] = o[0]+z[0]+u[0]+v[0]; frustum.y2[1] = o[1]+z[1]+u[1]+v[1]; frustum.y2[2] = o[2]+z[2]+u[2]+v[2];
    frustum.y3[0] = o[0]+z[0]-u[0]+v[0]; frustum.y3[1] = o[1]+z[1]-u[1]+v[1]; frustum.y3[2] = o[2]+z[2]-u[2]+v[2];
}

void TrackFrame::getScenePoint(int i, int j, Calibration &calib, float *x, float *y, float *z) {
    float *K = &calib.getCalibData()[KR_OFFSET];
    float iK[9]; inverse3x3(K,&iK[0]);
    float *zmap = (float*)depthRGB.ptr();
    float p[4],r[4];
    get3DPoint(float(i),float(j),zmap[i+j*depthRGB.cols], iK,&p[0],&p[1],&p[2]); p[3] = 1;
    float tT[16];
    transpose4x4(&T[0],&tT[0]);
    transformRT3(tT,p,r); *x = r[0]; *y = r[1]; *z = r[2];
}

void TrackFrame::getRay(int i, int j, int nX, int nY, RAY *ray)
{
    ray->o[0] = T[12];
    ray->o[1] = T[13];
    ray->o[2] = T[14];

    float x[3],len;
    float tX = (float(i)+0.5f)/float(nX);
    float tY = 1.0f-(float(j)+0.5f)/float(nY);

    x[0] = frustum.x0[0]+(frustum.x1[0]-frustum.x0[0])*tX + (frustum.x3[0]-frustum.x0[0])*tY - ray->o[0];
    x[1] = frustum.x0[1]+(frustum.x1[1]-frustum.x0[1])*tX + (frustum.x3[1]-frustum.x0[1])*tY - ray->o[1];
    x[2] = frustum.x0[2]+(frustum.x1[2]-frustum.x0[2])*tX + (frustum.x3[2]-frustum.x0[2])*tY - ray->o[2];
    len = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+1e-7f);

    ray->tmin = len;
    ray->dir[0] = x[0]/len;
    ray->dir[1] = x[1]/len;
    ray->dir[2] = x[2]/len;

    x[0] = frustum.y0[0]+(frustum.y1[0]-frustum.y0[0])*tX + (frustum.y3[0]-frustum.y0[0])*tY - ray->o[0];
    x[1] = frustum.y0[1]+(frustum.y1[1]-frustum.y0[1])*tX + (frustum.y3[1]-frustum.y0[1])*tY - ray->o[1];
    x[2] = frustum.y0[2]+(frustum.y1[2]-frustum.y0[2])*tX + (frustum.y3[2]-frustum.y0[2])*tY - ray->o[2];
    len = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+1e-7f);

    ray->tmax = len;
}


void TrackFrame::setCalibDevPtr(float *calibDataDev, float normalizedK11, float normalizedK22) {
    calibDataDevEXT = calibDataDev;
    // note: principal point is assumed to be in the middle of the screen!
    fovAngleX = 180.0f*2.0f*atan(1.0f/fabs(normalizedK11))/3.141592653f;
    fovAngleY = 180.0f*2.0f*atan(1.0f/fabs(normalizedK22))/3.141592653f;
   // printf("TrackFrame fovX: %f, fovY: %f\n",fovAngleX,fovAngleY);
}


void TrackFrame::selectPixels(float pixelSelectionPercent) {
   selectPixels(grayImage, depthCPU, baseMask, pixelSelectionMask, pixelSelectionPercent);
//   char buf[512];
//   sprintf(buf,"scratch/selectedPixels%d.ppm",int(pixelSelectionPercent*100.0f+0.5f));
//   imwrite(buf,pixelSelectionMask);

   int size = pixelSelectionMask.rows*pixelSelectionMask.cols;
   unsigned char *srcPtr = pixelSelectionMask.ptr();

   unsigned int *dstPtr = vbuffer.getIndexBufferCPU();
   int numMaxElems = vbuffer.getMaxElementsCount();

   int numElems = 0;
   for (int offset = 0; offset < size; offset++) {
       if ((srcPtr[offset] > 0) && (numElems < numMaxElems)) {
           dstPtr[numElems] = offset; numElems++;
       }
   }

   unsigned int  *indexPointer = (unsigned int*)vbuffer.lockIndex();
   cudaMemcpyAsync(indexPointer,dstPtr,sizeof(int)*numElems,cudaMemcpyHostToDevice,0);
   vbuffer.unlockIndex();
   vbuffer.setElementsCount(numElems);
}

// assume vbufferExt is locked
void TrackFrame::selectPixelsGPUCompressed(VertexBuffer2 &vbufferExt, int pixelSelectionAmount,Image2 *imDepth, bool rgbVisualization) {
    vbuffer.lock();
    vbuffer.lockIndex();
//    printf("num input vertices: %d, elements: %d\n",vbufferExt.getVertexCount(),vbufferExt.getElementsCount());
//    printf("num current vertices: %d, elements: %d\n",vbuffer.getMaxVertexCount(),vbuffer.getMaxElementsCount());
//    printf("pixelSelection amount: %d\n",pixelSelectionAmount);
//    fflush(stdin); fflush(stdout);

    extractGradientMagnitudes(&vbufferExt,gradientScratchDev);
    cudaHista(gradientScratchDev, histogramFloatDev, vbufferExt.getVertexCount(), 256,d_hist);

    //histogram256VBufGradMag(histogramDev,partialHistogramsDev,gradientScratchDev,vbufferExt.getVertexCount(),pixelSelectionAmount);
    //filterIndices3(&vbufferExt, gradientScratchDev, histogramDev, pixelSelectionAmount);
    filterIndices4(&vbufferExt, gradientScratchDev, histogramFloatDev, pixelSelectionAmount,256);

    addVertexAttributesCuda(imDepth, calibDataDevEXT, &vbufferExt, &grayPyramid);
    allocateJacobian(pixelSelectionAmount);
    precomputeJacobian(&vbufferExt);
    compressVertexBuffer(&vbufferExt,&vbuffer,rgbVisualization);
    vbuffer.unlock();
    vbuffer.unlockIndex();
}

void TrackFrame::selectPixelsGPUCompressedLock(VertexBuffer2 &vbufferExt, int pixelSelectionAmount,Image2 *imDepth, ImagePyramid2 *grayPyramidExt) {
    vbuffer.lock();
    vbuffer.lockIndex();
    extractGradientMagnitudes(&vbufferExt,gradientScratchDev);
    cudaHista(gradientScratchDev, histogramFloatDev, vbufferExt.getVertexCount(), 256,d_hist);
//    cudaHistb(gradientScratchDev, histogramFloatDev, vbufferExt.getVertexCount(), 256);
    //histogram256VBufGradMag(histogramDev,partialHistogramsDev,gradientScratchDev,vbufferExt.getVertexCount(),pixelSelectionAmount);
    //filterIndices3(&vbufferExt, gradientScratchDev, histogramDev, pixelSelectionAmount);
    filterIndices4(&vbufferExt, gradientScratchDev, histogramFloatDev, pixelSelectionAmount,256);

//    filterIndices(&vbufferExt,histogramDev,pixelSelectionAmount);
    addVertexAttributesCuda(imDepth, calibDataDevEXT, &vbufferExt, grayPyramidExt);
    allocateJacobian(pixelSelectionAmount);
    precomputeJacobian(&vbufferExt);
    compressVertexBuffer(&vbufferExt,&vbuffer);
}
/*
void TrackFrame::selectPixelsGPUCompressed2(float *verticesExt, int *indicesExt, int vertexCount, int pixelSelectionAmount, int stride) {
    // generate histogram of gradient magnitudes    
    histogram256VBufGradMagRaw(histogramDev,partialHistogramsDev,verticesExt,vertexCount*stride*sizeof(float),pixelSelectionAmount,stride);
    filterIndices2(verticesExt,vertexCount,histogramDev,pixelSelectionAmount,indicesExt,stride);
    compressVertexBuffer2(indicesExt,verticesExt,pixelSelectionAmount,stride,&vbuffer);
}*/

void TrackFrame::normalizeDepthMap(cv::Mat &depthMap, cv::Mat &normalizedMap, float depthMin, float depthMax) {
    int fw = depthMap.cols;
    int fh = depthMap.rows;
    float *srcData = (float*)depthMap.ptr();
    float *dstData = (float*)normalizedMap.ptr();

    int offset=0;
    for (int j = 0; j < fh; j++) {
        for (int i = 0; i < fw; i++,offset++) {
            dstData[offset] = (srcData[offset]-depthMin)/(depthMax-depthMin);
        }
    }
}

void TrackFrame::generatePointsWithoutNormals(Calibration *calib) {
    int fw = depthCPU.cols;
    int fh = depthCPU.rows;

    assert(fw == xyzImage.cols && fh == xyzImage.rows);

    float *srcData = (float*)depthCPU.ptr();
    float *dstData = (float*)xyzImage.ptr();

    float iKir[9]; inverse3x3(&calib->getCalibData()[KL_OFFSET],&iKir[0]);
    float *TLR = &calib->getCalibData()[TLR_OFFSET];

    int offset=0;
    for (int yi = 1; yi < (fh-1); yi++) {
        for (int xi = 1; xi < (fw-1); xi++) {
            offset = xi + yi*fw;
            float z   = srcData[offset];
            float t[3],p[3],u[3],v[3];            
            get3DPoint(float(xi),float(yi),z,iKir, &t[0], &t[1], &t[2]); transformRT3(TLR,&t[0],&p[0]);
            dstData[offset*3+0] = p[0];
            dstData[offset*3+1] = p[1];
            dstData[offset*3+2] = p[2];
        }
    }
}

void TrackFrame::generatePoints(Calibration *calib)
{
    int fw = depthCPU.cols;
    int fh = depthCPU.rows;

    assert(fw == normalImage.cols && fh == normalImage.rows);
    assert(fw == xyzImage.cols && fh == xyzImage.rows);

    calib->setupCalibDataBuffer(fw,fh);

    float *srcData = (float*)depthCPU.ptr();
    float *nData = (float*)normalImage.ptr();
    float *pData = (float*)xyzImage.ptr();

    float iKir[9]; inverse3x3(&calib->getCalibData()[KL_OFFSET],&iKir[0]);
    float *TLR = &calib->getCalibData()[TLR_OFFSET];

    int offset=0;
    for (int yi = 1; yi < (fh-1); yi++) {
        for (int xi = 1; xi < (fw-1); xi++) {
            offset = xi + yi*fw;
            float z   = srcData[offset];
            float zNu = srcData[offset+1];
            float zNv = srcData[offset+fw];

            float t[3],p[3],u[3],v[3];
            get3DPoint(float(xi),float(yi),z,iKir, &t[0], &t[1], &t[2]);     transformRT3(TLR,&t[0],&p[0]);
            get3DPoint(float(xi+1),float(yi),zNu,iKir, &t[0], &t[1], &t[2]); transformRT3(TLR,&t[0],&u[0]);
            get3DPoint(float(xi),float(yi+1),zNv,iKir, &t[0], &t[1], &t[2]); transformRT3(TLR,&t[0],&v[0]);

            float nu[3],nv[3],n[3];
            nu[0] = u[0] - p[0]; nu[1] = u[1] - p[1]; nu[2] = u[2] - p[2];
            nv[0] = v[0] - p[0]; nv[1] = v[1] - p[1]; nv[2] = v[2] - p[2];
            // compute normal as crossproduct
            n[0] =  nu[1] * nv[2] - nu[2] * nv[1];
            n[1] =-(nu[0] * nv[2] - nu[2] * nv[0]);
            n[2] =  nu[0] * nv[1] - nu[1] * nv[0];
            // normal to unit length
            float len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]+1e-5f);
            // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
            n[0] /= len; n[1] /= len; n[2] /= len;

            pData[offset*3+0] = p[0];
            pData[offset*3+1] = p[1];
            pData[offset*3+2] = p[2];

            nData[offset*3+0] = -n[0];
            nData[offset*3+1] = -n[1];
            nData[offset*3+2] = -n[2];

        }
    }
}

void TrackFrame::generateGPUData(int pixelSelectionAmount, bool renderableVBuffer, bool optimizeGPUMemory, bool undistortRGB, bool rgbVisualization,Calibration *calib, unsigned char *imRGBDev, float *imDepthDev, Image2 &depth1C, VertexBuffer2 *vbufferTemp, int nMultiResolutionLayers)
{
    int width = distortedRgbImageCPU.cols;
    int height = distortedRgbImageCPU.rows;

    Mat rgbImage2(height,width,CV_8UC3);
    Mat normalizedDepth(height,width,CV_32FC1);
    float *normalsDev = NULL;
    cudaMalloc((void **)&normalsDev, width*height*3*sizeof(float)); cudaMemset(&normalsDev[0],0,width*height*3*sizeof(float));

    vbufferTemp->lock(); vbufferTemp->lockIndex();

    int morePixels = width*height;

    if (optimizeGPUMemory) {
        vbuffer.init(pixelSelectionAmount,renderableVBuffer,COMPRESSED_STRIDE);
    } else {
        // note! vertex buffer id matches with pixel indices!
        // possible to change pixel selection % during run time
        // add pixel selection to render()
        renderableVBuffer = true;

        vbuffer.init(morePixels,renderableVBuffer,COMPRESSED_STRIDE);
    }

    // create gpu images at 320x240
    createHdrImage(NULL,width,height,3,&rgbImage,ONLY_GPU_TEXTURE, false);
    grayPyramid.createHdrPyramid(width,height,1,nMultiResolutionLayers,false,ONLY_GPU_TEXTURE,renderableVBuffer); grayPyramid.setName("grayPyramid1C");

    cvtColor(distortedRgbImageCPU,rgbImage2,cv::COLOR_RGB2BGR);
    cudaMemcpy(imRGBDev,rgbImage2.ptr(),width*height*3,cudaMemcpyHostToDevice);
    Image2 rgbInput(imRGBDev,width,height,width*3,3,false);


    rgbImage.lock();  grayPyramid.lock(); rgbInput.lock();
    convertToHDRCuda(&rgbInput,&rgbImage);
    rgb2GrayCuda(&rgbImage,&grayPyramid.getImageRef(0));
    grayPyramid.updateLayers(); //	gradientCuda(imBW,gradX,gradY,1);

    //        baselineWarp(depthCPU, grayImage,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet);
    cudaMemcpy(normalsDev,normalImage.ptr(),width*height*3*sizeof(float),cudaMemcpyHostToDevice);
    normalizeDepthMap(depthCPU,normalizedDepth,calib->getCalibData()[MIND_OFFSET],calib->getCalibData()[MAXD_OFFSET]);

    cudaMemcpy(imDepthDev,normalizedDepth.ptr(),width*height*sizeof(float),cudaMemcpyHostToDevice);
    Image2 imDepth(imDepthDev,width,height,width*sizeof(float),1,true);

    vbuffer.lock(); vbuffer.lockIndex(); depth1C.lock();
    if (optimizeGPUMemory) {
        z2CloudCuda(&imDepth, calibDataDevEXT, vbufferTemp, &rgbImage, &grayPyramid,&depth1C,true);
        setNormalsCuda(vbufferTemp,normalsDev,100.0f);
        selectPixelsGPUCompressed(*vbufferTemp,pixelSelectionAmount,&imDepth,rgbVisualization);
    } else {
        z2CloudCuda(&imDepth, calibDataDevEXT, vbufferTemp, &rgbImage, &grayPyramid,&depth1C,true);
        setNormalsCuda(vbufferTemp,normalsDev,100.0f);
        selectPixelsGPUCompressed(*vbufferTemp,morePixels,&imDepth,rgbVisualization);
//        compressVertexBuffer(vbufferTemp,&vbuffer,rgbVisualization);
        //vbuffer.setElementsCount(width*height);

    }
    // hack: visualize undistorted image instead in modelvisualizer!
    if (undistortRGB) {
        undistortRGBCuda(&rgbInput,&rgbImage,calibDataDevEXT);
    }
    vbuffer.unlock(); vbuffer.unlockIndex();
    grayPyramid.unlock(); rgbImage.unlock(); rgbInput.unlock(); depth1C.unlock();

    if (optimizeGPUMemory) {
        // point clouds are enough
        if (!undistortRGB) rgbImage.releaseData();
        // gray pyramids must exist when using interpolated key! (experimental)
        grayPyramid.releaseData();
    }
    /*        size_t freeMemoryCuda,totalMemoryCuda;
    cuMemGetInfo(&freeMemoryCuda,&totalMemoryCuda);
    printf("free memory on cuda : %f percent\n",100.0f*float(freeMemoryCuda)/float(totalMemoryCuda));
    fflush(stdout); fflush(stdin); fflush(stderr);*/
    vbufferTemp->unlock(); vbufferTemp->unlockIndex();
    cudaFree(normalsDev);
}

void TrackFrame::setBaseTransform(float *baseDev) {
    cudaMemcpyAsync(TabsDev,baseDev,sizeof(float)*16,cudaMemcpyDeviceToDevice, 0);
    cudaMemcpyAsync((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
    cudaMemcpyAsync(TnextDev,baseDev,sizeof(float)*16,cudaMemcpyDeviceToDevice, 0);
//    warpPoints(&vbuffer,weightsDev,TrelDev,calibDataDevEXT,baseBufferEXT);
}

void TrackFrame::resetTransform() {
    cudaMemcpyAsync((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
    cudaMemcpyAsync((void*)TabsDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
    cudaMemcpyAsync((void*)TnextDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
}

void TrackFrame::setupCPUTransform() {
    float mInv[16];
    float mDev[16];
    transpose4x4(&Tbase[0],&mInv[0]);
    invertRT4(&mInv[0],&mDev[0]);

    cudaMemcpyAsync((void*)TabsDev, (void*)&mDev[0],16*sizeof(float),cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync((void*)TnextDev, (void*)&mDev[0],16*sizeof(float),cudaMemcpyHostToDevice,0);
    //cudaMemcpyAsync((void*)TnextDev, (void*)TabsDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
    cudaMemcpyAsync((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
}

void TrackFrame::resetRelativeDev() {
    cudaMemcpyAsync((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice,0);
}

// inputT in OpenGL compatible form already
void TrackFrame::updateTransformGL(float *inputT) {
    memcpy(&T[0],&inputT[0],sizeof(float)*16);
    memcpy(&Tbase[0],&T[0],sizeof(float)*16);
    setupCPUTransform();
}

// inputT in standard form, is transposed into OpenGL convention
void TrackFrame::updateTransform(float *inputT) {
    transpose4x4(inputT,&T[0]);
    memcpy(&Tbase[0],&T[0],sizeof(float)*16);
    setupCPUTransform();
}

void TrackFrame::setupRelativeCPUTransform(float *T) {
    float relativeT[16];
    float invCur[16];
    float mRef[16],mCur[16];
    transpose4x4(&Tbase[0],&mRef[0]);
    transpose4x4(&T[0],&mCur[0]);

    invertRT4(&mCur[0],&invCur[0]);
    matrixMult4x4(&invCur[0], &mRef[0], &relativeT[0]);
    normalizeRT4(&relativeT[0]);
    cudaMemcpyAsync((void*)TrelDev, (void*)&relativeT[0],16*sizeof(float),cudaMemcpyHostToDevice,0);
}

void TrackFrame::setupRelativeGPUTransform(float *TDev) {
    invertPoseCuda(TabsDev, TtmpDev, 1);
    matrixMult4Cuda(TDev,TtmpDev,TrelDev);
}

void TrackFrame::getRelativeTransform(float *T, float *relativeT) {
    float invCur[16];
    float mRef[16],mCur[16];
    transpose4x4(&Tbase[0],&mRef[0]);
    transpose4x4(&T[0],&mCur[0]);
    invertRT4(&mCur[0],&invCur[0]);
    matrixMult4x4(&invCur[0], &mRef[0], &relativeT[0]);
}

TrackFrame::TrackFrame(int width, int height) {
    visible = true;
    fovAngleX = 0.0f; fovAngleY = 0.0f;
    weight = 1.0f;
    neighborIndices.clear();
    neighborKeys.clear();
    keyPoints.reserve(1000);
    normalImage = cv::Mat::zeros(height,width,CV_32FC3);
    xyzImage = cv::Mat::zeros(height,width,CV_32FC3);
    nIterations[0] = 10; nIterations[1] = 3; nIterations[2] = 2;

    float identityMatrix[16];
    identity4x4(&identityMatrix[0]);
    identity4x4(&T[0]);
    identity4x4(&Tbase[0]);

    cudaMalloc((void **)&TidentityDev, 16 * sizeof(float));  cudaMemcpy((void*)TidentityDev,&identityMatrix[0],16*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void **)&TrelDev,  16 * sizeof(float));  cudaMemcpy((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMalloc((void **)&TabsDev,  16 * sizeof(float));  cudaMemcpy((void*)TabsDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMalloc((void **)&TnextDev, 16 * sizeof(float));  cudaMemcpy((void*)TnextDev,(void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMalloc((void **)&TtmpDev, 16 * sizeof(float));   cudaMemcpy((void*)TtmpDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice);

    // shared work space memory
    if (keyFrameRefCount == 0) {
        initCudaDotProduct();
        cudaMalloc((void **)&residualDev,  maxResidualSize*sizeof(float));         cudaMemset(residualDev,0,maxResidualSize*sizeof(float));
        cudaMalloc((void **)&weightedResidualDev,  maxResidualSize*sizeof(float)); cudaMemset(weightedResidualDev,0,maxResidualSize*sizeof(float));
        cudaMalloc((void **)&residual6Dev, 6*sizeof(float));             cudaMemset(residual6Dev,0,6*sizeof(float));
        cudaMalloc((void **)&weightsDev,  maxResidualSize*sizeof(float));   cudaMemset(weightsDev, 0,maxResidualSize*sizeof(float));
        cudaMalloc((void **)&zWeightsDev,  maxResidualSize*sizeof(float));   cudaMemset(zWeightsDev, 0,maxResidualSize*sizeof(float));
        cudaMalloc((void **)&weightedJacobianTDev, 6*maxResidualSize*sizeof(float)); cudaMemset(weightedJacobianTDev,0,6*maxResidualSize*sizeof(float));
        cudaMalloc((void **)&JtJDev, 6*6*sizeof(float)); cudaMemset(JtJDev,0,6*6*sizeof(float));
        cudaMalloc((void **)&selectionMaskDev, width*height*sizeof(unsigned char)); cudaMemset(selectionMaskDev,0,width*height*sizeof(unsigned char));
//        cudaMalloc((void **)&histogramDev, HISTOGRAM256_BIN_COUNT * sizeof(unsigned int));
//        cudaMalloc((void **)&partialHistogramsDev, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(int));
        cudaMalloc((void **)&histogramFloatDev, 256 * sizeof(float));
        cudaMalloc((void**) &d_hist, 64 * 256 * sizeof(float));
        cudaMalloc((void **)&gradientScratchDev, width*height * sizeof(float));
        initHistogram64();
        printf("allocated scratch memory for keyframes!\n"); fflush(stdin); fflush(stdout);
    }
    keyFrameRefCount++; //printf("keyframe ref count : %d\n",keyFrameRefCount); fflush(stdin); fflush(stdout);

    // these must be allocated later when final pixel selection amount is clear
    jacobianTDev[0] = NULL;
    jacobianTDev[1] = NULL;
    jacobianTDev[2] = NULL;
}

void TrackFrame::reallocJacobian(int pixelSelectionAmount) {
    if (jacobianTDev[0] != NULL) cudaFree(jacobianTDev[0]); jacobianTDev[0] = NULL;
    if (jacobianTDev[1] != NULL) cudaFree(jacobianTDev[1]); jacobianTDev[1] = NULL;
    if (jacobianTDev[2] != NULL) cudaFree(jacobianTDev[2]); jacobianTDev[2] = NULL;
    allocateJacobian(pixelSelectionAmount);
}

void TrackFrame::allocateJacobian(int pixelSelectionAmount) {
    if (jacobianTDev[0] == NULL) {
        cudaMalloc((void **)&jacobianTDev[0], 6*pixelSelectionAmount*sizeof(float)); cudaMemset(jacobianTDev[0],0,6*pixelSelectionAmount*sizeof(float));
        cudaMalloc((void **)&jacobianTDev[1], 6*pixelSelectionAmount*sizeof(float)); cudaMemset(jacobianTDev[1],0,6*pixelSelectionAmount*sizeof(float));
        cudaMalloc((void **)&jacobianTDev[2], 6*pixelSelectionAmount*sizeof(float)); cudaMemset(jacobianTDev[2],0,6*pixelSelectionAmount*sizeof(float));
    }
}

void TrackFrame::precomputeJacobian(VertexBuffer2 *vbufferExt) {
    precomputeJacobianCuda(vbufferExt,calibDataDevEXT,jacobianTDev[0],jacobianTDev[1],jacobianTDev[2],optScaleIn);
}

void TrackFrame::precomputeJacobian() {
    precomputeJacobianCuda(&vbuffer,calibDataDevEXT,jacobianTDev[0],jacobianTDev[1],jacobianTDev[2],optScaleIn);
}

void TrackFrame::lock() {
        vbuffer.lockIndex(); vbuffer.lock(); if (rgbImage.data != NULL) rgbImage.lock(); if (grayPyramid.getImageRef(0).data != NULL) grayPyramid.lock();
}

void TrackFrame::unlock() {
    vbuffer.unlockIndex();
    vbuffer.unlock();
    if (rgbImage.data != NULL) rgbImage.unlock();
    if (grayPyramid.getImageRef(0).data != NULL) grayPyramid.unlock();
}


void TrackFrame::setIterationCounts(int *nIter) {
    for (int i = 0; i < 3; i++) nIterations[i] = nIter[i];
}


// for selected points:
    // parallel: warp points + interpolate -> residual
    // 1thread: m-estimator weights
    // parallel: Jpre -> reweight jacobian -> Jw
    // parallel: Jw^T * Jw, Jw^T * residual (system reduction)
    // 1thread: inv6x6, solve x
    // 1thread: expm -> Trel update

void TrackFrame::optimizePose(ImagePyramid2 &grayPyramid, VertexBuffer2 *vbufferCur, bool filterDepthFlag)
{
    // assume gray pyramid is locked already from previous update!
    int nLayers = grayPyramid.nLayers;

    if (vbufferCur == NULL || vbufferCur->devPtr == NULL) {
        printf("vbuffer is not locked!\n");
        return;
    }

   #if defined(PERFORMANCE_TEST)
        float delay = 0.0f;
        float delays[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        static float delaysCumulated[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        static int numExec = 0;
        PerformanceCounter timer;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        timer.StartCounter();
   #endif

    vbuffer.setStream(0);

    #if defined(PERFORMANCE_TEST)
          cudaThreadSynchronize(); timer.StopCounter(); delay = timer.GetElapsedTime()*1000.0f; delays[0] += delay; cudaEventRecord(start,0);
    #endif

    int firstLayer = 2;
    int finalLayer = 0;
    for (int i = firstLayer; i >= finalLayer; i--) {
        for (int j = 0; j < nIterations[i]; j++) {
            interpolateResidual(&vbuffer,vbufferCur,TrelDev,calibDataDevEXT,grayPyramid,i,residualDev,zWeightsDev);
            #if defined(PERFORMANCE_TEST)
                cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[1] += delay; cudaEventRecord(start,0);
            #endif
            generateWeights64(residualDev,vbuffer.getElementsCount(),weightsDev,zWeightsDev,weightedResidualDev);
            weightJacobian(jacobianTDev[i], weightsDev, vbuffer.getElementsCount(), weightedJacobianTDev);
            #if defined(PERFORMANCE_TEST)
                cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[2] += delay; cudaEventRecord(start,0);
            #endif
            JTJCuda(weightedJacobianTDev,vbuffer.getElementsCount(),JtJDev);
            JTresidualCuda(weightedJacobianTDev,weightedResidualDev,vbuffer.getElementsCount(),residual6Dev);
            #if defined(PERFORMANCE_TEST)
            cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[3] += delay; cudaEventRecord(start,0);
            #endif
            solveMotionCuda(JtJDev,residual6Dev,TrelDev,optScaleOut);
            #if defined(PERFORMANCE_TEST)
                cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[4] += delay; cudaEventRecord(start,0);
            #endif
        }
    }

//    if (filterDepthFlag) filterDepthIIR(&vbuffer, vbufferCur, TrelDev, calibDataDevEXT, zWeightsDev, grayPyramid.getImageRef(0).width, grayPyramid.getImageRef(0).height, 0.9f);
    matrixMult4Cuda(TrelDev,TabsDev,TnextDev);
    #if defined(PERFORMANCE_TEST)
        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[5] += delay;
    #endif

    #if defined(PERFORMANCE_TEST)
        for (int i = 0; i < 5; i++)
            delaysCumulated[i] = (delays[i] + numExec*delaysCumulated[i])/float(1+numExec);
        printf("baselock: %3.1f, interp: %3.1f, mest: %3.1f, gaussnewton: %3.1f, solveMotion: %3.1f, mtx: %3.1f\n",delaysCumulated[0],delaysCumulated[1],delaysCumulated[2],delaysCumulated[3],delaysCumulated[4],delaysCumulated[5]);
        numExec++;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#endif

}

// TODO:
// baseInterpolation luokka
// input : TnextDevit keyframeista
// output: filtteröity pose Tf
// 1) Tf suhteessa lähimpään keyframeen -> TfrelDev (warpPoints näyttää järkevää)
// 2) updateBase(Tf,TfrelDev) toiminto interpolateBase-luokkaan keyframen sijaan?

void TrackFrame::updateBase(BaseBuffer2 &baseBuffer, ImagePyramid2 &grayPyramid) {
    VertexBuffer2 *baseBufferEXT = baseBuffer.getVBuffer();
    baseBufferEXT->lock();
    warpPoints(&vbuffer,weightsDev,TrelDev,calibDataDevEXT,baseBufferEXT,&grayPyramid);
    warpBase(baseBufferEXT,TnextDev);
    baseBuffer.downloadBaseCPU(TnextDev);
    //cudaThreadSynchronize();
    baseBufferEXT->unlock();
}

void TrackFrame::release() {
    rgbImage.releaseData();
    grayPyramid.releaseData();
    vbuffer.release();
    //rgbImageCPU.release();
    depthCPU.release();
    depthRGB.release();
    grayImageHDR.release();
    grayImage.release();
    baseMask.release();
    pixelSelectionMask.release();
    normalImage.release();
    xyzImage.release();
    cudaFree(TrelDev);
    cudaFree(TabsDev);
    cudaFree(TtmpDev);
    cudaFree(TidentityDev);
    cudaFree(TnextDev);
    if (jacobianTDev[0] != NULL) cudaFree(jacobianTDev[0]);
    if (jacobianTDev[1] != NULL) cudaFree(jacobianTDev[1]);
    if (jacobianTDev[2] != NULL) cudaFree(jacobianTDev[2]);
    keyFrameRefCount--;
    if (keyFrameRefCount == 0) {
        releaseCudaDotProduct();
        cudaFree(residualDev);
        cudaFree(weightedResidualDev);
        cudaFree(residual6Dev);
        cudaFree(weightsDev);
        cudaFree(zWeightsDev);
        cudaFree(weightedJacobianTDev);
        cudaFree(JtJDev);
        cudaFree(selectionMaskDev);
//        cudaFree(histogramDev);
//        cudaFree(partialHistogramsDev);
        cudaFree(histogramFloatDev);
        cudaFree(d_hist);
        cudaFree(gradientScratchDev);
        closeHistogram64();
        printf("released keyframe scratch memory!\n"); fflush(stdin); fflush(stdout);
    }
    imageDescriptors.release();
}

void swapElements(std::vector<KeyPoint> &points, int indexA, int indexB) {
    KeyPoint pA = points[indexA];
    KeyPoint pB = points[indexB];
    points[indexA] = pB;
    points[indexB] = pA;
}

void swapElements(std::vector<int> &indices, int indexA, int indexB) {
    int pA = indices[indexA];
    int pB = indices[indexB];
    indices[indexA] = pB;
    indices[indexB] = pA;
}

/*
void TrackFrame::initializeFeatures(int nFeatures, int keyIndex, Calibration *calib) {
    std::vector<KeyPoint> keyPointsAll; keyPointsAll.reserve(10000);
    keyPoints.clear();
    cv::Mat overlayImage = rgbImageCPU.clone();
    unsigned char *ptr = overlayImage.ptr();
    //unsigned char *select = pixelSelectionMask.ptr();
    //int offset = 0;
    //for (int j = 0; j < grayImage.rows; j++) {
    //    for (int i = 0; i < grayImage.cols; i++,offset++) {
    //        if (select[offset] > 0) {
    //            KeyPoint keyPoint(float(i),float(j),15.0f);
    //            keyPointsAll.push_back(keyPoint);
    //        }
    //    }
    //}

    //Mat dst, dst_norm, dst_norm_scaled;
    //dst = Mat::zeros( grayImage.size(), CV_32FC1 );

    ///// Detector parameters
    //int blockSize = 2;
    //int apertureSize = 3;
    //double k = 0.04;

    ///// Detecting corners
    //cornerHarris( grayImage, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    ///// Normalizing
    //normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    //convertScaleAbs( dst_norm, dst_norm_scaled );
    //char buf2[512];
    //sprintf(buf2,"scratch/harris%04d.ppm",keyIndex);
    //imwrite(buf2,dst_norm_scaled);

      //Detect the keypoints using SIFT Detector
      //int minHessian = 400;
     // SurfFeatureDetector
      SiftFeatureDetector detector;//( minHessian );
      detector.detect( grayImage, keyPointsAll);//, pixelSelectionMask);
      printf("keypoint size: %d\n",int(keyPointsAll[0].size));

    // initialize SiftDetector : no edge thresholding due to selection mask!
    //SIFT sift;
    Mat imageDescriptorsAll; int featureLength;
    //sift(grayImage,pixelSelectionMask,keyPointsAll,imageDescriptorsAll,true); featureLength = imageDescriptorsAll.cols;
    SiftDescriptorExtractor extractor;
    extractor.compute( grayImage, keyPointsAll, imageDescriptorsAll); featureLength = imageDescriptorsAll.cols;
//
    // int threshold = 16;
    // FAST(keyFrame->grayImage, keyFrame->keyPoints, threshold);
    // initialize block counts
    std::map<int,int> blockPointsRemaining;
    for (int i = 0; i < 16*16; i++) blockPointsRemaining[i] = 0;

    // initialize feature blocks
    std::map<int,std::vector<int> > featureBlocks;
    for (int i = 0; i < keyPointsAll.size(); i++) {
        KeyPoint &p = keyPointsAll[i];
        int iX = int(16.0f*p.pt.x/float(grayImage.cols));
        int iY = int(16.0f*p.pt.y/float(grayImage.rows));
        int bi = iX + iY*16;
        featureBlocks[bi].push_back(i);
        blockPointsRemaining[bi] = featureBlocks[bi].size();
    }
    // select 100 points uniformly
    imageDescriptors = cv::Mat(nFeatures,featureLength,imageDescriptorsAll.type());
    int nSelected = 0;
    if (keyPointsAll.size() < nFeatures) nFeatures = keyPointsAll.size();
    int bi = -1;
    while (nSelected < nFeatures ) {
        bi = (bi+1)%(16*16);
        int nPoints = blockPointsRemaining[bi];
        if (nPoints == 0) continue;
        int pi = rand()%nPoints;
        imageDescriptors.row(nSelected) = imageDescriptorsAll.row(featureBlocks[bi][pi]);
        KeyPoint &p = keyPointsAll[featureBlocks[bi][pi]];
        keyPoints.push_back(p); nPoints--;
        swapElements(featureBlocks[bi],pi,nPoints);
        blockPointsRemaining[bi] = nPoints;
        nSelected++;
    }

    printf("keyframe %d, keypoints %d, descriptor length %d\n",keyIndex,(int)keyPoints.size(),featureLength);
    for (int i = 0; i < keyPoints.size(); i++) {
        KeyPoint &p = keyPoints[i];
        cv::circle(overlayImage,p.pt,1,cvScalar(0,255,0));
    }
    char buf[512];
    sprintf(buf,"scratch/overlay%04d.ppm",keyIndex);
    imwrite(buf,overlayImage);
}*/

int findNearestSIFT(unsigned char *descr, cv::Mat &imageDescriptorsAll) {
    int nCandidates = imageDescriptorsAll.rows;
    int featureLen = imageDescriptorsAll.cols;
    unsigned char *candidates = (unsigned char*)imageDescriptorsAll.ptr();

    int nearestFeature = 0;
    double distance = FLT_MAX;
    for (int i = 0; i < nCandidates; i++) {
        double dist = 0.0f;
        for (int j = 0; j < featureLen; j++) {
            dist += (candidates[i*featureLen+j]-descr[j])*(candidates[i*featureLen+j]-descr[j]);
        }
        if (dist < distance) {
            distance = dist;
            nearestFeature = i;
        }
    }
    return nearestFeature;
}

/*
void TrackFrame::searchFeatures(TrackFrame *curFrame, int keyIndex, Calibration *calib, int searchRegion, float depthTolerance) {
    calib->setupCalibDataBuffer(320,240);
    float *zmap = (float*)depthRGB.ptr();
    float *curZMap = (float*)curFrame->depthRGB.ptr();
    float v[3] = {0,0,0};
    float pc[3] = {0,0,0};
    float P[16],Tz[4],refT[16],curT[16];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float iKR[9]; inverse3x3(KR,&iKR[0]);
    // opengl->standard, current RGB -> world
    transpose4x4(&Tbase[0],&refT[0]);
    transpose4x4(&curFrame->Tbase[0],&curT[0]);
    projectInitZ(refT, KR, curT, P, Tz);

    std::vector<KeyPoint> candidates; candidates.reserve(searchRegion*searchRegion);

    //unsigned char *descriptors = (unsigned char*)imageDescriptors.ptr();

    cv::Mat overlayImage = curFrame->rgbImageCPU.clone();

    for (size_t i = 0; i < keyPoints.size(); i++) {
        KeyPoint &p = keyPoints[i];
        //unsigned char *descr = &descriptors[i*imageDescriptors.cols];
        int offset = int(p.pt.x) + int(p.pt.y)*grayImage.cols;
        float z = zmap[offset];
        get3DPoint(p.pt.x,p.pt.y, z, iKR, &v[0], &v[1], &v[2]);
        projectFastZ(v,pc,&z,P,Tz);
        if (pc[0] < 0.0f || pc[1] < 0.0f) continue;
        if (pc[0] > grayImage.cols-1 || pc[1] > grayImage.rows-1) continue;
        int curOffset = int(pc[0])+int(pc[1])*grayImage.cols;
        if (fabs((-z)-curZMap[curOffset]) > depthTolerance) { cv::circle(overlayImage,cvPoint(pc[0],pc[1]),1,cvScalar(0,0,255)); continue;}

        cv::circle(overlayImage,cvPoint(pc[0],pc[1]),1,cvScalar(0,0,0));

        Rect rect; rect.x = pc[0]-searchRegion/2; rect.y = pc[1]-searchRegion/2;
        rect.width = searchRegion; rect.height = searchRegion;
        cv::rectangle(overlayImage,rect,cvScalar(0,255,0),1);

        // enumerate search candidates
        candidates.clear();
        for (int v = rect.y; v <= rect.y+rect.height; v++) {
            for (int u = rect.x; u <= rect.x+rect.width; u++) {
                KeyPoint keyPoint(float(u),float(v),p.size);
                candidates.push_back(keyPoint);
            }
        }

       // SIFT sift;
        Mat candidateDescriptors; int featureLength;
        cv::Mat currentDescriptor = imageDescriptors.row(i);
        //sift(curFrame->grayImage,curFrame->pixelSelectionMask,candidates,candidateDescriptors,true); featureLength = candidateDescriptors.cols;

        //Calculate descriptors (feature vectors)
        SiftDescriptorExtractor extractor;

        extractor.compute( curFrame->grayImage, candidates, candidateDescriptors);
 //       extractor.compute( img_2, keypoints_2, descriptors_2 );

        //Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( currentDescriptor, candidateDescriptors, matches );
        float dist = FLT_MAX; int nearestIndex = 0;
        for (size_t k = 0; k < matches.size(); k++) {
            DMatch &match = matches[k];
            if (match.distance < dist) {
                dist = match.distance;
                nearestIndex = match.trainIdx;
            }
        }

        int featureID = nearestIndex;//findNearestSIFT(descr,candidateDescriptors);
        KeyPoint &found = candidates[featureID];
        cv::circle(overlayImage,found.pt,1,cvScalar(0,255,255));
        cv::line(overlayImage,cvPoint(pc[0],pc[1]),found.pt,cvScalar(255,255,255),1);
    }
    char buf[512];
    sprintf(buf,"scratch/overlay%04d.ppm",keyIndex);
    imwrite(buf,overlayImage);
}*/


void getSearchBound(TrackFrame *key0, TrackFrame *key1, float x, float y, float searchBoundZ, Calibration &calib, float *p1, float *p2) {
    float T0[16],T1[16];
    transpose4x4(&key0->T[0],&T0[0]);
    transpose4x4(&key1->T[0],&T1[0]);

    calib.setupCalibDataBuffer(320,240);
    float *K = &calib.getCalibData()[KR_OFFSET];
    float iK[9],P[16],Tz[4];
    inverse3x3(K,&iK[0]);

    float p3d[3],p3d_1[3],p3d_2[3];
    key0->getScenePoint(int(x),int(y), calib, &p3d[0],&p3d[1],&p3d[2]);
    float dir[3];
    dir[0] = p3d[0]-T0[3];
    dir[1] = p3d[1]-T0[7];
    dir[2] = p3d[2]-T0[11];
    normalize(dir);

    p3d_1[0] = p3d[0] - searchBoundZ*dir[0];
    p3d_1[1] = p3d[1] - searchBoundZ*dir[1];
    p3d_1[2] = p3d[2] - searchBoundZ*dir[2];

    p3d_2[0] = p3d[0] + searchBoundZ*dir[0];
    p3d_2[1] = p3d[1] + searchBoundZ*dir[1];
    p3d_2[2] = p3d[2] + searchBoundZ*dir[2];

    projectInitZ(T0,K,T1,P,Tz);
    projectFast(p3d_1,p1,P);
    projectFast(p3d_2,p2,P);
}

void getSearchBound(TrackFrame *key0, TrackFrame *key1, float *p3d, float searchBoundZ, Calibration &calib, float *p1, float *p2) {
    float T0[16],T1[16];
    transpose4x4(&key0->T[0],&T0[0]);
    transpose4x4(&key1->T[0],&T1[0]);

    calib.setupCalibDataBuffer(320,240);
    float *K = &calib.getCalibData()[KR_OFFSET];

    float dir[3];
    dir[0] = p3d[0]-T0[3];
    dir[1] = p3d[1]-T0[7];
    dir[2] = p3d[2]-T0[11];
    normalize(dir);

    float p3d_1[3],p3d_2[3];

    p3d_1[0] = p3d[0] - searchBoundZ*dir[0];
    p3d_1[1] = p3d[1] - searchBoundZ*dir[1];
    p3d_1[2] = p3d[2] - searchBoundZ*dir[2];

    p3d_2[0] = p3d[0] + searchBoundZ*dir[0];
    p3d_2[1] = p3d[1] + searchBoundZ*dir[1];
    p3d_2[2] = p3d[2] + searchBoundZ*dir[2];

    identity4x4(T0);
    float P[16],Tz[4];
    projectInitZ(T0,K,T1,P,Tz);
    projectFast(p3d_1,p1,P);
    projectFast(p3d_2,p2,P);
}


void getProjection(TrackFrame *key0, TrackFrame *key1, float *p3d, Calibration &calib, float *p ) {
    float T0[16],T1[16];
    transpose4x4(&key0->T[0],&T0[0]);
    transpose4x4(&key1->T[0],&T1[0]);

    calib.setupCalibDataBuffer(320,240);
    float *K = &calib.getCalibData()[KR_OFFSET];

    identity4x4(T0);
    float P[16],Tz[4];
    projectInitZ(T0,K,T1,P,Tz);
    projectFast(p3d,p,P);
}

void TrackFrame::gpuCopyKeyframe(int id, ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbufferExt, float *imDepthDevIR, int pixelSelectionAmount, VertexBuffer2 *vbufferTemp, int nMultiResolutionLayers, bool renderableVBuffer) {

    TrackFrame *kf = this;

    int width = frame1C.getImageRef(0).width;
    int height = frame1C.getImageRef(0).height;

    vbufferTemp->lock(); vbufferTemp->lockIndex();

    kf->vbuffer.init(pixelSelectionAmount,renderableVBuffer,COMPRESSED_STRIDE);

    // create gpu images at 320x240
    //createHdrImage(NULL,width,height,3,&kf->rgbImage,ONLY_GPU_TEXTURE, false);
    //kf->grayPyramid.createHdrPyramid(width,height,1,nMultiResolutionLayers,false,ONLY_GPU_TEXTURE,renderableVBuffer); kf->grayPyramid.setName("grayPyramid1C");

    kf->id = id;
    kf->lock();

    vbufferExt->copyTo(*vbufferTemp); //5kf->rgbImage.updateTextureInternal(frame3C.devPtr,false);
    /*for (int i = 0; i < kf->grayPyramid.nLayers; i++) {
        kf->grayPyramid.getImageRef(i).updateTextureInternal(frame1C.getImageRef(i).devPtr,false);
    }*/

    Image2 imDepth(imDepthDevIR,frame3C.width,frame3C.height,frame3C.width*sizeof(float),1,true);
    kf->selectPixelsGPUCompressedLock(*vbufferTemp,pixelSelectionAmount,&imDepth,&frame1C);
    kf->setBaseTransform(kf->getNextBaseDev());
    vbufferTemp->unlockIndex(); vbufferTemp->unlock();
    kf->unlock();
}

void TrackFrame::extractFeatures() {
/*
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( grayImageHDR.size(), CV_32FC1 );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 100;
    int max_thresh = 255;

    /// Detecting corners
    cornerHarris( grayImageHDR, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ ) {
        for( int i = 0; i < dst_norm.cols; i++ ) {
            if( (int) dst_norm.at<float>(j,i) > thresh ) {
                circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
    char buf[512]; static int joo = 0;
    sprintf(buf,"scratch/harris%04d.ppm",joo); joo++;
    imwrite(buf,dst_norm_scaled);*/
}
