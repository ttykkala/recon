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


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <multicore/multicore.h>
#include <omp.h>
#include <assert.h>
#include <tracker/basic_math.h>
#include <tracker/eigenMath.h>
#include <iostream>
//#include <image2/Image2.h>

using namespace cv;
static OMPFunctions multicore;

OMPFunctions::OMPFunctions() {
    init();
}

void OMPFunctions::init()
{
    /* get the total number of CPUs/cores available for OpenMP */
    NCPU = omp_get_num_procs();
    /* get the total number of threads requested */
    NTHR = omp_get_max_threads();

    printf("OpenMP cores: %i, threads: %i\n",NCPU,NTHR);

#pragma omp parallel
  {
        /* get the current thread ID in the parallel region */
        int tid = omp_get_thread_num();
        /* get the total number of threads available in this parallel region */
        int NPR = omp_get_num_threads();
       // printf("Hyper thread %i/%i says hello!\n",tid,NPR);
  }
    dxTable = new float[640*480*2];
    mappingPrecomputed = false;
}


OMPFunctions::~OMPFunctions() {
    delete[] dxTable;
}

OMPFunctions *getMultiCoreDevice() {
    return &multicore;
}

void OMPFunctions::convert2Float(Mat &dispImage, Mat &hdrImage) {
    int width  = dispImage.cols;
    int height = dispImage.rows;

    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *hdrPtr = (float*)hdrImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                hdrPtr[offset] = (float)dPtr[offset];
            }
        }
    }
}

void OMPFunctions::d2ZLowHdr(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;

    float *srcPtr = (float*)dispImage.ptr();
    float *dstPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;
                if (sxi < (width2-1) && syi < (height2-1)) {
                    int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                    int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                    int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                    int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                    float d1 = (float)srcPtr[srcIdx1];
                    float d2 = (float)srcPtr[srcIdx2];
                    float d3 = (float)srcPtr[srcIdx3];
                    float d4 = (float)srcPtr[srcIdx4];
                    // prefer points close to sensor
                    float d = d1;
                    if (d2 < d) d = d2;
                    if (d3 < d) d = d3;
                    if (d4 < d) d = d4;

                    float z = 0.0f;
                    //if (d < 2047)
                    {
                        z = fabs(1.0f/(c0+c1*d));
                        //float z = fabs(8.0f*b*fx/(B-d));
                        if (z > maxDist || z < minDist) z = 0.0f;
                    }
                    dstPtr[offset] = z;
                    continue;
                }
                dstPtr[offset] = 0.0f;
            }
        }
    }
}


 void OMPFunctions::undistortDisparityMap(Mat &dispImage, Mat &uDispImage, float alpha0, float alpha1, float *beta) {
     int width  = dispImage.cols;
     int height = dispImage.rows;

     unsigned short *dPtr = (unsigned short*)dispImage.ptr();
     float *hdrPtr = (float*)uDispImage.ptr();

     int nBlocks = 4;
     int blockSize = height/nBlocks;
     int blockID = 0;
     #pragma omp parallel for private(blockID)
     for (blockID = 0; blockID < nBlocks; blockID++) {
         int offset = blockID*blockSize*width;
         for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
             for (int xi = 0; xi < width; xi++,offset++) {
                 float d = (float)dPtr[offset];
                 if (beta[offset]>0.0f) {
                    hdrPtr[offset] = d + beta[offset]*expf(alpha0-alpha1*d);
                    //return disp + dc_beta(v,u)*std::expf(dc_alpha[0] - dc_alpha[1]*disp);
                 } else {
                    hdrPtr[offset] = d;
                 }
             }
         }
     }
 }

 void OMPFunctions::z2Pts(Mat &depthMap, float *K, float *pts3) {
     int width  = depthMap.cols;
     int height = depthMap.rows;

     float *zPtr = (float*)depthMap.ptr();
     float iK[9]; inverse3x3(K,&iK[0]);

     int nBlocks = 4;
     int blockSize = height/nBlocks;
     int blockID = 0;
     #pragma omp parallel for private(blockID)
     for (blockID = 0; blockID < nBlocks; blockID++) {
         int offset = blockID*blockSize*width;
         int offset3 = offset*3;
         for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
             for (float xi = 0; xi < width; xi++,offset++,offset3+=3) {
                 get3DPoint(xi,yi,zPtr[offset],iK,&pts3[offset3+0],&pts3[offset3+1],&pts3[offset3+2]);
             }
         }
     }
 }


void OMPFunctions::d2ZLow(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;


    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *zPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;

                if (sxi < (width2-1) && syi < (height2-1)) {
                        int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                        int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                        int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                        int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                        float d1 = (float)dPtr[srcIdx1];
                        float d2 = (float)dPtr[srcIdx2];
                        float d3 = (float)dPtr[srcIdx3];
                        float d4 = (float)dPtr[srcIdx4];
                        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
                                float d = d1;
                                if (d2 < d) d = d2;
                                if (d3 < d) d = d3;
                                if (d4 < d) d = d4;
                                float z = fabs(1.0f/(c0+c1*d));
                                //float z = fabs(8.0f*b*fx/(B-d));
                                if (z > maxDist || z < minDist) z = 0.0f;
                                zPtr[offset] = z;//(z-minDist)/(maxDist-minDist);
                                continue;
                        }
                }
                zPtr[offset] = 0.0f;
            }
        }
    }
}


void OMPFunctions::d2Z(Mat &dispImage, Mat &depthImage, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImage.cols;
    int height = depthImage.rows;

    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *zPtr          = (float*)depthImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = xi + xOff;
                unsigned int syi = yi + yOff;

                if (sxi < (width-1) && syi < (height-1)) {
                    int srcIdx = sxi + syi * width;
                    float d = (float)dPtr[srcIdx];
                    if (d < 2047) {
                        float z = fabs(1.0f/(c0+c1*d));
                        //float z = fabs(8.0f*b*fx/(B-d));
                        if (z > maxDist || z < minDist) z = 0.0f;
                        zPtr[offset] = z;//(z-minDist)/(maxDist-minDist);
                        continue;
                    }
                }
                zPtr[offset] = 0.0f;
            }
        }
    }
}

void OMPFunctions::d2ZHdr(Mat &dispImage, Mat &depthImage, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
	int width  = depthImage.cols;
	int height = depthImage.rows;

	float *dPtr = (float*)dispImage.ptr();
	float *zPtr          = (float*)depthImage.ptr();

	int nBlocks = 4;
	int blockSize = height/nBlocks;
	int blockID = 0;
#pragma omp parallel for private(blockID)
	for (blockID = 0; blockID < nBlocks; blockID++) {
		int offset = blockID*blockSize*width;
		for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
			for (int xi = 0; xi < width; xi++,offset++) {
				// IR image -> disparity image has constant offset (Konolige's tech guide)
				// http://www.ros.org/wiki/kinect_calibration/technical
				int sxi = xi + xOff;
				int syi = yi + yOff;
				if (sxi >= 0 && sxi < (width-1) && syi >= 0 && syi < (height-1)) {
					int srcIdx = sxi + syi * width;
					float d = (float)dPtr[srcIdx];
					if (d < 2047) {
						float z = fabs(1.0f/(c0+c1*d));
						//float z = fabs(8.0f*b*fx/(B-d));
						if (z > maxDist || z < minDist) z = 0.0f;
						zPtr[offset] = z;//(z-minDist)/(maxDist-minDist);
						continue;
					}
				}
				zPtr[offset] = 0.0f;
			}
		}
	}
}

void OMPFunctions::replaceUShortRange(Mat &dispImage, unsigned short valueStart, unsigned short valueEnd, unsigned short newValue) {
    int width  = dispImage.cols;
    int height = dispImage.rows;

    unsigned short *dPtr = (unsigned short*)dispImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                if (dPtr[offset] >= valueStart && dPtr[offset] <= valueEnd) dPtr[offset] = newValue;
            }
        }
    }
}


void OMPFunctions::d2ZLowGPU(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;


    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *zPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;

                if (sxi < (width2-1) && syi < (height2-1)) {
                        int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                        int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                        int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                        int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                        float d1 = (float)dPtr[srcIdx1];
                        float d2 = (float)dPtr[srcIdx2];
                        float d3 = (float)dPtr[srcIdx3];
                        float d4 = (float)dPtr[srcIdx4];
                        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
                                float d = d1;
                                if (d2 < d) d = d2;
                                if (d3 < d) d = d3;
                                if (d4 < d) d = d4;
                                 float z = fabs(1.0f/(c0+c1*d));
                                //float z = fabs(8.0f*b*fx/(B-d));
                                if (z > maxDist || z < minDist) z = 0.0f;
                                zPtr[offset] = (z-minDist)/(maxDist-minDist);
                                continue;
                        }
                }
                zPtr[offset] = 0.0f;
            }
        }
    }
}


void OMPFunctions::downSampleDepth(Mat &depthImage, Mat &depthImageSmall) {
    int width = depthImageSmall.cols;
    int height = depthImageSmall.rows;

    float *dstPtr = (float*)depthImageSmall.ptr();
    float *srcPtr = (float*)depthImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
               int offset2 = i*2+j*2*width*2;
               float z1 = srcPtr[offset2];
               float z2 = srcPtr[offset2+1];
               float z3 = srcPtr[offset2+width*2];
               float z4 = srcPtr[offset2+width*2+1];
               // TODO: bilateraalisuodatus?
               dstPtr[offset] = MIN(MIN(MIN(z1,z2),z3),z4);
            }
        }
    }
}


void OMPFunctions::baselineTransform(Mat &depthImageL,Mat &depthImageR,float *KL, float *TLR, float *KR) {
    int width  = depthImageL.cols;
    int height = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    float *zptrDst = (float*)depthImageR.ptr();
    for (int i = 0; i < width*height; i++) zptrDst[i] = FLT_MAX;

    float fx = KL[0];
    float fy = KL[4];
    float cx = KL[2];
    float cy = KL[5];

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float z = zptrSrc[offset];
                float p3[3],r3[3],p2[3];
                p3[0] = -(float(i) - cx) * z / fx;
                p3[1] = -(float(j) - cy) * z / fy;
                p3[2] = -z;
                transformRT3(TLR, p3, r3);
                matrixMultVec3(KR, r3, p2); p2[0] /= p2[2]; p2[1] /= p2[2];
                int xi = (int)p2[0];
                int yi = (int)p2[1];

               if (xi >= 0 && yi >= 0 && xi < width && yi < height) {
                    int offset = xi + yi * width;
                    float prevZ =  zptrDst[offset];
                    float newZ  = fabs(r3[2]);
                    if (newZ > 0 && newZ < prevZ) zptrDst[offset] = newZ;
                }
            }
        }
    }
    for (int i = 0; i < width*height; i++) if (zptrDst[i] == FLT_MAX) zptrDst[i] = 0.0f;
    return;
}

void OMPFunctions::baselineWarp(Mat &depthImageL,Mat &grayImageR,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet) {
    int width  = depthImageL.cols;
    int height = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    unsigned char *grayPtrDst = (unsigned char*)grayImageR.ptr();

    float fx = KL[0];
    float fy = KL[4];
    float cx = KL[2];
    float cy = KL[5];

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float z = zptrSrc[offset];
                float p3[3],r3[3],r3n[3],p2[3],p2n[3];
                p3[0] = -(float(i) - cx) * z / fx;
                p3[1] = -(float(j) - cy) * z / fy;
                p3[2] = -z;
                transformRT3(TLR, p3, r3); r3n[0] = r3[0]/r3[2]; r3n[1] = r3[1]/r3[2]; r3n[2] = 1.0f;
                // p2: distorted point
                distortPointCPU(r3n,kc,KR,p2);
                // p2n: undistorted point
                //matrixMultVec3(KR, r3n, p2n);
                int xi = (int)p2[0];
                int yi = (int)p2[1];
				fullPointSet[offset].rx2      = p2[0];
				fullPointSet[offset].ry2      = p2[1];
				fullPointSet[offset].px      = p3[0];
				fullPointSet[offset].py      = p3[1];
				fullPointSet[offset].pz      = p3[2];
                fullPointSet[offset].magGrad = -1;
                if (xi > 0 && yi > 0 && xi < (width-1) && yi < (height-1)) {
                    int offset2 = xi + yi * width;
                    unsigned char gx0 =  grayPtrDst[offset2-1];
                    unsigned char gx1 =  grayPtrDst[offset2+1];
                    unsigned char gy0 =  grayPtrDst[offset2-width];
                    unsigned char gy1 =  grayPtrDst[offset2+width];
                    unsigned char c   =  grayPtrDst[offset2];
                    int dx = gx1-gx0;
                    int dy = gy1-gy0;
                    fullPointSet[offset].magGrad  = (abs(dx) + abs(dy))/2;
                    fullPointSet[offset].color = float(c)/255.0f;
                }
            }
        }
    }
}
/*
void genProjectionMtx(float *Kin, float texWidth, float texHeight, float nearZ, float farZ, float *P) {
	P[0]  = -2*Kin[0]/texWidth;  P[1]  = 0;                   P[2]  = 1-(2*Kin[2]/texWidth);        P[3] = 0;
	P[4]  = 0.0f;                P[5]  = 2*Kin[4]/texHeight;  P[6]  = -1+(2*Kin[5]+2)/texHeight;    P[7] = 0;
	P[8]  = 0.0f;                P[9]  = 0.0f;                P[10] = (farZ+nearZ)/(nearZ-farZ);    P[11] = 2*nearZ*farZ/(nearZ-farZ);
	P[12] = 0.0f;                P[13] = 0.0f;                P[14] = -1;                           P[15] = 0;
}*/


void OMPFunctions::baselineWarpRGB(Mat &depthImageL,Mat &rgbImageR,Eigen::Matrix3f &KL, Eigen::Matrix4f &TLR, Eigen::Matrix3f &KR, float *kc, ProjectData *fullPointSet) {
    int width      = depthImageL.cols;
    int height     = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    unsigned char *rgbPtrDst = (unsigned char*)rgbImageR.ptr();

    float fx = KL(0,0);
    float fy = KL(1,1);
    float cx = KL(0,2);
    float cy = KL(1,2);

   // std::cout << KR << std::endl;
   // std::cout << KL << std::endl;
   // std::cout << TLR << std::endl;
   // for (int i = 0; i < 5; i++) printf("kc[%d]:%f\n",i,kc[i]);

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float z = zptrSrc[offset];
                Eigen::Vector4f p4;
                p4(0) = (float(i) - cx) * z / fx;
                p4(1) = (float(j) - cy) * z / fy;
                p4(2) = z;
                p4(3) = 1;
                Eigen::Vector4f r4 = TLR * p4;
                Eigen::Vector2f r2n(r4(0)/r4(2),r4(1)/r4(2));
                Eigen::Vector2f p2;
                radialDistort(r2n,kc,KR,p2);
                // p2n: undistorted point
                //matrixMultVec3(KR, r3n, p2n);
                int xi = (int)p2(0);
                int yi = (int)p2(1);
                fullPointSet[offset].rx2     = p2(0);
                fullPointSet[offset].ry2     = p2(1);
                fullPointSet[offset].rx      = r4(0);
                fullPointSet[offset].ry      = r4(1);
                fullPointSet[offset].rz      = r4(2);
                fullPointSet[offset].px      = p4(0);
                fullPointSet[offset].py      = p4(1);
                fullPointSet[offset].pz      = p4(2);
                fullPointSet[offset].magGrad = 0;
                //fullPointSet[offset].colorR = 1.0f;
                //fullPointSet[offset].colorG = 1.0f;
                //fullPointSet[offset].colorB = 1.0f;
                if (xi > 0 && yi > 0 && xi < (width-1) && yi < (height-1)) {
                    int offset2 = xi*3 + yi * width*3;
                    unsigned char cr =  rgbPtrDst[offset2+0];
                    unsigned char cg =  rgbPtrDst[offset2+1];
                    unsigned char cb =  rgbPtrDst[offset2+2];
                    fullPointSet[offset].colorR = float(cr)/255.0f;
                    fullPointSet[offset].colorG = float(cg)/255.0f;
                    fullPointSet[offset].colorB = float(cb)/255.0f;
                }
            }
        }
    }
}

void OMPFunctions::generateDepthMap(ProjectData *fullPointSet, Mat &depthImageR) {
    int width  = depthImageR.cols;
    int height = depthImageR.rows;
    int size = width*height;
    float *zptrDst = (float*)depthImageR.ptr();

    for (int i = 0; i < size; i++) zptrDst[i] = FLT_MAX;

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
				float px = fullPointSet[offset].px;
				float py = fullPointSet[offset].py;
				float z = -fullPointSet[offset].pz;
                int xi = (int)(px+0.5f);
                int yi = (int)(py+0.5f);
                if (xi >= 0 && yi >= 0 && xi <= (width-1) && yi <= (height-1)) {
                    int offset2 = xi + yi * width;
                    float cz = zptrDst[offset2];
                    if (z < cz) zptrDst[offset2] = z;
                }
            }
        }
    }

    float minVal = FLT_MAX;
    float maxVal = 0;
    for (int i = 0; i < size; i++) {
        if (zptrDst[i] < minVal) minVal = zptrDst[i];
        if (zptrDst[i] > maxVal) maxVal = zptrDst[i];
    }
    for (int i = 0; i < size; i++) {
        zptrDst[i] = 0.5f;//255.0f*(zptrDst[i]-minVal)/(maxVal-minVal);
    }
    //printf("jeap\n"); fflush(stdin); fflush(stdout);
}



void OMPFunctions::convert2Gray(Mat &rgbImage, Mat &grayImage) {
    assert(rgbImage.rows == grayImage.rows && rgbImage.cols == grayImage.cols);
    int nBlocks = 4;

    assert(NTHR >= nBlocks);

    int blockSize = rgbImage.cols*(rgbImage.rows/nBlocks);
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        unsigned char *srcPtr = rgbImage.ptr()  + blockID*blockSize*3;
        unsigned char *dstPtr = grayImage.ptr() + blockID*blockSize;

        for (int i = 0; i < blockSize; i++) {
            *dstPtr = (srcPtr[0]*19588 + srcPtr[1]*38469 + srcPtr[2]*7471)>>16;
            dstPtr++; srcPtr+=3;
        }
    }
}

void OMPFunctions::undistort(Mat &src, Mat &dst, float *K, float *iK, float *kc) {
    assert(src.ptr() != NULL && dst.ptr() != NULL && src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;

    // mark zero into upper-left corner
    unsigned char *ptr = src.ptr();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[src.cols] = 0;
    ptr[src.cols+1] = 0;

    if (mappingPrecomputed) { undistortLookup(src,dst); return; }

    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++)  {
   //     printf("blockID: %d\n",blockID);
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        unsigned char *dstPtr = dst.ptr();
        int y0 = blockID*blockHeight;
        float pu[2],pd[2],p[2],r2,r4,r6,radialDist;
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               pu[0] = float(xi); pu[1] = float(yi);
               // normalize point
               pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
               pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
               // define radial displacement
               r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); r4 = r2*r2; r6 = r4 * r2;
               radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
               pd[0] *= radialDist;
               pd[1] *= radialDist;
               // define sampling point in distorted image
               p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
               p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];

               int xdi = (int)p[0];
               int ydi = (int)p[1];
               if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
                       int srcOffset = xdi+ydi*width;
                       unsigned int fracX = (unsigned int)((p[0]-xdi)*256.0f);
                       unsigned int fracY = (unsigned int)((p[1]-ydi)*256.0f);

                       unsigned char *ptr = &srcPtr[srcOffset];
                       unsigned char i1 = ptr[0]; unsigned char i2 = ptr[1]; ptr += width;
                       unsigned char i4 = ptr[0]; unsigned char i3 = ptr[1];

                       const unsigned int c = fracX * fracY;
                       const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
                       const unsigned int b = (fracX<<8) - c;
                       const unsigned int d = 65536 - a - b - c;

                       dstPtr[offset] = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
                       dxTable[offset*2+0] = p[0]-xi;
                       dxTable[offset*2+1] = p[1]-yi;
               } else {
                       dstPtr[offset]  = 0;
                       dxTable[offset*2+0] = 0.0f-xi;
                       dxTable[offset*2+1] = 0.0f-yi;
               }
            }
        }
    }
    mappingPrecomputed = true;
}

void OMPFunctions::undistortLookup(Mat &src, Mat &dst)
{
    assert(src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;
    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        unsigned char *dstPtr = dst.ptr();
        int y0 = blockID*blockHeight;
        float p[2];
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               p[0] = dxTable[offset*2+0]+xi;
               p[1] = dxTable[offset*2+1]+yi;
               int xdi = int(p[0]);
               int ydi = int(p[1]);
               int srcOffset = xdi+ydi*width;
               unsigned int fracX = (unsigned int)((p[0]-xdi)*256.0f);
               unsigned int fracY = (unsigned int)((p[1]-ydi)*256.0f);
               unsigned char *ptr = &srcPtr[srcOffset];
               unsigned char i1 = ptr[0]; unsigned char i2 = ptr[1]; ptr += width;
               unsigned char i4 = ptr[0]; unsigned char i3 = ptr[1];

               const unsigned int c = fracX * fracY;
               const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
               const unsigned int b = (fracX<<8) - c;
               const unsigned int d = 65536 - a - b - c;

               dstPtr[offset] = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
            }
        }
    }
}

void OMPFunctions::undistortF(Mat &src, Mat &dst, float *K, float *iK, float *kc) {
    assert(src.ptr() != NULL && dst.ptr() != NULL && src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;

    // mark zero into upper-left corner
    unsigned char *ptr = src.ptr();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[src.cols] = 0;
    ptr[src.cols+1] = 0;

    if (mappingPrecomputed) { undistortLookupF(src,dst); return; }

    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
   //     printf("blockID: %d\n",blockID);
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        float *dstPtr = (float*)dst.ptr();
        int y0 = blockID*blockHeight;
        float pu[2],pd[2],p[2],r2,r4,r6,radialDist;
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               pu[0] = float(xi); pu[1] = float(yi);
               // normalize point
               pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
               pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
               // define radial displacement
               r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); r4 = r2*r2; r6 = r4 * r2;
               radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
               pd[0] *= radialDist;
               pd[1] *= radialDist;
               // define sampling point in distorted image
               p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
               p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];

               int xdi = (int)p[0];
               int ydi = (int)p[1];
               if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
                       int srcOffset = xdi+ydi*width;
                       float fracX = p[0]-xdi;
                       float fracY = p[1]-ydi;

                       unsigned char *ptr = &srcPtr[srcOffset];
                       float i1 = float(ptr[0]); float i2 = float(ptr[1]); ptr += width;
                       float i4 = float(ptr[0]); float i3 = float(ptr[1]);

                       const float c = fracX * fracY;
                       const float a = 1-fracY-fracX-c;
                       const float b = fracX-c;
                       const float d = 1-a-b-c;

                       dstPtr[offset] = a*i1 + b*i2 + c*i3 + d*i4;

                       dxTable[offset*2+0] = p[0]-xi;
                       dxTable[offset*2+1] = p[1]-yi;
               } else {
                       dstPtr[offset]  = 0.0f;
                       dxTable[offset*2+0] = 0.0f-xi;
                       dxTable[offset*2+1] = 0.0f-yi;
               }
            }
        }
    }
    mappingPrecomputed = true;
}

void OMPFunctions::undistortLookupF(Mat &src, Mat &dst)
{
    assert(src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;
    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize;
        unsigned char *srcPtr = (unsigned char*)src.ptr();
        float *dstPtr = (float*)dst.ptr();
        int y0 = blockID*blockHeight;
        float p[2];
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               p[0] = dxTable[offset*2+0]+xi;
               p[1] = dxTable[offset*2+1]+yi;
               int xdi = int(p[0]);
               int ydi = int(p[1]);
               int srcOffset = xdi+ydi*width;
               float fracX = p[0]-xdi;
               float fracY = p[1]-ydi;
               unsigned char *ptr = &srcPtr[srcOffset];
               float i1 = float(ptr[0]); float i2 = float(ptr[1]); ptr += width;
               float i4 = float(ptr[0]); float i3 = float(ptr[1]);

               const float c = fracX * fracY;
               const float a = 1-fracY-fracX-c;
               const float b = fracX-c;
               const float d = 1-a-b-c;

               dstPtr[offset] = a*i1 + b*i2 + c*i3 + d*i4;
            }
        }
    }
}

//rgbReference must be given as argument!

void OMPFunctions::downSamplePointCloud(cv::Mat &hiresXYZ, cv::Mat &lowresXYZ, int stride) {
    int width     = lowresXYZ.cols;   int height    = lowresXYZ.rows;
    int dstWidth  = hiresXYZ.cols;    //int dstHeight = hiresXYZ.rows;

    float *srcPts          = (float*)lowresXYZ.ptr();
    float *dstPts          = (float*)hiresXYZ.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        int offset3 = offset*3;
        int offsetp = offset*stride;
        int dwp = dstWidth*stride;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++,offset3+=3,offsetp+=stride) {
                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                int hiresOffsetP = hiresOffset*stride;

                float point[3] = {0,0,0};
                float normal[3] = {1,0,0};
                float maskValue = 0.0f;
                float grayIntensity = 0.0f;
                float gradMag = 0.0f;

                if ( (dstPts[hiresOffsetP+6] > 0)  && (dstPts[hiresOffsetP+stride+6] > 0) && (dstPts[hiresOffsetP+dwp+6] > 0) && (dstPts[hiresOffsetP+dwp+stride+6] > 0) ) {
                    point[0]   = (dstPts[hiresOffsetP+0] + dstPts[hiresOffsetP+stride+0] + dstPts[hiresOffsetP+0+dwp] + dstPts[hiresOffsetP+stride+0+dwp])/4.0f;
                    point[1]   = (dstPts[hiresOffsetP+1] + dstPts[hiresOffsetP+stride+1] + dstPts[hiresOffsetP+1+dwp] + dstPts[hiresOffsetP+stride+1+dwp])/4.0f;
                    point[2]   = (dstPts[hiresOffsetP+2] + dstPts[hiresOffsetP+stride+2] + dstPts[hiresOffsetP+2+dwp] + dstPts[hiresOffsetP+stride+2+dwp])/4.0f;
                    normal[0]  = (dstPts[hiresOffsetP+3] + dstPts[hiresOffsetP+stride+3] + dstPts[hiresOffsetP+3+dwp] + dstPts[hiresOffsetP+stride+3+dwp])/4.0f;
                    normal[1]  = (dstPts[hiresOffsetP+4] + dstPts[hiresOffsetP+stride+4] + dstPts[hiresOffsetP+4+dwp] + dstPts[hiresOffsetP+stride+4+dwp])/4.0f;
                    normal[2]  = (dstPts[hiresOffsetP+5] + dstPts[hiresOffsetP+stride+5] + dstPts[hiresOffsetP+5+dwp] + dstPts[hiresOffsetP+stride+5+dwp])/4.0f;
                    grayIntensity  = (dstPts[hiresOffsetP+7] + dstPts[hiresOffsetP+stride+7] + dstPts[hiresOffsetP+7+dwp] + dstPts[hiresOffsetP+stride+7+dwp])/4.0f;
                   // gradMag        = (dstPts[hiresOffsetP+8] + dstPts[hiresOffsetP+stride+8] + dstPts[hiresOffsetP+8+dwp] + dstPts[hiresOffsetP+stride+8+dwp])/4.0f;
                    // re-normalize
                    float len = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
                    normal[0] /= len; normal[1] /= len; normal[2] /= len;
                    maskValue = 255.0f;
                }
                srcPts[offsetp+0]     = point[0];
                srcPts[offsetp+1]     = point[1];
                srcPts[offsetp+2]     = point[2];
                srcPts[offsetp+3]     = normal[0];
                srcPts[offsetp+4]     = normal[1];
                srcPts[offsetp+5]     = normal[2];
                srcPts[offsetp+6]     = maskValue;
                srcPts[offsetp+7]     = gradMag;
                srcPts[offsetp+8]     = grayIntensity;
                srcPts[offsetp+9]     = 0;
                srcPts[offsetp+10]    = 0;

            }
        }
    }
}


void OMPFunctions::downSampleMask(cv::Mat &hiresMask, cv::Mat &lowresMask) {
    int width     = lowresMask.cols;   int height   = lowresMask.rows;
    int dstWidth  = hiresMask.cols;   int dstHeight = hiresMask.rows;

    unsigned char *srcMask = (unsigned char*)lowresMask.ptr();
    unsigned char *dstMask = (unsigned char*)hiresMask.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++) {

                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                int   maskValue = 0;
                if ( (dstMask[hiresOffset+0] > 0) && (dstMask[hiresOffset+1] > 0) && (dstMask[hiresOffset+0+dstWidth] > 0) && (dstMask[hiresOffset+1+dstWidth] > 0) ) {
                    maskValue = 255;
                }
                srcMask[offset]       = maskValue;
            }
        }
    }
}

void OMPFunctions::downSampleHdrImage(cv::Mat &hiresImage, cv::Mat &lowresImage) {
    int width     = lowresImage.cols;   int height   = lowresImage.rows;
    int dstWidth  = hiresImage.cols;

    float *src = (float*)lowresImage.ptr();
    float *dst = (float*)hiresImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++) {
                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                src[offset]       = (dst[hiresOffset+0] + dst[hiresOffset+1] + dst[hiresOffset+0+dstWidth] + dst[hiresOffset+1+dstWidth])/4.0f;
            }
        }
    }
}


void OMPFunctions::generateOrientedPoints(cv::Mat &depthCPU, cv::Mat &xyzImage, float *KL, cv::Mat &normalStatus, float *kc, float *KR, float *TLR, cv::Mat &grayImage, int stride)
{
    int fw = depthCPU.cols;
    int fh = depthCPU.rows;
    assert(fw == xyzImage.cols && fh == xyzImage.rows);

    float *srcData  = (float*)depthCPU.ptr();
    float *pData    = (float*)xyzImage.ptr();
    float *grayData = (float*)grayImage.ptr();
   // unsigned char *mask = (unsigned char*)maskImage.ptr();      memset(mask,0,fw*fh);
    unsigned char *status = (unsigned char*)normalStatus.ptr(); memset(status,0,fw*fh);

    float iKir[9]; inverse3x3(&KL[0],&iKir[0]);

    int nBlocks = 4;
    int blockSize = fh/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // if this is the last block, do not process the last row :
        int ydec = 0; if (blockID == nBlocks-1) ydec = 1;
        int yinc = 0; if (blockID == 0) yinc = 1;
        for (float yi = blockID*blockSize+yinc; yi < (blockID+1)*blockSize-ydec; yi++) {
            for (float xi = 1; xi < (fw-1); xi++) {
                int offset = xi+yi*fw;
                int offsetp = offset*stride;
                float z    = srcData[offset];
                float zNu1 = srcData[offset+1];
                float zNv1 = srcData[offset+fw];
                float zNu0 = srcData[offset-1];
                float zNv0 = srcData[offset-fw];

                // detect z-dynamics (are we at edge?)
                float minZ  = z; if (zNu1 < minZ) minZ = zNu1;  if (zNu0 < minZ) minZ = zNu0; if (zNv1 < minZ) minZ = zNv1; if (zNv0 < minZ) minZ = zNv0;
                float maxZ  = z; if (zNu1 > maxZ) maxZ = zNu1;  if (zNu0 > maxZ) maxZ = zNu0; if (zNv1 > maxZ) maxZ = zNv1; if (zNv0 > maxZ) maxZ = zNv0;
                float threshold = 250.0f;

                if (fabs(maxZ - minZ) < 100.0f && (minZ > threshold)) {
                    float p[3],u1[3],u0[3],v1[3],v0[3];
                    get3DPoint(float(xi),float(yi),z,iKir, &p[0], &p[1], &p[2]);

                    get3DPoint(float(xi+1),float(yi),zNu1,iKir, &u1[0], &u1[1], &u1[2]);
                    get3DPoint(float(xi-1),float(yi),zNu0,iKir, &u0[0], &u0[1], &u0[2]);
                    get3DPoint(float(xi),float(yi+1),zNv1,iKir, &v1[0], &v1[1], &v1[2]);
                    get3DPoint(float(xi),float(yi-1),zNv0,iKir, &v0[0], &v0[1], &v0[2]);

                    float nu[3],nv[3],n[3];
                    nu[0] = u1[0] - u0[0]; nu[1] = u1[1] - u0[1]; nu[2] = u1[2] - u0[2];
                    nv[0] = v1[0] - v0[0]; nv[1] = v1[1] - v0[1]; nv[2] = v1[2] - v0[2];
                    // compute normal as crossproduct
                    n[0] =  nu[1] * nv[2] - nu[2] * nv[1];
                    n[1] =-(nu[0] * nv[2] - nu[2] * nv[0]);
                    n[2] =  nu[0] * nv[1] - nu[1] * nv[0];
                    // normal to unit length
                    float len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]+1e-5f);
                    // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
                    n[0] /= len; n[1] /= len; n[2] /= len;

                    // compute area code for point selection
                    unsigned char statusIndex = 0; float maxDot = 0; float dot;
                    dot = -1.0*n[0]; if (dot > maxDot) { maxDot = dot; statusIndex = 1; }
                    dot =  1.0*n[0]; if (dot > maxDot) { maxDot = dot; statusIndex = 2; }
                    dot = -1.0*n[1]; if (dot > maxDot) { maxDot = dot; statusIndex = 3; }
                    dot =  1.0*n[1]; if (dot > maxDot) { maxDot = dot; statusIndex = 4; }
                    dot = -1.0*n[2]; if (dot > maxDot) { maxDot = dot; statusIndex = 5; }
                    int areaCodeX = 4*xi/fw;
                    int areaCodeY = 4*yi/fh;
                    int areaStatus = areaCodeX+areaCodeY*4;
                    // store normal direction index for oriented histograms
                    status[offset] = statusIndex+areaStatus*6;

                    // sample gray value and its gradient magnitude:
                    float magGrad=0,grayIntensity=0;
                    float r3[3],p2[3];
                    transformRT3(TLR, &p[0], r3); r3[0] /= r3[2]; r3[1] /= r3[2]; r3[2] = 1.0f;
                    // p2: distorted point
                    distortPointCPU(r3,kc,KR,p2);
                    int xi = (int)p2[0];
                    int yi = (int)p2[1];
                    if (xi > 0 && yi > 0 && xi < (fw-1) && yi < (fh-1)) {
                        int offset2 = xi + yi * fw;
                        // compute gradient magnitude
                        float gx0 =  grayData[offset2-1];
                        float gx1 =  grayData[offset2+1];
                        float gy0 =  grayData[offset2-fw];
                        float gy1 =  grayData[offset2+fw];
                        float dx = gx1-gx0;
                        float dy = gy1-gy0;
                        magGrad  = fabs(dx)+fabs(dy)*256.0; // encode dx and dy into a single float

                        // compute filtered gray value
                        float fx = p2[0]-xi; float fy = p2[1]-yi;
                        float a = (1-fx)*(1-fy);
                        float b = fx*(1-fy);
                        float c = (1-fx)*fy;
                        float d = fx*fy;

                        float v0 = grayData[offset2]; float v1 = gx1;
                        float v2 = gy1;               float v3 = grayData[offset2+fw+1];
                        grayIntensity = a*v0 + b*v1 + c*v2 + d*v3;
                    }
                    pData[offsetp+0]  = p[0];
                    pData[offsetp+1]  = p[1];
                    pData[offsetp+2]  = p[2];
                    pData[offsetp+3]  = -n[0];
                    pData[offsetp+4]  = -n[1];
                    pData[offsetp+5]  = -n[2];
                    pData[offsetp+6]  = 1.0f; // label this point and its neighbors valid
                    pData[offsetp+7]  = magGrad;
                    pData[offsetp+8]  = grayIntensity;
                    pData[offsetp+9]  = 0.0f;
                    pData[offsetp+10] = 0.0f;
                } else {
                    pData[offsetp+0]  = 0.0f;
                    pData[offsetp+1]  = 0.0f;
                    pData[offsetp+2]  = 0.0f;
                    pData[offsetp+3]  = 0.0f;
                    pData[offsetp+4]  = 0.0f;
                    pData[offsetp+5]  = 0.0f;
                    pData[offsetp+6]  = 0.0f; // label this point and its neighbors valid
                    pData[offsetp+7]  = 0.0f;
                    pData[offsetp+8]  = 0.0f;
                    pData[offsetp+9]  = 0.0f;
                    pData[offsetp+10] = 0.0f;
                }
            }
        }
    }
}
