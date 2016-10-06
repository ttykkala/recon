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


#include <image2/Image2.h>
//#include <types.h>
#include "hostUtils.h"
#include <helper_cuda.h>
#include <cwchar>
#include <calib/calib.h>
texture<float4, 2, cudaReadModeElementType> rgbdTex;

namespace convutils {
    #include "kernelUtils.h"
}
using namespace convutils;


// Kernel that executes on the CUDA device
/*
__global__ void rgb2GrayKernel( unsigned char *rgbPtr, unsigned char *grayPtr,int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	grayPtr[idx] = (rgbPtr[idx*3+0]*19588 + rgbPtr[idx*3+1]*38469 + rgbPtr[idx*3+2]*7471) >> 16; 	
}*/


__global__ void rgb2GrayHdrKernel( unsigned char *rgbPtr, float *grayPtr,int width)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width;
	int offsetR = 3*xi + yi*width*3;
	int offsetG = offsetR+1;
	int offsetB = offsetR+2;	
	grayPtr[offset] = float((rgbPtr[offsetR]*19588 + rgbPtr[offsetG]*38469 + rgbPtr[offsetB]*7471) >> 16); 	
}


__global__ void rgbF2GrayHdrKernel( float *rgbPtr, float *grayPtr,int width)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width;
	int offsetR = 3*xi + yi*width*3;
	int offsetG = offsetR+1;
	int offsetB = offsetR+2;	
	grayPtr[offset] = 0.3f*rgbPtr[offsetR] + 0.59f*rgbPtr[offsetG] + 0.11f*rgbPtr[offsetB]; 	
}

__global__ void rgbTex2GrayHdrZKernel(float *grayData, int width, int height, float *zMap, float *calibDataDev)
{
    int xi     = blockIdx.x*blockDim.x+threadIdx.x;
    int yi     = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    float4 rgbd = tex2D(rgbdTex, xi, height-1-yi);
  //  float4 rgbd = tex2D(rgbdTex, xi, yi);

    grayData[offset] = 0.3f*rgbd.x + 0.59f*rgbd.y + 0.11f*rgbd.z;
    float minDist    = calibDataDev[MIND_OFFSET];
    float maxDist    = calibDataDev[MAXD_OFFSET]; 
    zMap[offset]     = -(rgbd.w*(maxDist-minDist)+minDist);
}

__global__ void rgbdTex2DepthKernel(int width, int height, float *zMap, float *calibDataDev) {
    int xi     = blockIdx.x*blockDim.x+threadIdx.x;
    int yi     = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    float4 rgbd = tex2D(rgbdTex, xi, height-1-yi);

    float minDist    = calibDataDev[MIND_OFFSET];
    float maxDist    = calibDataDev[MAXD_OFFSET];
    zMap[offset]     = -(rgbd.w*(maxDist-minDist)+minDist);//-rgbd.w*10000.0f;//-(rgbd.w*(maxDist-minDist)+minDist);
}

__global__ void rgbdTex2DepthNormalizedKernel(int width, int height,float *zMap, float *calibDataDev) {
    int xi     = blockIdx.x*blockDim.x+threadIdx.x;
    int yi     = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    float4 rgbd = tex2D(rgbdTex, xi, height-1-yi);
    zMap[offset]     = rgbd.w;//-rgbd.w*10000.0f;//-(rgbd.w*(maxDist-minDist)+minDist);
}

__global__ void xyz2DepthKernel(float4 *points, int width, int height, float *zMap, float *calibDataDev) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;
    zMap[offset] = points[offset].z;
}

__global__ void xyz2DepthNormalizedKernel(float4 *points, int width, int height,float *zMap, float *calibDataDev) {
    int xi           = blockIdx.x*blockDim.x+threadIdx.x;
    int yi           = blockIdx.y*blockDim.y+threadIdx.y;
    int offset      = xi+yi*width;
    float minDist    = calibDataDev[MIND_OFFSET];
    float maxDist    = calibDataDev[MAXD_OFFSET];
    float zval = -points[offset].z;
    if (zval < minDist) zval = minDist;
    if (zval > maxDist) zval = maxDist;
    zMap[offset]     = (zval-minDist)/(maxDist-minDist);//-rgbd.w*10000.0f;//-(rgbd.w*(maxDist-minDist)+minDist);
}


__global__ void xyz6ToDepthKernel(float *points, int width, int height, float *zMap, float *calibDataDev) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;
    zMap[offset] = points[offset*6+2];
}

__global__ void xyz6ToDepthNormalizedKernel(float *points, int width, int height,float *zMap, float *calibDataDev) {
    int xi           = blockIdx.x*blockDim.x+threadIdx.x;
    int yi           = blockIdx.y*blockDim.y+threadIdx.y;
    int offset      = xi+yi*width;
    float minDist    = calibDataDev[MIND_OFFSET];
    float maxDist    = calibDataDev[MAXD_OFFSET];
    float zval = -points[offset*6+2];
    if (zval < minDist) zval = minDist;
    if (zval > maxDist) zval = maxDist;
    zMap[offset]     = (zval-minDist)/(maxDist-minDist);//-rgbd.w*10000.0f;//-(rgbd.w*(maxDist-minDist)+minDist);
}

__global__ void xyz6DiffKernel(float *xyz6Dev1,int *selectDev1,float *T1,float *calibDataDev, float *xyz6Dev2,int *selectDev2,int width,int height,float a, float b, float *diffImage) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;
    diffImage[offset] = xyz6Dev1[offset*6+2]-xyz6Dev2[offset*6+2];
}

__global__ void xyz6DiffNormalizedKernel(float *xyz6Dev1,int *selectDev1,float *T, float *calibDataDev, float *xyz6Dev2,int *selectDev2,int width,int height,float a, float b,float *diffImage) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;

    if (selectDev1[offset] == 0) {
        diffImage[offset] = 0.0f;
        return;
    }
    float *K     = &calibDataDev[KL_OFFSET];
    float *p     = &xyz6Dev1[offset*6+0];
    float3 p3    = make_float3(p[0],p[1],p[2]);
    float3 q3,q2;
    matrixMultVec4(T,p3,q3);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    float u = a*(K[0]*q2.x + K[2])+b;
    float v = a*(K[4]*q2.y + K[5])+b;
    if (u >= 0 && v >= 0 && u < width-1 && v < height-1)
    {
        int off2 = int(u+0.5f)+int(v+0.5f)*width;
        if (selectDev2[off2] > 0)
        {
            diffImage[offset] = min(fabs(q3.z-xyz6Dev2[off2*6+2])/30.0f,1.0f);
            return;
        }
    }
    diffImage[offset] = 0.0f;
}

__global__ void listSelectedKernel(float *curPointsDev,int *select,int nSelect, float *T,float *calibDataDev,float *selectionPointsDev, float *selectionColorsDev) {
    int offset    = blockIdx.x*blockDim.x+threadIdx.x;
    if (offset > nSelect) return;
    float *K     = &calibDataDev[KL_OFFSET];
    float *p     = &curPointsDev[select[offset]*6+0];
    float3 p3    = make_float3(p[0],p[1],p[2]);
    float3 q3,q2;
    matrixMultVec4(T,p3,q3);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    selectionPointsDev[offset*2+0] = K[0]*q2.x + K[2];
    selectionPointsDev[offset*2+1] = K[4]*q2.y + K[5];
    selectionColorsDev[offset] = 1.0f;
}


__global__ void icpResidualKernel(float *curPointsDev,int *selectionMask,int width,int height,float *T, float *calibDataDev, float *refPointsDev, int *selectionMaskRef, float a, float b, float *residualDev, int *residualMask) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;
    // set zero to residual as default (this is also a magic value which states that this point should not be used later on)
    if (selectionMask[offset] == 0) {
        residualMask[offset] = 0;
        return;
    }
    float *K     = &calibDataDev[KL_OFFSET];
    float *p     = &curPointsDev[offset*6+0];
    float3 p3    = make_float3(p[0],p[1],p[2]);
    float3 q3,q2,p2;
    matrixMultVec4(T,p3,q3);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    p2.x = a*(K[0]*q2.x + K[2])+b;
    p2.y = a*(K[4]*q2.y + K[5])+b;
    if (p2.x >= 0 && p2.x < (width-1) && p2.y >= 0 && p2.y < (height-1)) {
        int xdi = int(p2.x);
        int ydi = int(p2.y);
        int off2 = xdi + ydi*width;
        if (selectionMaskRef[off2] && selectionMaskRef[off2+1] && selectionMaskRef[off2+width] && selectionMaskRef[off2+width+1]) {
            float fx = p2.x - xdi;
            float fy = p2.y - ydi;
            float3 refPoint,refNormal;
            bilinearInterpolation6N(&refPointsDev[off2*6],   fx, fy, width*6, refPoint,refNormal);
            float3 delta         = make_float3(refPoint.x-q3.x,refPoint.y-q3.y,refPoint.z-q3.z);
//            float3 normal        = make_float3(refPointsDev[off2*6+3],refPointsDev[off2*6+4],refPointsDev[off2*6+5]);
            residualDev[offset]  = delta.x*refNormal.x+delta.y*refNormal.y+delta.z*refNormal.z;
            residualMask[offset] = 1;
            return;
        }
    }
    residualMask[offset] = 0;
}

__global__ void icpResidualKernel2(float *refPointsDev,int *selectionMask,int width,int height,float *T, float *calibDataDev, float *curPointsDev, int *selectionMaskCur, float a, float b, float *residualDev, int *residualMask) {
    int xi       = blockIdx.x*blockDim.x+threadIdx.x;
    int yi       = blockIdx.y*blockDim.y+threadIdx.y;
    int offset   = xi+yi*width;
    // set zero to residual as default (this is also a magic value which states that this point should not be used later on)
    if (selectionMask[offset] == 0) {
        residualMask[offset] = 0;
        return;
    }
    float *K         = &calibDataDev[KL_OFFSET];
    float *p         = &refPointsDev[offset*6+0];
    float3 p3        = make_float3(p[0],p[1],p[2]);
    float3 refNormal = make_float3(p[3],p[4],p[5]);
    float3 q3,q2,p2,curNormal;
    matrixMultVec4(T,p3,q3);
    matrixRot4(T,refNormal,curNormal);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    p2.x = a*(K[0]*q2.x + K[2])+b;
    p2.y = a*(K[4]*q2.y + K[5])+b;
    if (p2.x >= 0 && p2.x < (width-1) && p2.y >= 0 && p2.y < (height-1)) {
        int xdi = int(p2.x);
        int ydi = int(p2.y);
        int off2 = xdi + ydi*width;
        if (selectionMaskCur[off2] && selectionMaskCur[off2+1] && selectionMaskCur[off2+width] && selectionMaskCur[off2+width+1]) {
            float fx = p2.x - xdi;
            float fy = p2.y - ydi;
            float3 curPoint;//,curNormal;
            bilinearInterpolation6(&curPointsDev[off2*6],   fx, fy, width*6, curPoint);
            //bilinearInterpolation6N(&curPointsDev[off2*6],   fx, fy, width*6, curPoint,curNormal);
            float3 delta         = make_float3(curPoint.x-q3.x,curPoint.y-q3.y,curPoint.z-q3.z);
//            float3 normal        = make_float3(refPointsDev[off2*6+3],refPointsDev[off2*6+4],refPointsDev[off2*6+5]);
            residualDev[offset]  = delta.x*curNormal.x+delta.y*curNormal.y+delta.z*curNormal.z;
            residualMask[offset] = 1;
            return;
        }
    }
    residualMask[offset] = 0;
}

__global__ void icpResidualIndexKernel(float *curPointsDev,int *select,int nElems,int width, int height,float *T, float *calibDataDev, float *refPointsDev, int *refMask, float a,float b, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev) {
    int idx       = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= nElems) return;

    float *K     = &calibDataDev[KL_OFFSET];
    float *p     = &curPointsDev[select[idx]*6];
    float3 p3    = make_float3(p[0],p[1],p[2]);
    float3 q3,q2,p2;
    matrixMultVec4(T,p3,q3);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    p2.x = a*(K[0]*q2.x + K[2])+b;
    p2.y = a*(K[4]*q2.y + K[5])+b;

    if (p2.x >= 0 && p2.x < (width-1) && p2.y >= 0 && p2.y < (height-1)) {
        float weight = weightsDev[idx];
        int xdi = int(p2.x);
        int ydi = int(p2.y);
        float fx = p2.x - xdi;
        float fy = p2.y - ydi;
        int off2 = xdi + ydi*width;
        float3 refPoint,refNormal;
        // interpolate a reference point and a reference normal:
        bilinearInterpolation6N(&refPointsDev[off2*6],  fx, fy, width*6, refPoint,refNormal);
        normalize3CudaSafe(&refNormal);
        // sanity check for depth values (must be within 30cm)
        if (refMask[off2] && weight > 0.0f && fabs(refPoint.z-q3.z) < 300.0f)
        {
            float3 delta      = make_float3(refPoint.x-q3.x,refPoint.y-q3.y,refPoint.z-q3.z);
            residualDev[idx]  = weight*(delta.x*refNormal.x+delta.y*refNormal.y+delta.z*refNormal.z);
            float3 dP,dB;
            dP.x = 0; dP.y = -p3.z; dP.z = p3.y;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*0]  = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            dP.x = p3.z; dP.y = 0; dP.z = -p3.x;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*1] = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            dP.x = -p3.y; dP.y = p3.x; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*2] = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            dP.x = 1; dP.y = 0; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*3] = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            dP.x = 0; dP.y = 1; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*4] = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            dP.x = 0; dP.y = 0; dP.z = 1;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*5] = weight*(refNormal.x*dB.x+refNormal.y*dB.y+refNormal.z*dB.z);
            return;
        }
    }
    residualDev[idx]  = 0.0f;
    jacobianDev[idx+nPadded*0] = 0.0f;
    jacobianDev[idx+nPadded*1] = 0.0f;
    jacobianDev[idx+nPadded*2] = 0.0f;
    jacobianDev[idx+nPadded*3] = 0.0f;
    jacobianDev[idx+nPadded*4] = 0.0f;
    jacobianDev[idx+nPadded*5] = 0.0f;
}


__global__ void icpResidualIndexKernel2(float *refPointsDev,int *select,int nElems,int width, int height,float *T, float *calibDataDev, float *curPointsDev, int *curMask, float a,float b, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev, float optScaleIn) {
    int idx       = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= nElems) return;

    float *K     = &calibDataDev[KL_OFFSET];
    float *p     = &refPointsDev[select[idx]*6];
    float3 p3    = make_float3(p[0],p[1],p[2]);
    float3 refNormal    = make_float3(p[3],p[4],p[5]);
    float3 q3,q2,p2,curNormal;
    matrixMultVec4(T,p3,q3);
    matrixRot4(T,refNormal,curNormal);
    q2.x = q3.x/q3.z; q2.y = q3.y/q3.z; q2.z = 1;
    p2.x = a*(K[0]*q2.x + K[2])+b;
    p2.y = a*(K[4]*q2.y + K[5])+b;

    if (p2.x >= 0 && p2.x < (width-1) && p2.y >= 0 && p2.y < (height-1)) {
        float weight = weightsDev[idx];
        int xdi = int(p2.x);
        int ydi = int(p2.y);
        float fx = p2.x - xdi;
        float fy = p2.y - ydi;
        int off2 = xdi + ydi*width;
        float3 curPoint;
        bilinearInterpolation6(&curPointsDev[off2*6],  fx, fy, width*6, curPoint);
        // interpolate a reference point and a reference normal:
        //bilinearInterpolation6N(&curPointsDev[off2*6],  fx, fy, width*6, curPoint,curNormal);
        //normalize3CudaSafe(&curNormal);
        // sanity check for depth values (must be within 30cm)
        if (curMask[off2] && weight > 0.0f && fabs(curPoint.z-q3.z) < 300.0f)
        {
            float3 residual    = make_float3(curPoint.x-q3.x,curPoint.y-q3.y,curPoint.z-q3.z);
            residualDev[idx]   = optScaleIn*weight*(residual.x*curNormal.x+residual.y*curNormal.y+residual.z*curNormal.z);
            float3 dP,dB,dN;
            dP.x = 0; dP.y = -p3.z; dP.z = p3.y;
            rotate3Cuda(T,&dP,&dB);
            dP.x = 0; dP.y = -refNormal.z; dP.z = refNormal.y;
            rotate3Cuda(T,&dP,&dN);
            jacobianDev[idx+nPadded*0] = -optScaleIn*weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z+residual.x*dN.x+residual.y*dN.y+residual.z*dN.z);
            dP.x = p3.z; dP.y = 0; dP.z = -p3.x;
            rotate3Cuda(T,&dP,&dB);
            dP.x = refNormal.z; dP.y = 0; dP.z = -refNormal.x;
            rotate3Cuda(T,&dP,&dN);
            jacobianDev[idx+nPadded*1] = -optScaleIn*weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z+residual.x*dN.x+residual.y*dN.y+residual.z*dN.z);
            dP.x = -p3.y; dP.y = p3.x; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            dP.x = -refNormal.y; dP.y = refNormal.x; dP.z = 0;
            rotate3Cuda(T,&dP,&dN);
            jacobianDev[idx+nPadded*2] = -optScaleIn*weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z+residual.x*dN.x+residual.y*dN.y+residual.z*dN.z);
            dP.x = 1; dP.y = 0; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*3] = -weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z);
            dP.x = 0; dP.y = 1; dP.z = 0;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*4] = -weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z);
            dP.x = 0; dP.y = 0; dP.z = 1;
            rotate3Cuda(T,&dP,&dB);
            jacobianDev[idx+nPadded*5] = -weight*(-curNormal.x*dB.x-curNormal.y*dB.y-curNormal.z*dB.z);
            return;
        }
    }
    residualDev[idx]  = 0.0f;
    jacobianDev[idx+nPadded*0] = 0.0f;
    jacobianDev[idx+nPadded*1] = 0.0f;
    jacobianDev[idx+nPadded*2] = 0.0f;
    jacobianDev[idx+nPadded*3] = 0.0f;
    jacobianDev[idx+nPadded*4] = 0.0f;
    jacobianDev[idx+nPadded*5] = 0.0f;
}


__device__ unsigned int retirementCount = 0;

__global__ void determineIndexRangePerBlockKernel(int *mask,int nElems,int *partialSumIntDev) {
    __shared__ int sThreadStore[1024];
    __shared__ int lastBlock;

    unsigned int blockOffset = blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockOffset + tid;

    // load input into shared memory:
    int val = 0;
    if (idx < nElems && mask[idx]) val = 1;
    sThreadStore[tid] = val;
    __syncthreads();

    // first thread in the block collects valid indices into shared memory  
    unsigned int ticket = 0;
    if (tid == 0) {
        lastBlock = false;
        int blockCnt = 0;
        for (int ti = 0; ti < blockDim.x; ti++) {
            if (sThreadStore[ti]) { blockCnt++; }
        }
        partialSumIntDev[blockIdx.x] = blockCnt;
        ticket = atomicAdd(&retirementCount,1);
        lastBlock = (ticket == gridDim.x-1);
    }
    __syncthreads();
    // exit all threads which are not in the last block:
    if (!lastBlock) return;

    if (tid == 0) {
        // generate output offsets per block:
        for (int bi = 1; bi < gridDim.x; bi++) {
            partialSumIntDev[bi] += partialSumIntDev[bi-1];
        }
        retirementCount = 0;
    }
}

__global__ void selectValidKernel(int *mask, int nElems, int *partialSumIntDev, int *selectionIndexDev) {
    __shared__ int sThreadStore[1024];
    unsigned int blockOffset = blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockOffset + tid;

    // load input into shared memory:
    int val = 0;
    if (idx < nElems && mask[idx]>0) val = 1;
    sThreadStore[tid] = val;
    __syncthreads();

    // first thread in the block collects valid indices into shared memory
    if (tid == 0) {
        int startOffset = 0;
        if ( blockIdx.x > 0) startOffset = partialSumIntDev[blockIdx.x-1];

        int blockCnt = 0;
        for (int ti = 0; ti < blockDim.x; ti++) {
            if (sThreadStore[ti]) { selectionIndexDev[startOffset+blockCnt] = blockOffset+ti; blockCnt++; }
        }
        if (idx == 0) {
            // store number of selected points in the beginning of the array
            partialSumIntDev[0] = partialSumIntDev[gridDim.x-1];
        }
    }
}

__global__ void packIndexKernel(int *selectionIndex,float *fullResidualDev,int nElems,float *residualDev, float *residual2Dev) {
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= nElems) return;
    float r = fullResidualDev[selectionIndex[idx]];
    residualDev[idx] = r;
    residual2Dev[idx] = r*r;
}

__global__ void initWeightsKernel(float *residual2Dev,int nElems,float sigma2, float *weightsDev) {
    int idx      = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= nElems) return;
    weightsDev[idx] = 6/(5+residual2Dev[idx]/sigma2);
}

// reduces all blocks elements in a block into a single sum
template<unsigned int numThreads> __device__ void reduction4LogStepShared(float *out, volatile float *partials) {
    const int tid = threadIdx.x;
    if (numThreads >= 1024) {
        if (tid < 512) {
            partials[tid] += partials[tid+512];
        }
        __syncthreads();
    }
    if (numThreads >= 512) {
        if (tid < 256) {
            partials[tid] += partials[tid+256];
        }
        __syncthreads();
    }
    if (numThreads >= 256) {
        if (tid < 128) {
            partials[tid] += partials[tid+128];
        }
        __syncthreads();
    }
    if (numThreads >= 128) {
        if (tid < 64) {
            partials[tid] += partials[tid+64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (numThreads >= 64) { partials[tid] += partials[tid+32];  }
        if (numThreads >= 32) { partials[tid] += partials[tid+16];  }
        if (numThreads >= 16) { partials[tid] += partials[tid+ 8];  }
        if (numThreads >=  8) { partials[tid] += partials[tid+ 4];  }
        if (numThreads >=  4) { partials[tid] += partials[tid+ 2];  }
        if (numThreads >=  2) { partials[tid] += partials[tid+ 1];  }
        if (tid == 0) {
            *out = partials[0];
        }
    }
}

__global__ void estimateVariance1024(float *input1Dev, float *input2Dev, int N, float *blockSumDev) {
    __shared__ float sThreadSum[1024];
    unsigned int tid = threadIdx.x;
    int sum = 0;
    for (size_t i = blockDim.x*blockIdx.x + tid; i < N; i+= blockDim.x*gridDim.x) {
        sum += input1Dev[i]*input2Dev[i];
    }
    sThreadSum[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1) {
        reduction4LogStepShared<1024>(&blockSumDev[0],sThreadSum); return;
    }
    reduction4LogStepShared<1024>(&blockSumDev[blockIdx.x],sThreadSum);

    __shared__ bool lastBlock;
    // wait for outstanding memory instructions in this thread
    __threadfence();

    if (tid == 0) {
        unsigned int ticket = atomicAdd(&retirementCount,1);
        lastBlock = (ticket == gridDim.x-1);
    }
    __syncthreads();

    if (lastBlock) {
        // last block threads sum through blockSums
        // -> nThreads elems produced, zero padding for slots >= gridDim.x
        int sum = 0;
        for (size_t i = tid; i < gridDim.x; i+= blockDim.x) {
            sum += blockSumDev[i];
        }
        sThreadSum[threadIdx.x] = sum;
        __syncthreads();
        reduction4LogStepShared<1024>(&blockSumDev[0],sThreadSum);
        float minVal = float(N);
        if (blockSumDev[0] < minVal) blockSumDev[0] = minVal;
        // reset counter after execution
        retirementCount = 0;
    }
}

__global__ void updateWeightsKernel(float *residual2Dev,int nElems,float *variance, float *weightsDev) {
    int idx      = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= nElems) return;
//    weightsDev[idx] = 6/(5+residual2Dev[idx]/(float(nElems)/float(nElems)));
    weightsDev[idx] = 6/(5+residual2Dev[idx]/(variance[0]/float(nElems)));
}

__global__ void clampWeightsKernel(float *weightsDev,int nElems) {
    int idx      = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= nElems) return;
    weightsDev[idx] = min(weightsDev[idx],1.0f);
}
__global__ void convertZMapToXYZKernel(float *zPtr, float4 *refPoints, float *calibDataDev, int width)
{
    unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int idx    = xi+yi*width;
    float z   = zPtr[idx];
    float minDist      = calibDataDev[MIND_OFFSET];
    float maxDist      = calibDataDev[MAXD_OFFSET];
    if (z < -minDist && z > -maxDist) {
        float fx      = calibDataDev[KL_OFFSET];
        float fy      = calibDataDev[KL_OFFSET+4];
        float cx      = calibDataDev[KL_OFFSET+2];
        float cy      = calibDataDev[KL_OFFSET+5];
        refPoints[idx].x = (float(xi) - cx) * z / fx;
        refPoints[idx].y = (float(yi) - cy) * z / fy;
        refPoints[idx].z = z;
        refPoints[idx].w = 1;
    } else {
        refPoints[idx].x = 0;
        refPoints[idx].y = 0;
        refPoints[idx].z = 0;
        refPoints[idx].w = 0;
    }
}

__global__ void convertZMapToXYZ6Kernel(float *zPtr, float *refPoints, int *selectionMask, float *calibDataDev, int width)
{
    unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int idx    = xi+yi*width;
    float z   = zPtr[idx];
    float minDist      = calibDataDev[MIND_OFFSET];
    float maxDist      = calibDataDev[MAXD_OFFSET];
    if (z < -minDist && z > -maxDist) {
        float fx      = calibDataDev[KL_OFFSET];
        float fy      = calibDataDev[KL_OFFSET+4];
        float cx      = calibDataDev[KL_OFFSET+2];
        float cy      = calibDataDev[KL_OFFSET+5];
        refPoints[idx*6+0] = (float(xi) - cx) * z / fx;
        refPoints[idx*6+1] = (float(yi) - cy) * z / fy;
        refPoints[idx*6+2] = z;
        selectionMask[idx] = 1;
//        refPoints[idx*6+3] = 0;
//        refPoints[idx*6+4] = 0;
//        refPoints[idx*6+5] = 0;
    } else {
        refPoints[idx*6+0] = 0;
        refPoints[idx*6+1] = 0;
        refPoints[idx*6+2] = 0;
        selectionMask[idx] = 0;
//        refPoints[idx*6+3] = 0;
//        refPoints[idx*6+4] = 0;
//        refPoints[idx*6+5] = 0;
    }
}


__global__ void rgbTex2GrayHdrKernel(float *grayData, int width, int height)
{
    int xi     = blockIdx.x*blockDim.x+threadIdx.x;
    int yi     = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    float4 rgbd = tex2D(rgbdTex, xi, height-1-yi);
    grayData[offset] = 0.3f*rgbd.x + 0.59f*rgbd.y + 0.11f*rgbd.z;
}


__global__ void convertHdrRGBKernel(unsigned char *srcPtr, float *dstPtr, unsigned int width, unsigned int height) {
    unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int pitch = width*3;
    int offsetR = 3*xi + yi*pitch;
    int offsetG = offsetR+1;
    int offsetB = offsetR+2;
    dstPtr[offsetR] = ((float)srcPtr[offsetR])/255.0f;
    dstPtr[offsetG] = ((float)srcPtr[offsetG])/255.0f;
    dstPtr[offsetB] = ((float)srcPtr[offsetB])/255.0f;
}

extern "C" void rgb2GrayCuda(Image2 *rgbImage, Image2 *grayImage)
{
	if (rgbImage == 0 || rgbImage->devPtr == NULL || grayImage == 0 || grayImage->devPtr == NULL) return;
	float *grayPtr = (float*)grayImage->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(grayImage->width/cudaBlockSize.x,grayImage->height/cudaBlockSize.y,1);

	if (!rgbImage->hdr) {
	     unsigned char *rgbPtr = (unsigned char*)rgbImage->devPtr;
	     rgb2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,rgbImage->cudaStream>>>(rgbPtr,grayPtr,grayImage->width);
	} else {
	     float *rgbPtr = (float*)rgbImage->devPtr;
	     rgbF2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,rgbImage->cudaStream>>>(rgbPtr,grayPtr,grayImage->width);
	}
}

extern "C" void convertZmapToXYZCuda(float *zReferenceDev, float4 *refPointsDev, float *calibDataDev, int width, int height, cudaStream_t stream) {
    if (zReferenceDev == 0 || refPointsDev == 0 || calibDataDev == 0) return;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);
    convertZMapToXYZKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(zReferenceDev,refPointsDev,calibDataDev,width);
}

extern "C" void convertZmapToXYZ6Cuda(float *zReferenceDev, float *refPointsDev, int *selectionMask, float *calibDataDev, int width, int height, cudaStream_t stream) {
    if (zReferenceDev == 0 || refPointsDev == 0 || calibDataDev == 0) return;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);
    convertZMapToXYZ6Kernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(zReferenceDev,refPointsDev,selectionMask, calibDataDev,width);
    checkCudaError("convertZmapToXYZ6Cuda error");
}




extern "C" void rgbTex2GrayCuda(cudaArray *rgbdImage, Image2 *grayImage, float *zMap, float *calibDataDev, cudaStream_t stream)
{
    if (rgbdImage == 0 || grayImage == 0 || grayImage->devPtr == NULL) return;
    float *grayPtr = (float*)grayImage->devPtr;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(grayImage->width/cudaBlockSize.x,grayImage->height/cudaBlockSize.y,1);

    checkCudaErrors(cudaBindTextureToArray(rgbdTex, rgbdImage));
    struct cudaChannelFormatDesc desc;
    checkCudaErrors(cudaGetChannelDesc(&desc, rgbdImage));

    if (zMap != NULL && calibDataDev != NULL) {
        rgbTex2GrayHdrZKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(grayPtr,grayImage->width,grayImage->height,zMap,calibDataDev);
    } else {
        rgbTex2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(grayPtr,grayImage->width,grayImage->height);
    }
//    rgbF2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(rgbPtr,grayPtr,grayImage->width);
}

extern "C" void rgbdTex2DepthCuda(cudaArray *rgbdImage, int width, int height, float *zMap, float *calibDataDev, bool normalizeDepth, cudaStream_t stream) {
    if (rgbdImage == 0 || zMap == 0 || calibDataDev == NULL) return;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);
    checkCudaErrors(cudaBindTextureToArray(rgbdTex, rgbdImage));
    struct cudaChannelFormatDesc desc;
    checkCudaErrors(cudaGetChannelDesc(&desc, rgbdImage));

    if (!normalizeDepth) {
        rgbdTex2DepthKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(width,height,zMap,calibDataDev);
    } else {
        rgbdTex2DepthNormalizedKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(width,height,zMap,calibDataDev);
    }
    checkCudaError("rgbdTex2DepthCuda error");
}

extern "C" void xyz2DepthCuda(float4 *xyzDev, int width, int height, float *zMap, float *calibDataDev, bool normalizeDepth) {
    if (xyzDev == 0 || zMap == 0 || calibDataDev == NULL) return;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);

    cudaStream_t stream = 0;
    if (!normalizeDepth) {
        xyz2DepthKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyzDev,width,height,zMap,calibDataDev);
    } else {
        xyz2DepthNormalizedKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyzDev,width,height,zMap,calibDataDev);
    }
//    rgbF2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(rgbPtr,grayPtr,grayImage->width);
}

extern "C" void pointCloud6ToDepthCuda(float *xyz6Dev, int width, int height, float *zMap, float *calibDataDev, bool normalizeDepth, cudaStream_t stream) {
    if (xyz6Dev == 0 || zMap == 0 || calibDataDev == NULL) return;
    dim3 cudaBlockSize(16,15,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);

    if (!normalizeDepth) {
        xyz6ToDepthKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyz6Dev,width,height,zMap,calibDataDev);
    } else {
        xyz6ToDepthNormalizedKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyz6Dev,width,height,zMap,calibDataDev);
    }
//    rgbF2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(rgbPtr,grayPtr,grayImage->width);
}

extern "C" void pointCloud6DiffCuda(float *xyz6Dev1, int *selectDev1, float *Tdev, float *calibDataDev, float *xyz6Dev2, int *selectDev2, int width, int height, int layer, float *diffImage,bool normalize, cudaStream_t stream) {
    if (xyz6Dev1 == 0 || xyz6Dev2 == 0 || selectDev1 == 0 || selectDev2 == 0 || diffImage == 0 || Tdev == 0 || calibDataDev == 0 || width < 1 || height < 1) return;
    dim3 cudaBlockSize(16,15,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    if (!normalize) {
        xyz6DiffKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyz6Dev1,selectDev1,Tdev, calibDataDev,xyz6Dev2,selectDev2,width,height,a, b,diffImage);
    } else {
        xyz6DiffNormalizedKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(xyz6Dev1,selectDev1,Tdev, calibDataDev, xyz6Dev2,selectDev2,width,height,a,b,diffImage);
    }
}

extern "C" void listDepthSelectedCuda(float *curPointsDev, int *select, int nSelect, float *TDev, float *calibDataDev, float *selectionPointsDev, float *selectionColorsDev, cudaStream_t stream) {
    if (curPointsDev == 0 || select == 0 || TDev == 0 || calibDataDev == 0 || selectionPointsDev == 0 || selectionColorsDev == 0 ) return;
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nSelect/cudaBlockSize.x,1,1); if (nSelect%cudaBlockSize.x != 0) cudaGridSize.x++;
    listSelectedKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(curPointsDev,select,nSelect,TDev,calibDataDev, selectionPointsDev, selectionColorsDev);
}

extern "C" void icpResidualMaskCuda(float *curPointsDev,int *selectionMask,int width,int height,float *TDev,float *calibDataDev, float *refPointsDev, int *selectionMaskRef, int layer, float *residualDev, int *residualMask, cudaStream_t stream) {
    if (curPointsDev == 0 || selectionMask == 0 || TDev == 0 || calibDataDev == 0 || refPointsDev == 0 || selectionMaskRef == 0 || residualDev == 0 || residualMask == 0) return;
    dim3 cudaBlockSize(16,15,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    icpResidualKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(curPointsDev,selectionMask,width,height,TDev, calibDataDev, refPointsDev, selectionMaskRef, a,b,residualDev, residualMask);
}

extern "C" void icpResidualMaskCuda2(float *refPointsDev,int *selectionMask,int width,int height,float *TDev,float *calibDataDev, float *curPointsDev, int *selectionMaskRef, int layer, float *residualDev, int *residualMask, cudaStream_t stream) {
    if (curPointsDev == 0 || selectionMask == 0 || TDev == 0 || calibDataDev == 0 || refPointsDev == 0 || selectionMaskRef == 0 || residualDev == 0 || residualMask == 0) return;
    dim3 cudaBlockSize(16,15,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    icpResidualKernel2<<<cudaGridSize,cudaBlockSize,0,stream>>>(refPointsDev,selectionMask,width,height,TDev, calibDataDev, curPointsDev, selectionMaskRef, a,b,residualDev, residualMask);
}


extern "C" void icpResidualCuda(float *curPointsDev, int *select, int nElems, int width, int height, int layer, float *TDev, float *calibDataDev, float *refPointsDev, int *refMask, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev, cudaStream_t stream) {
    if (curPointsDev == 0 || select == 0 || TDev == 0 || calibDataDev == 0 || refPointsDev == 0 || residualDev == 0 || jacobianDev == 0 || weightsDev == 0 || refMask == 0) return;
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nElems/cudaBlockSize.x,1,1); if (nElems%cudaBlockSize.x != 0) cudaGridSize.x++;

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    if (nPadded % 1024 != 0) {
        printf("icpResidualCuda: nPadded not multiple of 1024!\n"); fflush(stdout); return;
    }
    icpResidualIndexKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(curPointsDev,select,nElems,width,height,TDev, calibDataDev, refPointsDev, refMask, a,b, nPadded, residualDev, jacobianDev, weightsDev);
//    icpResidualIndexKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(curPointsDev,select,nElems,width,height,TDev, calibDataDev, refPointsDev, refMask, a,b, nPadded, residualDev, jacobianDev, weightsDev);
    checkCudaError("icpResidualCuda error");
}

extern "C" void icpResidualCuda2(float *refPointsDev, int *select, int nElems, int width, int height, int layer, float *TDev, float *calibDataDev, float *curPointsDev, int *curMask, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev, float optScaleIn, cudaStream_t stream) {
    if (curPointsDev == 0 || select == 0 || TDev == 0 || calibDataDev == 0 || refPointsDev == 0 || residualDev == 0 || jacobianDev == 0 || weightsDev == 0 || curMask == 0) return;
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nElems/cudaBlockSize.x,1,1); if (nElems%cudaBlockSize.x != 0) cudaGridSize.x++;

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    if (nPadded % 1024 != 0) {
        printf("icpResidualCuda2: nPadded not multiple of 1024!\n"); fflush(stdout); return;
    }
    icpResidualIndexKernel2<<<cudaGridSize,cudaBlockSize,0,stream>>>(refPointsDev,select,nElems,width,height,TDev, calibDataDev, curPointsDev, curMask, a,b, nPadded, residualDev, jacobianDev, weightsDev, optScaleIn);
//    icpResidualIndexKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(curPointsDev,select,nElems,width,height,TDev, calibDataDev, refPointsDev, refMask, a,b, nPadded, residualDev, jacobianDev, weightsDev);
    checkCudaError("icpResidualCuda2 error");
}

extern "C" void selectValidIndex(int *residualMask, int nElems,int *partialSumIntDev, int maxBlocks, int *selectionIndexDev, cudaStream_t stream)
{
    if (nElems <= 0 || partialSumIntDev == NULL || selectionIndexDev == NULL || residualMask == NULL) {
        printf("bad arguments given to selectValidIndex!\n"); fflush(stdout); return;
    }
    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(nElems/cudaBlockSize.x,1,1); if (nElems%cudaBlockSize.x != 0) cudaGridSize.x++;

    if (cudaGridSize.x > maxBlocks) {
        printf("too many blocks in selectValidIndex!\n"); fflush(stdout); return;
    }
    determineIndexRangePerBlockKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residualMask,nElems,partialSumIntDev);
    selectValidKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residualMask,nElems,partialSumIntDev,selectionIndexDev);
}

extern "C" void packIndex(int *selectionIndexDev,int nElems,float *fullResidualDev,float *residualDev, float *residual2Dev, cudaStream_t stream) {
    if (selectionIndexDev == 0 || residualDev == 0 || nElems <= 0 || fullResidualDev == NULL || residualDev == NULL || residual2Dev == NULL) {
        printf("bad arguments given to packIndex!\n"); fflush(stdout); return;
    }
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nElems/cudaBlockSize.x,1,1); if (nElems%cudaBlockSize.x != 0) cudaGridSize.x++;

    packIndexKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(selectionIndexDev,fullResidualDev,nElems,residualDev,residual2Dev);
}



extern "C" void generateStudentTWeights(float *residual2Dev,int nElems,float *blockSumDev, int maxBlocks, float *weightsDev, cudaStream_t stream) {
    if (residual2Dev == 0 || nElems <= 0 || weightsDev == 0 || nElems < 1 || blockSumDev == NULL || maxBlocks < 1) {
        printf("bad arguments given to generateStudentTWeights!\n"); fflush(stdout); return;
    }
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nElems/cudaBlockSize.x,1,1); if (nElems%cudaBlockSize.x != 0) cudaGridSize.x++;

    if (cudaGridSize.x > maxBlocks) {
        printf("too many blocks in generateStudentTWeights!\n"); fflush(stdout); return;
    }

    initWeightsKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residual2Dev,nElems,100.0f,weightsDev);
    int N = 3;
    for (int i = 0; i < N; i++) {
        // returns sum residualDev .* weightsDev -> blockSumDev[0]
        estimateVariance1024<<<cudaGridSize,cudaBlockSize,0,stream>>>(residual2Dev,weightsDev,nElems,blockSumDev);
        updateWeightsKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residual2Dev,nElems,blockSumDev,weightsDev);
/*        float var = 0.0f;
        cudaMemcpy(&var,blockSumDev,sizeof(float),cudaMemcpyDeviceToHost);
        printf("var[%d]: %f\n",i,var/float(nElems)); fflush(stdout);*/
    }
    clampWeightsKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(weightsDev,nElems);

}


extern "C" void convertToHDRCuda(Image2 *imRGB, Image2 *imRGBHDR)
{
    if (imRGB == 0 || imRGB->devPtr == NULL || imRGBHDR == 0 || imRGBHDR->devPtr == NULL) return;
    unsigned char *srcPtr = (unsigned char*)imRGB->devPtr;
    float *dstPtr= (float*)imRGBHDR->devPtr;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(imRGB->width/cudaBlockSize.x,imRGB->height/cudaBlockSize.y,1);
    convertHdrRGBKernel<<<cudaGridSize,cudaBlockSize,0,imRGB->cudaStream>>>(srcPtr,dstPtr,(unsigned int)imRGB->width,(unsigned int)imRGB->height);
}
