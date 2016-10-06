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
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
//#include <types.h>
#include <hostUtils.h>
#include <calib/calib.h>
#include <rendering/VertexBuffer2.h>
#include <math.h>
#include <cwchar>
#include <math.h>
#include <assert.h>
#include <helper_cuda.h>
#include <tracker/basic_math.h>
//#include <cpp_type_traits.h>

using namespace std;
namespace warputils {
    #include "expmCuda.h"
    #include "f2cCuda.h"
    #include "kernelUtils.h"
    #include "reduction_kernel.cu"
    #include "expmCuda.cu"

    #define  SIZE 6
    typedef doublereal CHOLMAT[SIZE][SIZE];
    typedef doublereal CHOLVEC[SIZE];
    #include "cholesky.cu"
}
using namespace warputils;

texture<float, 2, cudaReadModeElementType> texC;

__global__ void collectPointsIntoImageKernel(int *iDataSrc, float *vDataSrc, float *Tsrc, int skipper, float *vDataDst, float *Tdst, int width, int height,float *calibDataDev, int stride) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idxStride = iDataSrc[idx]*stride;

    float3 p3,r3;
    p3.x = vDataSrc[idxStride+0];
    p3.y = vDataSrc[idxStride+1];
    p3.z = vDataSrc[idxStride+2];

    float TsrcInv[16],T[16];
    invertRT4Cuda(Tsrc,TsrcInv);
    matrixMult4(Tdst,TsrcInv,T);

    matrixMultVec4(T, p3, r3);

    float2 p2,pp;
    p2.x = r3.x / r3.z;
    p2.y = r3.y / r3.z;

    float *K     = &calibDataDev[KR_OFFSET];
    pp.x = K[0]*p2.x+K[2];
    pp.y = K[4]*p2.y+K[5];

    unsigned int xi = (unsigned int)(pp.x);
    unsigned int yi = (unsigned int)(pp.y);

    if ((xi < width) && (yi < height)) {
        int offset = (xi + yi * width)*stride;

        vDataDst[offset+0] = r3.x;
        vDataDst[offset+1] = r3.y;
        vDataDst[offset+2] = r3.z;

        for (int i = 3; i < stride; i++) {
            vDataDst[offset+i] = vDataSrc[idxStride+i];
        }
    }
}

__global__ void collectPointsKernel(int *iDataSrc, float *vDataSrc, float *Tsrc, int skipper, float *vDataDst, float *Tdst, int stride) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    int srcIndex = idx * skipper;
    int idxStrideSrc = iDataSrc[srcIndex]*stride;
    int idxStrideDst = idx*stride;

    float3 p3,r3;
    p3.x = vDataSrc[idxStrideSrc+0];
    p3.y = vDataSrc[idxStrideSrc+1];
    p3.z = vDataSrc[idxStrideSrc+2];

    float TsrcInv[16],T[16];
    invertRT4Cuda(Tsrc,TsrcInv);
    matrixMult4(Tdst,TsrcInv,T);

    matrixMultVec4(T, p3, r3);

    vDataDst[idxStrideDst+0] = r3.x;
    vDataDst[idxStrideDst+1] = r3.y;
    vDataDst[idxStrideDst+2] = r3.z;

    vDataDst[idxStrideDst+3] = vDataSrc[idxStrideSrc+3];
    vDataDst[idxStrideDst+4] = vDataSrc[idxStrideSrc+4];
    vDataDst[idxStrideDst+5] = vDataSrc[idxStrideSrc+5];

    vDataDst[idxStrideDst+6]  = 0;
    vDataDst[idxStrideDst+7]  = 0;
    vDataDst[idxStrideDst+8]  = vDataSrc[idxStrideSrc+8];
    vDataDst[idxStrideDst+9]  = vDataSrc[idxStrideSrc+9];
    vDataDst[idxStrideDst+10] = vDataSrc[idxStrideSrc+10];
    vDataDst[idxStrideDst+11] = vDataSrc[idxStrideSrc+11];
    vDataDst[idxStrideDst+12] = vDataSrc[idxStrideSrc+12];
    vDataDst[idxStrideDst+13] = vDataSrc[idxStrideSrc+13];
    vDataDst[idxStrideDst+14] = vDataSrc[idxStrideSrc+14];;
    vDataDst[idxStrideDst+15] = vDataSrc[idxStrideSrc+15];
    vDataDst[idxStrideDst+16] = vDataSrc[idxStrideSrc+16];
    vDataDst[idxStrideDst+17] = vDataSrc[idxStrideSrc+17];
    vDataDst[idxStrideDst+18] = vDataSrc[idxStrideSrc+18];
    vDataDst[idxStrideDst+19] = vDataSrc[idxStrideSrc+19];
    vDataDst[idxStrideDst+20] = vDataSrc[idxStrideSrc+20];
}


// no need to check screen bounds here (only opengl vertices)
__global__ void warpPointsKernel(int *iData, float *vData, float *weightsDev, float *T, float *calibDataDev, float *scratchPtr, float *imgData1, float *imgData2, float *imgData3, int width, int srcStride, int targetStride, int rgbOffset)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*srcStride;
    // indices to src and target vertices in the baseBuffer
    int dstIdx1 = (10000+idx)*targetStride;
    int dstIdx2 = (10000+320*240+idx)*targetStride;

    float3 p3,r3;

    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];

    matrixMultVec4(T, p3, r3);

    // project r3 into screenspace for obtaining pixel coordinates
    float *KR = &calibDataDev[KR_OFFSET];
    float *kc = &calibDataDev[KcR_OFFSET];

    float2 pu,p1_1,p2_1;//,p2_2,p2_3;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p2_1);

    // generate reference point also
    pu.x = p3.x/p3.z;
    pu.y = p3.y/p3.z;
    distortPoint(pu,kc,KR,p1_1);
    //// FETCH current depth, intensity1, intensity2, intensity3 for KEYFRAME update!

    /*
    // compute low-resolution coordinates
    float a = 0.5f; float b = -0.25f;
    p2_2.x = a*p2_1.x + b; p2_2.y = a*p2_1.y + b;
    p2_3.x = a*p2_2.x + b; p2_3.y = a*p2_2.y + b;

    float color1 = 0, color2  = 0, color3  = 0;
    int xdi = 0, ydi = 0;
    float fracX = 0.0f, fracY = 0.0f;

    xdi = (int)p2_1.x;
    ydi = (int)p2_1.y;
    fracX = p2_1.x - xdi;
    fracY = p2_1.y - ydi;
    bilinearInterpolation(xdi,   ydi,   fracX, fracY, width, imgData1, color1);
//    bilinearInterpolation(xdi,   ydi,   fracX, fracY, width, depthData1, depth1);
    // TODO
    // determine inv(T) at this point! (compute in every thread?) transpose + vector multiplication is piece of cake
    // reconstruct 3D point + intensity in all 3 layers
    // map it back to reference using inv(T)
    // IIR 3D point + intensity
    // effect -> grid is lost, but consistency maintained?
    xdi = (int)p2_2.x;
    ydi = (int)p2_2.y;
    fracX = p2_2.x - xdi;
    fracY = p2_2.y - ydi;
    bilinearInterpolation(xdi,   ydi,   fracX, fracY, width/2, imgData2, color2);

    xdi = (int)p2_3.x;
    ydi = (int)p2_3.y;
    fracX = p2_3.x - xdi;
    fracY = p2_3.y - ydi;
    bilinearInterpolation(xdi,   ydi,   fracX, fracY, width/4, imgData3, color3);
*/
    float w = weightsDev[idx];

    scratchPtr[dstIdx1+0] = p1_1.x;//vData[idxStride+6];
    scratchPtr[dstIdx1+1] = p1_1.y;//vData[idxStride+7];
    scratchPtr[dstIdx1+2] = 0.0f;
    scratchPtr[dstIdx1+rgbOffset+0] = 1.0f - w;
    scratchPtr[dstIdx1+rgbOffset+1] = w;
    scratchPtr[dstIdx1+rgbOffset+2] = 0.0f;

//    float maxDist = calibDataDev[MAXD_OFFSET];
    if (w > 0) {
        scratchPtr[dstIdx2+0] = p2_1.x;
        scratchPtr[dstIdx2+1] = p2_1.y;
    } else {
        scratchPtr[dstIdx2+0] = -1000.0f;
        scratchPtr[dstIdx2+1] = -1000.0f;

    }
    scratchPtr[dstIdx2+2] = 0.0f;
    scratchPtr[dstIdx2+rgbOffset+0] = 1.0f - w;
    scratchPtr[dstIdx2+rgbOffset+1] = w;
    scratchPtr[dstIdx2+rgbOffset+2] = 0.0f;
}


__global__ void warpBaseKernel(float *vData, float *T, int emptyVertexSlot, int stride, int rgbOffset)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idxStrideSrc = idx*stride;
    int idxStrideDst = (idx+6)*stride;

    float3 p3,r3;

    p3.x = vData[idxStrideSrc+0];
    p3.y = vData[idxStrideSrc+1];
    p3.z = vData[idxStrideSrc+2];
    float Tinv[16];

    invertRT4Cuda(T, Tinv);
    matrixMultVec4(Tinv, p3, r3);

    vData[idxStrideDst+0] = r3.x; // target point x
    vData[idxStrideDst+1] = r3.y; // target point y
    vData[idxStrideDst+2] = r3.z; // target point z

    // add new line segment as extra job :)
    // avoids yet another lock/unlock with additional vbuffer
    if (idx == 0) {
        int previousSlot = emptyVertexSlot - 1;
        if (previousSlot < 0) previousSlot = 0;
        float px = vData[previousSlot*stride+0];
        float py = vData[previousSlot*stride+1];
        float pz = vData[previousSlot*stride+2];

        vData[emptyVertexSlot*stride+0] = px;
        vData[emptyVertexSlot*stride+1] = py;
        vData[emptyVertexSlot*stride+2] = pz;
        vData[emptyVertexSlot*stride+rgbOffset+0] = 1;
        vData[emptyVertexSlot*stride+rgbOffset+1] = 0;
        vData[emptyVertexSlot*stride+rgbOffset+2] = 0;

        vData[(emptyVertexSlot+1)*stride+0] = r3.x;
        vData[(emptyVertexSlot+1)*stride+1] = r3.y;
        vData[(emptyVertexSlot+1)*stride+2] = r3.z;
        vData[(emptyVertexSlot+1)*stride+rgbOffset+0] = 1;
        vData[(emptyVertexSlot+1)*stride+rgbOffset+1] = 0;
        vData[(emptyVertexSlot+1)*stride+rgbOffset+2] = 0;
    }
}

__global__ void interpolateResidualKernel2(int *iData, float *vData, float *T, float *calibDataDev, float a, float b, int refColorOffset, float *imgData, int width, int height, float *zCurrentDev, float *zWeightsDev, float *residual, int srcStride, int zwidth, int zheight)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*srcStride;

    float *KR   = &calibDataDev[KR_OFFSET];
    float *kc   = &calibDataDev[KcR_OFFSET];
    float *TLR  = &calibDataDev[TLR_OFFSET];

    float3 p3,r3_ir,r3;

    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];

    matrixMultVec4(T,   p3,    r3_ir);   // reference IR  -> current IR
    matrixMultVec4(TLR, r3_ir, r3);      // current IR  -> current RGB

    float2 pu,p2;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p2);

    // resolution tweak:
    float2 p;
    p.x = a*p2.x + b;
    p.y = a*p2.y + b;

    float iResidual = 1.0f; // set max residual value for points outside fov
    int xdi = (int)p.x;
    int ydi = (int)p.y;
    float zWeight = 0.0f;

    if (xdi >= 0 && ydi >= 0 && xdi < width-1 && ydi < height-1) {
        float fx = p.x - xdi;
        float fy = p.y - ydi;
        float color = 0;
        bilinearInterpolation(xdi,   ydi,   fx, fy, width, imgData, color);
        iResidual = vData[idxStride+refColorOffset] - color; // residual range [-1,1]

        // fetch depth coordinate from vertex buffer (offset runs over IR image)
        float *KL  = &calibDataDev[KL_OFFSET];
        float *TRL = &calibDataDev[TRL_OFFSET];
        float3 rl3,pl2;
        matrixMultVec4(TRL, r3, rl3);   // current RGB -> current IR
        rl3.x /= rl3.z; rl3.y /= rl3.z; rl3.z = 1; // normalize
        matrixMultVec3(KL, rl3, pl2);   // project to image space
        int xdi2 = (int)(pl2.x+0.5f);   // nearest point sample in IR view
        int ydi2 = (int)(pl2.y+0.5f);

        if (xdi2 >= 0 && ydi2 >= 0 && xdi2 < zwidth && ydi2 < zheight) {
            int offset = xdi2 + ydi2*zwidth;
            float zcur = zCurrentDev[offset];
            float zerr = zcur-r3.z;
            zerr *= zerr;
            if (zerr < 100*100) {
                zWeight = 1.0f-zerr/(100.0f*100.0f);
            }
        }
    }
    residual[idx] = iResidual;
    zWeightsDev[idx] = zWeight;
}


__global__ void interpolateResidualKernel(int *iData, float *vData, float *T, float *calibDataDev, float a, float b, int refColorOffset, float *imgData, int width, int height, float *vDataCur, int zwidth, int zheight, float *residual, float *zWeights, int srcStride, int dstStride)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*srcStride;

    float *KR  = &calibDataDev[KR_OFFSET];
    float *kc  = &calibDataDev[KcR_OFFSET];

    float3 p3,r3;

    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];


    matrixMultVec4(T, p3, r3); // reference RGB  -> current RGB

    float2 pu,p2;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p2);

    // resolution tweak:
    float2 p;
    p.x = a*p2.x + b;
    p.y = a*p2.y + b;

    float zWeight   = 0.0f;
    float iResidual = 1.0f; // set max residual value for points outside fov
    int xdi = (int)p.x;
    int ydi = (int)p.y;

    if (xdi >= 0 && ydi >= 0 && xdi < width-1 && ydi < height-1) {
        float fx = p.x - xdi;
        float fy = p.y - ydi;
        float color = 0;
        bilinearInterpolation(xdi,   ydi,   fx, fy, width, imgData, color);
        iResidual = vData[idxStride+refColorOffset] - color; // residual range [-1,1]

        // fetch depth coordinate from vertex buffer (offset runs over IR image)
        float *KL  = &calibDataDev[KL_OFFSET];
        float *TRL = &calibDataDev[TRL_OFFSET];
        float3 rl3,pl2;
        matrixMultVec4(TRL, r3, rl3);   // current RGB -> current IR
        rl3.x /= rl3.z; rl3.y /= rl3.z; rl3.z = 1; // normalize
        matrixMultVec3(KL, rl3, pl2);   // project to image space
        int xdi2 = (int)(pl2.x+0.5f);   // nearest point sample in IR view
        int ydi2 = (int)(pl2.y+0.5f);
        if (xdi2 >= 0 && ydi2 >= 0 && xdi2 < zwidth && ydi2 < zheight) {
            int offset = xdi2 + ydi2*zwidth;
            float zcur = vDataCur[offset*dstStride+2];
            float zerr = zcur-r3.z;
            zerr *= zerr;
            if (zerr < 300*300) {
                zWeight = 1.0f-zerr/(300.0f*300.0f);
            }
        }
    }
    residual[idx] = iResidual;
    zWeights[idx] = zWeight;
}

__global__ void filterDepthIIRKernel(int *iData, float *vData, float *T, float *calibDataDev, int width, int height, float *vDataCur, float *weightsDev, float weightThreshold, int stride)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*stride;

    // do not average depth values if M-estimator weight is too small
    if (weightsDev[idx] < weightThreshold) return;

    float *KR  = &calibDataDev[KR_OFFSET];
    float *kc  = &calibDataDev[KcR_OFFSET];

    float3 p3,r3;
    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];
    matrixMultVec4(T, p3, r3); // reference RGB  -> current RGB

    float2 pu,p;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p);

    int xdi = (int)p.x;
    int ydi = (int)p.y;

    if (xdi >= 0 && ydi >= 0 && xdi < width-1 && ydi < height-1) {
        // fetch depth coordinate from vertex buffer (offset runs over IR image)
        float *KL  = &calibDataDev[KL_OFFSET];
        float *TRL = &calibDataDev[TRL_OFFSET];

        float3 rl3,pl2;
        matrixMultVec4(TRL, r3, rl3);   // current RGB -> current IR
        rl3.x /= rl3.z; rl3.y /= rl3.z; rl3.z = 1; // normalize
        matrixMultVec3(KL, rl3, pl2);   // project to image space
        int xdi2 = (int)(pl2.x+0.5f);   // nearest point sample in IR view
        int ydi2 = (int)(pl2.y+0.5f);
        if (xdi2 >= 0 && ydi2 >= 0 && xdi2 < width && ydi2 < height) {
            int offset = xdi2 + ydi2*width;
            float3 pc,pr,ray;
            // pc is in RGB frame of the current view
            pc.x = vDataCur[offset*stride+0];
            pc.y = vDataCur[offset*stride+1];
            pc.z = vDataCur[offset*stride+2];

            // generate mapping from current to reference
            float iT[16];
            invertRT4Cuda(&T[0],&iT[0]);
            // map current point to reference
            matrixMultVec4(iT, pc, pr); // current RGB -> reference RGB

            // generate a ray from current origin towards p3
            float len = sqrtf(p3.x*p3.x+p3.y*p3.y+p3.z*p3.z);
            ray.x = p3.x / len;  ray.y = p3.y / len; ray.z = p3.z / len;
            // project pr to ray
            float rayProj = pr.x*ray.x+pr.y*ray.y+pr.z*ray.z;
            ray.x *= rayProj; ray.y *= rayProj; ray.z *= rayProj;
            // compute orthogonal displacement from ray
/*            float3 dp;
            dp.x = pr.x - ray.x;
            dp.y = pr.y - ray.y;
            dp.z = pr.z - ray.z;*/
            // determine squared length
            //float rayDist2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
            // compute weight based on ray distance
            float distWeight = 0.9f;//__expf(-xyDist/50.0f+1e-5f);
            vData[idxStride+2] = p3.z*0.9f*(1.0f-distWeight) + 0.1f*pr.z*distWeight;
        }
    }
}


__global__ void compressVertexBufferKernel(int *iDataSrc, float *vDataSrc, int *iDataDst, float *vDataDst, int srcStride, int dstStride, bool rgbVisualization) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idxStrideSrc = iDataSrc[idx]*srcStride;

    // identity mapping
    iDataDst[idx] = idx;
    int idxStrideDst = idx*dstStride;

    if (srcStride == dstStride) {
        for (int i = 0; i < srcStride; i++) {
            vDataDst[idxStrideDst+i] = vDataSrc[idxStrideSrc+i];
        }
    } else if (srcStride == VERTEXBUFFER_STRIDE && dstStride == COMPRESSED_STRIDE){
        vDataDst[idxStrideDst+0] = vDataSrc[idxStrideSrc+0];
        vDataDst[idxStrideDst+1] = vDataSrc[idxStrideSrc+1];
        vDataDst[idxStrideDst+2] = vDataSrc[idxStrideSrc+2];
        vDataDst[idxStrideDst+3] = vDataSrc[idxStrideSrc+3];
        vDataDst[idxStrideDst+4] = vDataSrc[idxStrideSrc+4];
        vDataDst[idxStrideDst+5] = vDataSrc[idxStrideSrc+5];
        if (!rgbVisualization) {
            vDataDst[idxStrideDst+6] = vDataSrc[idxStrideSrc+14];
            vDataDst[idxStrideDst+7] = vDataSrc[idxStrideSrc+17];
            vDataDst[idxStrideDst+8] = vDataSrc[idxStrideSrc+20];
        } else {
            vDataDst[idxStrideDst+6] = vDataSrc[idxStrideSrc+8];
            vDataDst[idxStrideDst+7] = vDataSrc[idxStrideSrc+9];
            vDataDst[idxStrideDst+8] = vDataSrc[idxStrideSrc+10];
        }
    }
}


__global__ void precomputeJacobian4Kernel(int *iData, float *vData, float *calibDataDev, int vectorLength, float *jacobian1, float *jacobian2, float *jacobian3, float *jacobian4, int stride, float optScaleIn)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*stride;
    float *K  = &calibDataDev[KR_OFFSET];
    float *T  = &calibDataDev[TLR_OFFSET];
    float *kc = &calibDataDev[KcR_OFFSET];

    float3 p3,r3;
    float3 dp2,dd2,dpn2; dd2.z = 0; dpn2.z = 0;
    float3 dp3,dr3;

    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];

    matrixMultVec4(T,p3,r3);

    // input points can be scaled without altering perspective projection
    // because it is useful to have uniform magnitudes during optimization, unit scaling is supported
    r3.x *= optScaleIn;
    r3.y *= optScaleIn;
    r3.z *= optScaleIn;

    float gradX1 = vData[idxStride+11];
    float gradY1 = vData[idxStride+12];
    float gradX2 = vData[idxStride+15];
    float gradY2 = vData[idxStride+16];
    float gradX3 = vData[idxStride+18];
    float gradY3 = vData[idxStride+19];
    float gradX4 = vData[idxStride+21];
    float gradY4 = vData[idxStride+22];

    //	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
    //	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
    //	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
    //	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;

    float dN[6];
    dN[0] = 1.0f/r3.z; dN[1] =         0; dN[2] = -r3.x/(r3.z*r3.z);
    dN[3] =         0; dN[4] = 1.0f/r3.z; dN[5] = -r3.y/(r3.z*r3.z);

    float x = r3.x/r3.z; float y = r3.y/r3.z;
    float x2 = x*x;      float y2 = y*y;
    float x4 = x2*x2;    float y4 = y2*y2;
    float r2 = x2+y2;    float r4 = r2*r2;
    float dD[4];
    dD[0] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*x2+y2);
    dD[1] = kc[0]*2*x*y + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
    dD[2] = kc[0]*2*y*x + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
    dD[3] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*y2+x2);

    // param1
    dp3.x = 0.0f;
    dp3.y =-r3.z;
    dp3.z = r3.y;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;    
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*0 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*0 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*0 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*0 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;

    // param2
    dp3.x = r3.z;
    dp3.y = 0.0f;
    dp3.z =-r3.x;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*1 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*1 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*1 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*1 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;

    // param3
    dp3.x =-r3.y;
    dp3.y = r3.x;
    dp3.z = 0.0f;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*2 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*2 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*2 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*2 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;


    // param4
    dp3.x = 1.0f;
    dp3.y = 0.0f;
    dp3.z = 0.0f;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*3 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*3 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*3 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*3 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;

    // param5
    dp3.x = 0.0f;
    dp3.y = 1.0f;
    dp3.z = 0.0f;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x =  dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y =  dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*4 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*4 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*4 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*4 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;

    // param6
    dp3.x = 0.0f;
    dp3.y = 0.0f;
    dp3.z = 1.0f;
    matrixRot4(T,dp3,dr3); // basetransform only influences rotation (after torque, w=0)
    dpn2.x = dN[0]*dr3.x + dN[1]*dr3.y + dN[2]*dr3.z;
    dpn2.y = dN[3]*dr3.x + dN[4]*dr3.y + dN[5]*dr3.z;
    dd2.x =  dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y =  dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*5 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*5 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*5 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
    jacobian4[vectorLength*5 + idx] = (dp2.x/8.0f)*gradX4 + (dp2.y/8.0f)*gradY4;
}

__global__ void precomputeJacobianKernel(int *iData, float *vData, float *calibDataDev, int vectorLength, float *jacobian1, float *jacobian2, float *jacobian3, int stride, float optScaleIn)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*stride;
    float *K  = &calibDataDev[KR_OFFSET];
    float *kc = &calibDataDev[KcR_OFFSET];

    float3 p3;
    float3 dp3,dp2,dd2,dpn2; dd2.z = 0; dpn2.z = 0;

    // input points can be scaled without altering perspective projection
    // because it is useful to have uniform magnitudes during optimization, unit scaling is supported
    p3.x = vData[idxStride+0]*optScaleIn;
    p3.y = vData[idxStride+1]*optScaleIn;
    p3.z = vData[idxStride+2]*optScaleIn;

    float gradX1 = vData[idxStride+11];
    float gradY1 = vData[idxStride+12];
    float gradX2 = vData[idxStride+15];
    float gradY2 = vData[idxStride+16];
    float gradX3 = vData[idxStride+18];
    float gradY3 = vData[idxStride+19];

    //	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
    //	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
    //	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
    //	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;

    float dN[6];
    dN[0] = 1.0f/p3.z; dN[1] =         0; dN[2] = -p3.x/(p3.z*p3.z);
    dN[3] =         0; dN[4] = 1.0f/p3.z; dN[5] = -p3.y/(p3.z*p3.z);

    float x = p3.x/p3.z; float y = p3.y/p3.z;
    float x2 = x*x;      float y2 = y*y;
    float x4 = x2*x2;    float y4 = y2*y2;
    float r2 = x2+y2;    float r4 = r2*r2;
    float dD[4];
    dD[0] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*x2+y2);
    dD[1] = kc[0]*2*x*y + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
    dD[2] = kc[0]*2*y*x + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
    dD[3] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*y2+x2);

    // param1
    dp3.x = 0.0f;
    dp3.y =-p3.z;
    dp3.z = p3.y;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*0 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*0 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*0 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;

    // param2
    dp3.x = p3.z;
    dp3.y = 0.0f;
    dp3.z =-p3.x;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*1 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*1 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*1 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;

    // param3
    dp3.x =-p3.y;
    dp3.y = p3.x;
    dp3.z = 0.0f;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*2 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*2 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*2 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;


    // param4
    dp3.x = 1.0f;
    dp3.y = 0.0f;
    dp3.z = 0.0f;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x = dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y = dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*3 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*3 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*3 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;

    // param5
    dp3.x = 0.0f;
    dp3.y = 1.0f;
    dp3.z = 0.0f;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x =  dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y =  dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*4 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*4 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*4 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;

    // param6
    dp3.x = 0.0f;
    dp3.y = 0.0f;
    dp3.z = 1.0f;
    dpn2.x = dN[0]*dp3.x + dN[1]*dp3.y + dN[2]*dp3.z;
    dpn2.y = dN[3]*dp3.x + dN[4]*dp3.y + dN[5]*dp3.z;
    dd2.x =  dD[0]*dpn2.x+dD[1]*dpn2.y;
    dd2.y =  dD[2]*dpn2.x+dD[3]*dpn2.y;
    matrixMultVec3(K, dd2, dp2);
    jacobian1[vectorLength*5 + idx] = dp2.x*gradX1 + dp2.y*gradY1;
    jacobian2[vectorLength*5 + idx] = (dp2.x/2.0f)*gradX2 + (dp2.y/2.0f)*gradY2;
    jacobian3[vectorLength*5 + idx] = (dp2.x/4.0f)*gradX3 + (dp2.y/4.0f)*gradY3;
}

__global__ void weightJacobianKernel(float *jacobian, float *weights, int vectorLength,  float *weightedJacobian)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    float w = weights[idx];
    weightedJacobian[vectorLength*0 + idx] = jacobian[vectorLength*0 + idx] * w;
    weightedJacobian[vectorLength*1 + idx] = jacobian[vectorLength*1 + idx] * w;
    weightedJacobian[vectorLength*2 + idx] = jacobian[vectorLength*2 + idx] * w;
    weightedJacobian[vectorLength*3 + idx] = jacobian[vectorLength*3 + idx] * w;
    weightedJacobian[vectorLength*4 + idx] = jacobian[vectorLength*4 + idx] * w;
    weightedJacobian[vectorLength*5 + idx] = jacobian[vectorLength*5 + idx] * w;
}


__global__ void elementwiseMultKernel(float *vecA, float *vecB, float *result)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    result[idx] = vecA[idx]*vecB[idx];
}

__global__ void sumElemsKernel2(float *blockScratch, int nblocks, float *resultA, float *resultB) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx == 0) {
        float sumElems = 0.0f;
        for (int i = 0; i < nblocks; i++) {
            sumElems += blockScratch[i];
        }
        resultA[0] = sumElems;
        if (resultB != NULL)
            resultB[0] = sumElems;
    }
}
__global__ void sumElemsKernel(float *blockScratch, int nblocks, float *resultA) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx == 0) {
        float sumElems = 0.0f;
        for (int i = 0; i < nblocks; i++) {
            sumElems += blockScratch[i];
        }
        resultA[0] = sumElems;
    }
}


__global__ void matrixMult4Kernel(float *A, float *B, float *C) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx == 0) {
        matrixMult4(A,B,C);
    }
}

__global__ void matrixMult4NormalizedKernel(float *A, float *B, float *C) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx == 0) {
        matrixMult4(A,B,C);
        normalizeMat4(C);
    }
}


__global__ void invertMatrix4Kernel(float *A, float *iA, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N) {
        invertRT4Cuda(A+idx*16,iA+idx*16);
    }
}

__global__ void convertToAxisAngleKernel(float *A, float *posAxisAngle, int N) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N) {
        float m[16];
        invertRT4Cuda(A+idx*16,m);
        float q[4];
        rot2QuaternionCuda(m,q);
        float axisAngle[4];
        quaternion2AxisAngleCuda(q,axisAngle);
        posAxisAngle[idx*7+0] = m[3];
        posAxisAngle[idx*7+1] = m[7];
        posAxisAngle[idx*7+2] = m[11];
        posAxisAngle[idx*7+3] = axisAngle[0];
        posAxisAngle[idx*7+4] = axisAngle[1];
        posAxisAngle[idx*7+5] = axisAngle[2];
        posAxisAngle[idx*7+6] = axisAngle[3];
        // also save pose matrices for debugging
        for (int i = 0; i < 16; i++) A[idx*16+i] = m[i];
    }
}

__global__ void filterPoseKernel(float *posAxisAngle, float *weightsDev, int N, float *invT) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx != 0) return;
    float avg[7],weightSum = 1e-7f;
    for (int j = 0; j < 7; j++) { avg[j] = 0; }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 7; j++) {
            avg[j] += posAxisAngle[i*7+j]*weightsDev[i];
        }
        weightSum += weightsDev[i];
    }

    for (int j = 0; j < 7; j++) avg[j] = avg[j]/weightSum;
    // normalize rotation axis
 //   float len = sqrtf(avg[3]*avg[3]+avg[4]*avg[4]+avg[5]*avg[5]+1e-7f);
 //   avg[3] /= len;
  //  avg[4] /= len;
  //  avg[5] /= len;

    float T[16];
    axisAngle2RotCuda(&avg[3],T);
    T[3] = avg[0]; T[7] = avg[1]; T[11] = avg[2];
    invertRT4Cuda(T,invT);
/*
    float T[16];
    axisAngle2RotCuda(&posAxisAngle[3],T);
    T[3] = posAxisAngle[0]; T[7] = posAxisAngle[1]; T[11] = posAxisAngle[2];
    invertRT4Cuda(T,invT);
*/
/*    float q[4],T[16];
    quaternion2RotCuda(&posAxisAngle[3],T);
    T[3] = posAxisAngle[0]; T[7] = posAxisAngle[1]; T[11] = posAxisAngle[2];
    invertRT4Cuda(T,invT);*/
}


__device__ doublereal dotProduct6(doublereal *a, doublereal *b) {
    doublereal dot = 0;
    for (int i = 0; i < 6; i++) dot += a[i]*b[i];
    return dot;
}

__device__ void matrixMultVec6(doublereal *A, doublereal *x, doublereal *r)
{
    for (int i = 0; i < 6; i++) r[i] = (doublereal)0.0;
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                r[j] += A[j*6+k]*x[k];
            }
        }
}

__device__ void generateA(doublereal *x, doublereal *A) {
    A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
    A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
    A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
    A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
}

// TODO: normalize N out from quantities!
__global__ void linearFuseKernel(float *JtJDevExt, float *residual6DevExt, float weight1, float iN1, float *JtJDev, float *residual6Dev, float weight2, float iN2) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < 6) {
        residual6DevExt[idx] = iN1*weight1*residual6DevExt[idx]+iN2*weight2*residual6Dev[idx];
    }
    JtJDevExt[idx] = iN1*weight1*JtJDevExt[idx]+iN2*weight2*JtJDev[idx];
}

__global__ void conjugateGradientKernel(float *JtJDev, float *bb, doublereal tol, int maxSteps, doublereal *ADev)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx != 0) return;

    doublereal A[6*6];
    doublereal x[6];
    doublereal r[6];
    doublereal b[6];
    doublereal dir[6];
    // copy matrix to local memory for speeding up access
    for (int i = 0; i < 36; i++) A[i] = (doublereal)JtJDev[i];
    //for (int i = 0; i < 6; i++) A[i+i*6] += (doublereal)1e-8;
    for (int i = 0; i < 6; i++) { b[i] = (doublereal)bb[i]; x[i] = 0.0; r[i] = b[i]; dir[i] = b[i]; }
    int nSteps = 0;
    while (nSteps < maxSteps) {
        doublereal Adir[6];
        matrixMultVec6(A,dir,Adir);
        //step length
        doublereal rr = dotProduct6(r,r);
        doublereal Adirdir = dotProduct6(Adir,dir);
        // compute abs(Adirdir), its numerically more stable than |Adirdir|:
        doublereal div = Adirdir; if (div < 0) div = -div;
        doublereal stepLength = 0.0;
        // prevent division by zero:
        if (div > tol) stepLength = rr/Adirdir;
        // update error:
        for (int i = 0; i < 6; i++) { r[i] -= stepLength*Adir[i]; }
        doublereal rr2 = dotProduct6(r,r);
        /*
        // early exit with previous x, (minimization step failed!)
        if (rr2 > rr) {
            generateA(x,ADev);
            return;
        }
        */
        // update params:
        for (int i = 0; i < 6; i++) { x[i] += stepLength*dir[i];}

        // early exit, residual is below a threshold:
        if (sqrt(rr2) < tol) {
            generateA(x,ADev);
            return;
        }
        doublereal beta = rr2/rr;
        for (int i = 0; i < 6; i++) { dir[i] = r[i] + beta*dir[i]; }
        nSteps++;
    }
    generateA(x,ADev);
}

// only one block
__global__ void choleskyKernel(float *JtJDev, float *bb, doublereal *ADev)
{
    unsigned int idxI = threadIdx.x;
    unsigned int idxJ = threadIdx.y;
    __shared__ doublereal iA[SIZE][SIZE];
    __shared__ doublereal B[SIZE];
   // __shared__ float x[6];
    bool firstThread = (idxI == 0 && idxJ == 0);
    // load data into local memory
    iA[idxJ][idxI] = (doublereal)JtJDev[idxJ*6+idxI];
    if (idxJ == 0) B[idxI] = (doublereal)bb[idxI];
    __syncthreads();
    // single thread only:
    if (firstThread) {
        CHOLVEC P;
        // cholesky decomposition
        choldc1(6, iA,P);
        choldcsl2(6,iA,P);
        choleskyInverse(6,iA);
    }
    __syncthreads();

    __shared__ doublereal x[6];
    if (idxJ == 0) {
        x[idxI] = iA[idxI][0] * B[0] + iA[idxI][1] * B[1] + iA[idxI][2] * B[2] + iA[idxI][3] * B[3] + iA[idxI][4] * B[4] + iA[idxI][5] * B[5];
    }
    __syncthreads();
    // fill A(x) elements
    if (firstThread) generateA(x,ADev);
}

/*
__global__ void dotProductKernel(float *vecA, float *vecB, int nblocks, float *blockScratch, float *result)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (threadIdx.x > 1023) return;
    if (idx >= nblocks*1024) return;
    __shared__ float sharedMem[1024];
    sharedMem[threadIdx.x] = vecA[idx]*vecB[idx];

    for(uint stride = 512; stride > 0; stride >>= 1) {
        __syncthreads();
        if(threadIdx.x < stride)
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
    }
    if (threadIdx.x == 0)
        blockScratch[blockIdx.x] = sharedMem[0];

   __syncthreads();
    // BUG here: blockSums might not be updated yet, cuda doesn't support block synchronization, only threads
 if(idx == 0) {
        float dotSum = 0.0f;
        for (int i = 0; i < nblocks; i++) dotSum += blockScratch[i];
        result[0] = dotSum;
     }
}
*/

extern "C" void warpBase(VertexBuffer2 *vbuffer,float *T) {
    if (vbuffer == NULL || T == NULL || vbuffer->devPtr == NULL) return;
    if (vbuffer->getVertexCount() >= 10014) { /*printf("linebuffer ran out!\n");*/ return; }
    float *vData = (float*)vbuffer->devPtr;

    int targetStride = vbuffer->getStride();
    int rgbOffset = 0;
    if (targetStride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (targetStride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }

    int freeVertex = vbuffer->getVertexCount();
    dim3 cudaBlockSize(6,1,1);
    dim3 cudaGridSize(1,1,1);
    warpBaseKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(vData, T, freeVertex,targetStride,rgbOffset);

//    printf("new vertex amount : %d\n",vbuffer->getVertexCount()+2);
    vbuffer->setVertexAmount(vbuffer->getVertexCount()+2);

    checkCudaError("warpBase error");
}


extern "C" void warpPoints(VertexBuffer2 *vbuffer, float *weightsDev, float *T, float *calibDataDev, VertexBuffer2 *baseBuf, ImagePyramid2 *grayPyramid) {
    if (vbuffer == NULL || T == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || calibDataDev == NULL || weightsDev == NULL || baseBuf == NULL || baseBuf->devPtr == NULL || grayPyramid == NULL) return;

    float *imgData[3];
    assert(grayPyramid->nLayers == 3);
    for (int i = 0; i < 3; i++) {
        imgData[i] = (float*)grayPyramid->getImageRef(i).devPtr;
        if (imgData[i] == NULL) {
            printf("warpPoints error: grayPyramid layer %d not locked! panik exit \n",i);
            return;
        }
    }
    int targetStride = baseBuf->getStride();

    int rgbOffset = 0;
    if (targetStride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (targetStride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("warp points: vbuffer has wrong number of selected pixels!\n");
        return;
    }

    float *vData = (float*)vbuffer->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;
    float *dstData = (float*)baseBuf->devPtr;

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    warpPointsKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(iData,vData,weightsDev,T,calibDataDev,dstData, imgData[0],imgData[1],imgData[2],grayPyramid->getImageRef(0).width,vbuffer->getStride(),targetStride,rgbOffset);
    checkCudaError("warpPoints error");
}

extern "C" void interpolateResidual(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, ImagePyramid2 &grayPyramid, int layer, float *residual, float *zWeightsDev)
{
    if (vbuffer == NULL || vbufferCur == NULL || T == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || vbufferCur->devPtr == NULL || calibDataDev == NULL || residual == NULL || zWeightsDev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("interpolateResidual: vbuffer has wrong number of selected pixels!\n");
        return;
    }

    float *imgData = (float*)grayPyramid.getImageRef(layer).devPtr;
    cudaArray *cArray = (cudaArray*)grayPyramid.getImageRef(layer).cArray;
    if (imgData == NULL/* || cArray == NULL*/) {
        printf("given image does not have data allocated!\n");
        return;
    }
/*
    // set texture parameters
    texC.addressMode[0] = cudaAddressModeClamp;
    texC.addressMode[1] = cudaAddressModeClamp;
    texC.filterMode = cudaFilterModeLinear;
    texC.normalized = false;
    // bind the array to the texture
    cudaBindTextureToArray(texC, cArray);
*/
    int srcStride = vbuffer->getStride();
    int dstStride = vbufferCur->getStride();

    int width = grayPyramid.getImageRef(layer).width;
    int height = grayPyramid.getImageRef(layer).height;

    float *vData = (float*)vbuffer->devPtr;
    float *vDataCur = (float*)vbufferCur->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;

    int colorOffset = 0;
    if (srcStride == VERTEXBUFFER_STRIDE) {
        colorOffset = 14;
        if (layer == 1) {
            colorOffset = 17;
        } else if (layer == 2) {
            colorOffset = 20;
        }
    } else if (srcStride == COMPRESSED_STRIDE) {
        colorOffset = 6;
        if (layer == 1) {
            colorOffset = 7;
        } else if (layer == 2) {
            colorOffset = 8;
        }
    }

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    interpolateResidualKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(iData,vData,T, calibDataDev, a, b, colorOffset, imgData, width, height, vDataCur, grayPyramid.getImageRef(0).width, grayPyramid.getImageRef(0).height, residual, zWeightsDev, srcStride,dstStride);
    checkCudaError("interpolateResidual error");
}

extern "C" void interpolateResidual2(VertexBuffer2 *vbuffer, float *T, float *calibDataDev, ImagePyramid2 &grayPyramid, int layer, float *zCurrentDev, float *zWeightsDev, float *residual, cudaStream_t stream)
{
    if (vbuffer == NULL || T == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || calibDataDev == NULL || residual == NULL || zCurrentDev == NULL || zWeightsDev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("interpolateResidual: vbuffer has wrong number of selected pixels!\n");
        return;
    }
    if (layer > grayPyramid.nLayers-1) {
        printf("interpolateResidual: invalid layer number!\n");
        return;
    }

    float *imgData = (float*)grayPyramid.getImageRef(layer).devPtr;
    if (imgData == NULL) {
        printf("given image does not have data allocated!\n");
        return;
    }
    int srcStride = vbuffer->getStride();

    int width = grayPyramid.getImageRef(layer).width;
    int height = grayPyramid.getImageRef(layer).height;

    float *vData = (float*)vbuffer->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;

    int colorOffset = 0;
    if (srcStride == VERTEXBUFFER_STRIDE) {
        colorOffset = 14;
        if (layer == 1) {
            colorOffset = 17;
        } else if (layer == 2) {
            colorOffset = 20;
        } else if (layer == 3) {
            colorOffset = 23;
        }
    } else if (srcStride == COMPRESSED_STRIDE) {
        colorOffset = 6;
        if (layer == 1) {
            colorOffset = 7;
        } else if (layer == 2) {
            colorOffset = 8;
        } else {
            printf("compressed stride does not have 4th layer attributes!\n");
            return;
        }
    }

    int divisor = 1<<layer;
    float a = 1.0f/float(divisor);
    float b = 0.5f*(a-1.0f);

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    //printf("%d %d\n",vbuffer->getElementsCount(),cudaBlockSize.x);
    interpolateResidualKernel2<<<cudaGridSize,cudaBlockSize,0,stream>>>(iData,vData, T, calibDataDev, a, b, colorOffset, imgData, width, height, zCurrentDev, zWeightsDev, residual, srcStride, grayPyramid.getImageRef(0).width,grayPyramid.getImageRef(0).height);
    checkCudaError("interpolateResidual2 error");
}


/*
static int maxDist2Reso = 20*20;
static float *expTableDev = NULL;

extern "C" void initExpTable(int maxD2) {
    if (expTableDev == NULL) {
        maxDist2Reso = maxD2;
        cudaMalloc((void **)&expTableDev, maxDist2Reso*sizeof(float));
        float *expTable = new float[resolution];
        for (int i = 0; i < resolution; i++) {
            expTable[i] = exp(-50.0f*float(i)/(resolution*50.0f));
        }
    }
}

extern "C" void releaseCudaDotProduct() {
    if (blockSumDev != NULL) {
        cudaFree(blockSumDev); blockSumDev = NULL;
        cudaFree(ADev); ADev = NULL;
    }
}*/


extern "C" void filterDepthIIR(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, float *weightsDev, int width, int height, float weightThreshold)
{
    if (vbuffer == NULL || vbufferCur == NULL || T == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || vbufferCur->devPtr == NULL || calibDataDev == NULL || weightsDev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("filterDepthIIR: vbuffer has wrong number of selected pixels!\n");
        return;
    }

//    float *vData = (float*)vbuffer->devPtr;
//    float *vDataCur = (float*)vbufferCur->devPtr;
//    int   *iData = (int*)vbuffer->indexDevPtr;
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
//    filterDepthIIRKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(iData,vData,T, calibDataDev, width, height, vDataCur, weightsDev, weightThreshold);
    checkCudaError("filterDepthIIR error");
    //printf("executing iir depth filter\n"); fflush(stdin); fflush(stdout);
}


extern "C" void compressVertexBuffer(VertexBuffer2 *vbufferSrc, VertexBuffer2 *vbufferDst, bool rgbVisualization) {
    if (vbufferSrc == NULL || vbufferSrc->devPtr == NULL || vbufferSrc->indexDevPtr == NULL) return;
    if (vbufferDst == NULL || vbufferDst->devPtr == NULL || vbufferDst->indexDevPtr == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbufferSrc->getElementsCount()%1024 != 0) {
        printf("compressVertexBuffer: vbufferSrc has wrong number of selected pixels!\n");
        return;
    }


    if (vbufferDst->getMaxVertexCount() < vbufferSrc->getElementsCount()) {
        printf("vbufferDst : %d, vbufferSrc: %d\n",vbufferDst->getElementsCount(),vbufferSrc->getElementsCount());
        printf("compressVertexBuffer: vbufferDst max vertex size != vbufferSrc element size!\n");
        fflush(stdin);
        fflush(stdout);
        return;
    }
    int srcStride = vbufferSrc->getStride();
    int dstStride = vbufferDst->getStride();

    vbufferDst->setElementsCount(vbufferSrc->getElementsCount());

    float *vDataSrc = (float*)vbufferSrc->devPtr;
    int   *iDataSrc = (int*)vbufferSrc->indexDevPtr;

    float *vDataDst = (float*)vbufferDst->devPtr;
    int   *iDataDst = (int*)vbufferDst->indexDevPtr;

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(vbufferDst->getElementsCount()/cudaBlockSize.x,1,1);
    compressVertexBufferKernel<<<cudaGridSize,cudaBlockSize,0,vbufferDst->cudaStream>>>(iDataSrc,vDataSrc,iDataDst,vDataDst,srcStride,dstStride,rgbVisualization);

    char buf[512]; sprintf(buf,"compressVertexBufferKernel() execution failed, arguments: %d, %d, %d, elems: %d\n",srcStride,dstStride,int(rgbVisualization),vbufferDst->getElementsCount());
    getLastCudaError(buf);
 //   checkCudaError("compressVertexBuffer error");
}

extern "C" void compressVertexBuffer2(int *indicesExt,float *verticesExt,int pixelSelectionAmount,int srcStride, VertexBuffer2 *vbufferDst) {
    if (verticesExt == NULL || indicesExt  == NULL) return;
    if (vbufferDst == NULL || vbufferDst->devPtr == NULL || vbufferDst->indexDevPtr == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (pixelSelectionAmount % 1024 != 0) {
        printf("compressVertexBuffer2: wrong number of selected pixels!\n");
        return;
    }

    int dstStride = vbufferDst->getStride();

    if (vbufferDst->getMaxVertexCount() < pixelSelectionAmount) {
        printf("vbufferDst : %d, vbufferSrc: %d\n",vbufferDst->getElementsCount(),pixelSelectionAmount);
        printf("compressVertexBuffer2: vbufferDst max vertex size != vbufferSrc element size!\n");
        fflush(stdin);
        fflush(stdout);
        return;
    }
    bool rgbVisualization = false;
    float *vDataSrc = verticesExt;
    int   *iDataSrc = indicesExt;

    float *vDataDst = (float*)vbufferDst->devPtr;
    int   *iDataDst = (int*)vbufferDst->indexDevPtr;

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(pixelSelectionAmount/cudaBlockSize.x,1,1);
    compressVertexBufferKernel<<<cudaGridSize,cudaBlockSize,0,vbufferDst->cudaStream>>>(iDataSrc,vDataSrc,iDataDst,vDataDst,srcStride,dstStride,rgbVisualization);
    vbufferDst->setElementsCount(pixelSelectionAmount);
    checkCudaError("compressVertexBuffer2 error");
}

extern "C" void precomputeJacobian4Cuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobian1Dev, float *jacobian2Dev, float *jacobian3Dev, float *jacobian4Dev, float optScaleIn, cudaStream_t stream)
{
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || jacobian1Dev == NULL || jacobian2Dev == NULL || jacobian3Dev == NULL|| jacobian4Dev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("precomputeJacobian: vbuffer has wrong number of selected pixels!\n");
        return;
    }

    int stride = vbuffer->getStride();
    float *vData = (float*)vbuffer->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;

    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    precomputeJacobian4Kernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(iData,vData,calibDataDev,vbuffer->getElementsCount(),jacobian1Dev,jacobian2Dev,jacobian3Dev,jacobian4Dev,stride, optScaleIn);
    checkCudaError("precomputeJacobian4Cuda error");
}

extern "C" void precomputeJacobianCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobian1Dev, float *jacobian2Dev, float *jacobian3Dev, float optScaleIn)
{
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || jacobian1Dev == NULL || jacobian2Dev == NULL || jacobian3Dev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("precomputeJacobian: vbuffer has wrong number of selected pixels!\n");
        return;
    }

    int stride = vbuffer->getStride();
    float *vData = (float*)vbuffer->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;

    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    precomputeJacobianKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(iData,vData,calibDataDev,vbuffer->getElementsCount(),jacobian1Dev,jacobian2Dev,jacobian3Dev,stride, optScaleIn);
    checkCudaError("precomputeJacobianCuda error");
}

/*
extern "C" void precomputeJacobianUncompressedCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3) {
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || jacobian1Dev == NULL || jacobian2Dev == NULL || jacobian3Dev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0) {
        printf("precomputeJacobian: vbuffer has wrong number of selected pixels!\n");
        return;
    }

    int stride = vbuffer->getStride();
    float *vData = (float*)vbuffer->devPtr;
    int   *iData = (int*)vbuffer->indexDevPtr;

    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(vbuffer->getElementsCount()/cudaBlockSize.x,1,1);
    precomputeJacobianUncompressedKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(iData,vData,calibDataDev,vbuffer->getElementsCount(),jacobian1Dev,jacobian2Dev,jacobian3Dev,stride);
    checkCudaError("precomputeJacobianCuda error");
}*/

extern "C" void weightJacobian(float *jacobianTDev, float *weights, int count, float *weightedJacobianTDev, cudaStream_t stream) {
    if (jacobianTDev == NULL || count < 1024 || weightedJacobianTDev == NULL) return;

    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("wrong count for weightJacobian\n");
        return;
    }

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(count/cudaBlockSize.x,1,1);
    weightJacobianKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(jacobianTDev, weights, count, weightedJacobianTDev);
    checkCudaError("weightJacobian error");
}


static float *blockSumDev = NULL;
static doublereal *ADev = NULL;

extern "C" void initCudaDotProduct() {
    if (blockSumDev == NULL) {
        cudaMalloc((void **)&blockSumDev, 1024*sizeof(float));           cudaMemset(blockSumDev,0,1024*sizeof(float));
        cudaMalloc((void **)&ADev, 16*sizeof(doublereal));               cudaMemset(ADev,  0,  16*sizeof(doublereal));
    }
}

extern "C" void releaseCudaDotProduct() {
    if (blockSumDev != NULL) {
        cudaFree(blockSumDev); blockSumDev = NULL;
        cudaFree(ADev); ADev = NULL;
    }
}


extern "C" void dotProductCuda(float *vecA, float *vecB, int count, float *resultA, float *resultB, cudaStream_t stream) {
    if (vecA == NULL || vecB == NULL || resultA == NULL || count < 1024 || blockSumDev == NULL) {
        printf("invalid input to dotProductCuda!\n");
        return;
    }
    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("count has wrong number of pixels!\n"); fflush(stdout);
        return;
    }    
    int nthreads = 256;//512;
    int nblocks = count/nthreads;
    reduceProducts<float>(count, nthreads, nblocks, 6, vecA, vecB, blockSumDev,stream);
    dim3 cudaBlockSize(1,1,1);
    dim3 cudaGridSize(1,1,1);
    if (resultB != NULL) {
        sumElemsKernel2<<<cudaGridSize,cudaBlockSize,0,stream>>>(blockSumDev,nblocks,resultA,resultB);
    } else {
        sumElemsKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(blockSumDev,nblocks,resultA);
    }
    checkCudaError("dotProductCuda error");
}

extern "C" void JTresidualCuda(float *JT, float *residual, int count, float *result6, cudaStream_t stream) {
    if (JT == NULL || residual == NULL || count < 1024 || result6 == NULL) {
        printf("invalid input to JTresidualCuda!\n");
        return;
    }
    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("count has wrong number of pixels!\n");
        return;
    }

    dotProductCuda(JT+0*count, residual, count, result6+0,NULL,stream);
    dotProductCuda(JT+1*count, residual, count, result6+1,NULL,stream);
    dotProductCuda(JT+2*count, residual, count, result6+2,NULL,stream);
    dotProductCuda(JT+3*count, residual, count, result6+3,NULL,stream);
    dotProductCuda(JT+4*count, residual, count, result6+4,NULL,stream);
    dotProductCuda(JT+5*count, residual, count, result6+5,NULL,stream);
}

extern "C" void JTJCuda(float *JT,int count, float *JtJDev, cudaStream_t stream) {
    if (JT == NULL || count < 1024 || JtJDev == NULL) {
        printf("invalid parameters to JTJCuda.\n");
        return;
    }
    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("count has wrong number of pixels!\n"); fflush(stdout);
        return;
    }
    for (int j = 0; j < 6; j++) {
        for (int i = j; i < 6; i++) {
            dotProductCuda(JT+j*count, JT+i*count, count, JtJDev+i+j*6, JtJDev+i*6+j,stream);
        }
    }
}

void dumpp(const char *str, const float *M, int rows, int cols) {
    printf("%s:\n",str);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++)
            printf("%e ",M[i+j*cols]);
        printf("\n");
    }
}

extern "C" void linearFuseCuda(float *JtJDevExt,float *residual6DevExt, float weight1, int N1, float *JtJDev, float *residual6Dev, float weight2, int N2, cudaStream_t stream) {
    if (JtJDevExt == NULL || JtJDev == NULL || residual6DevExt == NULL || residual6Dev == NULL) {
        printf("linearFuseCuda: invalid parameters.\n");
        return;
    }
    double invN1 = 1.0/double(N1);
    double invN2 = 1.0/double(N2);
    dim3 cudaBlockSize(36,1,1);
    dim3 cudaGridSize(1,1,1);
    linearFuseKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(JtJDevExt,residual6DevExt, weight1, float(invN1), JtJDev, residual6Dev, weight2, float(invN2));
    checkCudaError("linearFuseCuda error");
}


extern "C" void solveMotionCuda(float *JtJDev, float *b, float *TDev, float scaleOut, cudaStream_t stream) {
    if (JtJDev == NULL || b == NULL || TDev == NULL || ADev == NULL) {
        printf("invalid parameters to solveMotionCuda.\n");
        return;
    }
/*
    float delay = 0.0f;
    float delays[4] = {0,0,0,0};
    int N = 1;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
*/
/*
    float *JJ = new float[36];
    cudaMemcpyAsync(&JJ[0],JtJDev,sizeof(float)*36,cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    dumpMatrix("JtJ",&JJ[0],6,6);
    delete[] JJ;
*/

  /*  for (int i = 0; i < N; i++) {*/
        doublereal tol=1e-8;
        int maxSteps = 6;
        dim3 cudaBlockSize(1,1,1);
        dim3 cudaGridSize(1,1,1);
        conjugateGradientKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(JtJDev,b,tol,maxSteps,ADev);
   // }
/*
        doublereal *A = new doublereal[16];
        cudaMemcpyAsync(&A[0],ADev,sizeof(doublereal)*16,cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        dumpMatrix("A",(double*)&A[0],4,4);
        delete[] A;*/
/*     // TODO: load ADev pÃ¤Ã¤muistiin ja tarkastele onko suuria eroja arovjen skaaloissa trans vs rot params?
        static float xmin[6] = {1e6,1e6,1e6,1e6,1e6,1e6};
        static float xmax[6] = {0,0,0,0,0,0};


        A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
        A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
        A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
        A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;



        float *A = new float[16];
        cudaMemcpy(A,ADev,sizeof(float)*16,cudaMemcpyDeviceToHost);
        float angle = sqrt(A[6]*A[6]+A[2]*A[2]+A[1]*A[1]);
        if (angle > xmax[0]) xmax[0] = angle;
        if (angle > xmax[1]) xmax[1] = angle;
        if (angle > xmax[2]) xmax[2] = angle;
        if (fabs(A[3]) > xmax[3]) xmax[3] = fabs(A[3]);
        if (fabs(A[7]) > xmax[4]) xmax[4] = fabs(A[7]);
        if (fabs(A[11])> xmax[5]) xmax[5] = fabs(A[11]);

        dumpp("xmax",xmax,1,6);

        delete[] A;
*/

     /*
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[0] += delay;


    cudaEventRecord(start,0);

    dim3 cudaBlockSize(6,6,1);
    dim3 cudaGridSize(1,1,1);
    for (int i = 0; i < N; i++) {
        choleskyKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(JtJDev,b,ADev);
    }
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[1] += delay;

    for (int i = 0; i < N; i++) {*/
      expmCuda(ADev, TDev, scaleOut, stream);
    /*}
    cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); delays[0] += delay;

    printf("expm: %fms\n",delays[0]/N);
*/
  //  printf("cgm: %fms, chol: %fms\n",delays[0]/N,delays[1]/N);


    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    checkCudaError("solveMotion error");
}

extern "C" void matrixMult4Cuda(float *A, float *B, float *C) {
    if (A == NULL || B == NULL || C == NULL) {
        printf("invalid arguments to matrixMult4Cuda\n");
        return;
    }

    dim3 cudaBlockSize(1,1,1);
    int nblocks = 1;
    dim3 cudaGridSize(nblocks,1,1);
    matrixMult4Kernel<<<cudaGridSize,cudaBlockSize,0,0>>>(A,B,C);
    //cudaThreadSynchronize();
}

extern "C" void matrixMult4NormalizedCuda(float *A, float *B, float *C)
{
    if (A == NULL || B == NULL || C == NULL) {
        printf("invalid arguments to matrixMult4Cuda\n");
        return;
    }

    dim3 cudaBlockSize(1,1,1);
    int nblocks = 1;
    dim3 cudaGridSize(nblocks,1,1);
    matrixMult4NormalizedKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(A,B,C);
}

extern "C" void invertPoseCuda(float *A, float *iA, int N, cudaStream_t stream) {
    if (A == NULL || iA == NULL || N < 1 || N > 1024) {
        printf("invalid arguments to invertPoseCuda\n");
        return;
    }
    dim3 cudaBlockSize(N,1,1);
    int nblocks = 1;
    dim3 cudaGridSize(nblocks,1,1);
    invertMatrix4Kernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(A,iA,N);
}


extern "C" void convertMatrixToPosAxisAngleCuda(float *A, float *posAxisAngle, int N) {
    if (A == NULL || posAxisAngle == NULL || N < 1) {
        printf("invalid arguments to convertMatrixToPosAxisAngleCuda\n");
        return;
    }
    dim3 cudaBlockSize(N,1,1);
    int nblocks = 1;
    dim3 cudaGridSize(nblocks,1,1);
    convertToAxisAngleKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(A,posAxisAngle,N);
}

extern "C" void filterPoseCuda(float *posAxisAngle, float *weightsDev, int N, float *T) {
    if (T == NULL || posAxisAngle == NULL || weightsDev == NULL ||  N < 1) {
        printf("invalid arguments to filterPoseCuda\n");
        return;
    }
    dim3 cudaBlockSize(1,1,1);
    int nblocks = 1;
    dim3 cudaGridSize(nblocks,1,1);
    filterPoseKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(posAxisAngle,weightsDev,N,T);
}


extern "C" void collectPointsCuda(VertexBuffer2 *vbufferSrc, float *Tsrc,  int collectedPoints256, VertexBuffer2 *vbufferDst, float *Tdst)
{
    if (vbufferSrc == NULL || vbufferSrc->devPtr == NULL || vbufferSrc->indexDevPtr == NULL) {
        printf("invalid source vbuffer (collectPointsCuda)\n");
        return;
    }
    if (vbufferDst == NULL || vbufferDst->devPtr == NULL || vbufferDst->indexDevPtr == NULL) {
        printf("invalid destination vbuffer (collectPointsCuda)\n");
        return;
    }
    if (Tsrc == NULL || Tdst == NULL || collectedPoints256 < 1) {
        printf("invalid Tsrc, Tdst or collectedPoints (collectPointsCuda)\n");
        return;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (vbufferSrc->getElementsCount()%256 != 0 || vbufferSrc->getElementsCount() <= 0) {
        printf("collectPointsCuda: vbuffer has wrong number of selected points!\n");
        return;
    }

    int stride = vbufferSrc->getStride();

    float *vDataSrc = (float*)vbufferSrc->devPtr;
    int   *iDataSrc = (int*)vbufferSrc->indexDevPtr;

    float *vDataDst = (float*)vbufferDst->devPtr;
   // int   *iDataDst = (int*)vbufferDst->indexDevPtr;

    int collectedPoints = collectedPoints256*256;
    int existingPoints = vbufferDst->getElementsCount();

    int skipper = vbufferSrc->getElementsCount()/collectedPoints;
    if (skipper < 1) skipper = 1;

    dim3 cudaBlockSize(256,1,1);
    dim3 cudaGridSize(collectedPoints/cudaBlockSize.x,1,1);
    collectPointsKernel<<<cudaGridSize,cudaBlockSize,0,vbufferSrc->cudaStream>>>(iDataSrc,vDataSrc,Tsrc, skipper,&vDataDst[existingPoints*stride],Tdst, vbufferSrc->getStride());
    vbufferDst->setElementsCount(existingPoints+collectedPoints);
    checkCudaError("collectPointsCuda error");
    // printf("elem count: %d, collected: %d, skipper: %d\n",vbufferSrc->getElementsCount(),collectedPoints,skipper);
     //fflush(stdin);
     //fflush(stdout);
}

extern "C" void collectPointsCuda2(VertexBuffer2 *vbufferSrc, float *Tsrc,  int collectedPoints256, float *vertexImageDev, float *Tdst)
{
    if (vbufferSrc == NULL || vbufferSrc->devPtr == NULL || vbufferSrc->indexDevPtr == NULL) {
        printf("invalid source vbuffer (collectPointsCuda)\n");
        return;
    }
    if (vertexImageDev == NULL) {
        printf("invalid destination vbuffer (collectPointsCuda)\n");
        return;
    }
    if (Tsrc == NULL || Tdst == NULL || collectedPoints256 < 1) {
        printf("invalid Tsrc, Tdst or collectedPoints (collectPointsCuda)\n");
        return;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (vbufferSrc->getElementsCount()%256 != 0 || vbufferSrc->getElementsCount() <= 0) {
        printf("collectPointsCuda: vbuffer has wrong number of selected points!\n");
        return;
    }

    float *vDataSrc = (float*)vbufferSrc->devPtr;
    int   *iDataSrc = (int*)vbufferSrc->indexDevPtr;

    int collectedPoints = collectedPoints256*256;
    int skipper = vbufferSrc->getElementsCount()/collectedPoints;
    if (skipper < 1) skipper = 1;

    dim3 cudaBlockSize(256,1,1);
    dim3 cudaGridSize(collectedPoints/cudaBlockSize.x,1,1);
    collectPointsKernel<<<cudaGridSize,cudaBlockSize,0,vbufferSrc->cudaStream>>>(iDataSrc,vDataSrc,Tsrc, skipper,vertexImageDev,Tdst, vbufferSrc->getStride());
    checkCudaError("collectPointsCuda error");
    // printf("elem count: %d, collected: %d, skipper: %d\n",vbufferSrc->getElementsCount(),collectedPoints,skipper);
     //fflush(stdin);
     //fflush(stdout);
}

extern "C" void setPointIntensityCuda(VertexBuffer2 *vbuffer, float *Tsrc,float *Tdst,ImagePyramid2 *grayPyramid) {
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL) {
        printf("invalid source vbuffer (setPointIntensityCuda)\n");
        return;
    }
    if (Tsrc == NULL || Tdst == NULL) {
        printf("invalid Tsrc or Tdst (setPointIntensityCuda)\n");
        return;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (vbuffer->getElementsCount()%1024 != 0 || vbuffer->getElementsCount() <= 0) {
        printf("setPointIntensityCuda: vbuffer has wrong number of selected points!\n");
        return;
    }

    if (grayPyramid == NULL || grayPyramid->getImagePtr(0) == NULL || grayPyramid->getImagePtr(1) == NULL || grayPyramid->getImagePtr(2) == NULL) {
        printf("setPointIntensityCuda: graypyramid is invalid\n");
        return;
    }

    float *imgData[3];
    assert(grayPyramid->nLayers == 3);
    for (int i = 0; i < 3; i++) {
        imgData[i] = (float*)grayPyramid->getImageRef(i).devPtr;
        if (imgData[i] == NULL) {
            printf("setPointIntensityCuda error: grayPyramid layer %d not locked! panik exit \n",i);
            return;
        }
        if (grayPyramid->getImageRef(i).renderable) {
            printf("setPointIntensityCuda error %d: grayPyramid layer is set renderable for no reason!\n",i);
        }
    }

//    float *vDataSrc = (float*)vbuffer->devPtr;
//    int   *iDataSrc = (int*)vbuffer->indexDevPtr;

    int numPoints = vbuffer->getElementsCount();

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(numPoints/cudaBlockSize.x,1,1);
    //collectPointsKernel<<<cudaGridSize,cudaBlockSize,0,vbufferSrc->cudaStream>>>(iDataSrc,vDataSrc,Tsrc, skipper,vertexImageDev,Tdst);
    checkCudaError("setPointIntensityCuda error");
}

extern "C" void collectPointsIntoImageCuda(VertexBuffer2 *vbufferSrc, float *Tsrc, int collectedPoints256, float *vertexImageDev, float *Tdst, int width, int height, float *calibDataDev) {
    if (vbufferSrc == NULL || vbufferSrc->devPtr == NULL || vbufferSrc->indexDevPtr == NULL) {
        printf("invalid source vbuffer (collectPointsIntoImageCuda)\n");
        return;
    }
    if (vertexImageDev == NULL) {
        printf("invalid destination vbuffer (collectPointsIntoImageCuda)\n");
        return;
    }
    if (Tsrc == NULL || Tdst == NULL || collectedPoints256 < 1) {
        printf("invalid Tsrc, Tdst or collectedPoints (collectPointsIntoImageCuda)\n");
        return;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (vbufferSrc->getElementsCount()%256 != 0 || vbufferSrc->getElementsCount() <= 0) {
        printf("collectPointsIntoImageCuda: vbuffer has wrong number of selected points!\n");
        return;
    }

    float *vDataSrc = (float*)vbufferSrc->devPtr;
    int   *iDataSrc = (int*)vbufferSrc->indexDevPtr;

    int collectedPoints = collectedPoints256*256;
    int skipper = vbufferSrc->getElementsCount()/collectedPoints;
    if (skipper < 1) skipper = 1;

    dim3 cudaBlockSize(256,1,1);
    dim3 cudaGridSize(collectedPoints/cudaBlockSize.x,1,1);
    collectPointsIntoImageKernel<<<cudaGridSize,cudaBlockSize,0,vbufferSrc->cudaStream>>>(iDataSrc,vDataSrc,Tsrc, skipper,vertexImageDev,Tdst,width,height,calibDataDev, vbufferSrc->getStride());
    checkCudaError("collectPointsIntoImageCuda error");
}

__global__  void vecProductKernel(float *vecA,float *vecB,float *result){

     int idx = blockIdx.x*blockDim.x+threadIdx.x;
     result[idx] = vecA[idx]*vecB[idx];
}



extern "C" void vectorProductCuda(float *vecA,float *vecB,int count,float *result, cudaStream_t stream) {
    if (vecA == NULL || vecB == NULL || result == NULL || count < 1024) {
        printf("invalid input to vectorProductCuda!\n");
        return;
    }

    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("count has wrong number of pixels! (vectorProductCuda)\n");
        return;
    }

    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(count/cudaBlockSize.x,1,1);
    vecProductKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(vecA,vecB,result);
    checkCudaError("vectorProductCuda error");
}

__global__ void listKernel(float *vData, int stride, float *selectedPoints) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // make sure stride has matching number of elements stored here!
    int idxStride = idx*stride;

    selectedPoints[idx*2+0] = vData[idxStride+6];//r3.x;//p_1.x;
    selectedPoints[idx*2+1] = vData[idxStride+7];//r3.y;//p_1.y;

}

__global__ void listSelectedRefKernel(int *indexPointer, float *vData, int stride, float *selectedPoints, float *selectionColors) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // make sure stride has matching number of elements stored here!
    int idxStride = indexPointer[idx]*stride;

    selectedPoints[idx*2+0] = vData[idxStride+6];
    selectedPoints[idx*2+1] = vData[idxStride+7];
    selectionColors[idx]     = vData[idxStride+13];
}

__global__ void listSelectedCurKernel(int *indexPointer, float *vData, float *calibDataDev, float *T, int stride, float *selectedPoints, float *selectionColors) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // make sure stride has matching number of elements stored here!
    int idxStride = indexPointer[idx]*stride;

    float *kc     = &calibDataDev[KcR_OFFSET];
    float *KR     = &calibDataDev[KR_OFFSET];

    float3 p3,r3;
    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];

    matrixMultVec4(T, p3, r3);

    float2 p_1;

    float2 pu;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p_1);

    selectedPoints[idx*2+0] = p_1.x;
    selectedPoints[idx*2+1] = p_1.y;
    selectionColors[idx]    = 0.5f;//vData[idxStride+13];
}


extern "C" void listSelectedRefCuda(VertexBuffer2 *vbuffer, float *selectionPointsDev, float *selectionColorsDev) {
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || selectionColorsDev == NULL) {
        printf("listSelectedRefCuda: null pointer given!\n"); return;
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getElementsCount()%512 != 0) {
          printf("listSelectedRefCuda: vbuffer has wrong number of selected pixels! (%d)\n",vbuffer->getElementsCount());
          return;
    }

    int *indexPointer = (int*)vbuffer->indexDevPtr;
    float *vertexData = (float*)vbuffer->devPtr;
    int nElements = vbuffer->getElementsCount();
    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
    listSelectedRefKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(indexPointer,vertexData,vbuffer->getStride(),selectionPointsDev,selectionColorsDev);

/*
     float *vertexData = (float*)vbuffer->devPtr;
     int nElements = vbuffer->getVertexCount();
     dim3 cudaBlockSize(512,1,1);
     dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
     listKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(vertexData,vbuffer->getStride(),selectionPointsDev);
*/
    checkCudaError("listSelectedRefCuda error");
}

extern "C" void listSelectedCurCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *TrelDev, float *selectionPointsDev, float *selectionColorsDev, cudaStream_t stream)
{
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || selectionPointsDev == NULL || selectionColorsDev == NULL) {
        printf("listSelectedCurCuda: null pointer given!\n"); return;
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getElementsCount()%512 != 0) {
          printf("listSelectedCurCuda: vbuffer has wrong number of selected pixels! (%d)\n",vbuffer->getElementsCount());
          return;
    }

    int *indexPointer = (int*)vbuffer->indexDevPtr;
    float *vertexData = (float*)vbuffer->devPtr;
    int nElements = vbuffer->getElementsCount();
    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
    listSelectedCurKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(indexPointer,vertexData,calibDataDev, TrelDev, vbuffer->getStride(),selectionPointsDev,selectionColorsDev);
/*
     float *vertexData = (float*)vbuffer->devPtr;
     int nElements = vbuffer->getVertexCount();
     dim3 cudaBlockSize(512,1,1);
     dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
     listKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(vertexData,vbuffer->getStride(),selectionPointsDev);
*/
    checkCudaError("listSelectedCurCuda error");
}


__global__ void xyz2DiffKernel(int *iData, float *vData, int vWidth, int vHeight, float *T, float *calibDataDev, float a, float b, int refColorOffset, float *imgData, int width, int height, int srcStride, float *diffImage)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int vindex = iData[idx];
    int idxStride = vindex*srcStride;

    float *KR   = &calibDataDev[KR_OFFSET];
    float *kc   = &calibDataDev[KcR_OFFSET];
    float *TLR  = &calibDataDev[TLR_OFFSET];

    float3 p3,r3_ir,r3;

    p3.x = vData[idxStride+0];
    p3.y = vData[idxStride+1];
    p3.z = vData[idxStride+2];

    matrixMultVec4(T,   p3,    r3_ir);   // reference IR  -> current IR
    matrixMultVec4(TLR, r3_ir, r3);      // current IR  -> current RGB

    float2 pu,p2;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p2);

    // resolution tweak:
    float2 p;
    p.x = a*p2.x + b;
    p.y = a*p2.y + b;

    float iResidual = 1.0f; // set max residual value for points outside fov
    int xdi = (int)p.x;
    int ydi = (int)p.y;

    if (xdi >= 0 && ydi >= 0 && xdi < width-1 && ydi < height-1) {
        float fx = p.x - xdi;
        float fy = p.y - ydi;
        float color = 0;
        bilinearInterpolation(xdi,   ydi,   fx, fy, width, imgData, color);
        iResidual = fabs(vData[idxStride+refColorOffset] - color); // residual range [-1,1]
    }
    int x = vindex % vWidth;
    int y = (vindex - x)/vWidth;
    diffImage[x+y*vWidth] = min(iResidual*25.0f,1);
}



extern "C" void xyz2DiffCuda(VertexBuffer2 *vbuffer, int vWidth, int vHeight, float *calibDataDev, float *TrelDev, float *diffImage, int width, int height, int layer, ImagePyramid2 *grayPyramidCur, cudaStream_t stream) {
    if (vbuffer == NULL || vbuffer->devPtr == NULL || calibDataDev == NULL || vbuffer->indexDevPtr == NULL || TrelDev == NULL || diffImage == NULL || grayPyramidCur == NULL) {
        printf("xyz2DiffCuda: null pointer given!\n"); return;
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getElementsCount()%512 != 0) {
          printf("xyz2DiffCuda: vbuffer has wrong number of selected pixels! (%d)\n",vbuffer->getElementsCount());
          return;
    }

     float *imgData = (float*)grayPyramidCur->getImageRef(layer).devPtr;
     int imgWidth = grayPyramidCur->getImageRef(layer).width;
     int imgHeight = grayPyramidCur->getImageRef(layer).height;
     if (imgData == NULL) {
         printf("xyz2DiffCuda: given image does not have data allocated!\n");
         return;
     }
     int srcStride = vbuffer->getStride();

     int colorOffset = 0;
     if (srcStride == VERTEXBUFFER_STRIDE) {
         colorOffset = 14;
         if (layer == 1) {
             colorOffset = 17;
         } else if (layer == 2) {
             colorOffset = 20;
         } else if (layer == 3) {
             colorOffset = 23;
         }
     } else if (srcStride == COMPRESSED_STRIDE) {
         colorOffset = 6;
         if (layer == 1) {
             colorOffset = 7;
         } else if (layer == 2) {
             colorOffset = 8;
         } else {
             printf("compressed stride does not have 4th layer attributes!\n");
             return;
         }
     }

     int divisor = 1<<layer;
     float a = 1.0f/float(divisor);
     float b = 0.5f*(a-1.0f);

    int *indexPointer = (int*)vbuffer->indexDevPtr;
    float *vertexData = (float*)vbuffer->devPtr;
    int nElements = vbuffer->getElementsCount();
    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
    cudaMemsetAsync(diffImage,0,sizeof(float)*vWidth*vHeight,stream);
    xyz2DiffKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(indexPointer,vertexData,vWidth,vHeight,TrelDev, calibDataDev, a, b, colorOffset, imgData, imgWidth, imgHeight, srcStride, diffImage);
    checkCudaError("xyz2DiffCuda error");
}

