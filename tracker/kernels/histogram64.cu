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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "histogram_common.h"
#include <cuda_histogram.h>

// minimal test - 1 key per input index
struct test_xform {
  __host__ __device__
  void operator() (int* input, int i, int* res_idx, int* res, int nres) const {
    *res_idx++ = input[i];
    *res++ = 1;
  }
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun {
    __device__ __host__ int operator() (int res1, int res2) const{
        return res1 + res2;
    }
};

__global__ void discretizeResidualKernel(float *residualDev, int *discreteResidual) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    discreteResidual[idx] = (int)(63.0f*fabs(residualDev[idx]));
}


// this kernel assumes only one thread!
__global__ void seekThreshold64Kernel(int *d_Histogram, int pixelAmount) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx != 0) return;

    int binThreshold = 0; int selectedPixelAmount = 0;
    for (int i = 0; i < HISTOGRAM64_BIN_COUNT; i++) {
        selectedPixelAmount += d_Histogram[i];
        if (selectedPixelAmount > pixelAmount) {
            binThreshold = i;
            break;
        }
    }
    d_Histogram[0] = binThreshold;
}

/*
void MEstimator::tukeyW(float deviation, unsigned char *val, float *weights, unsigned int length) {
        float thresdev = 4.6851f*deviation;
        for (unsigned int i = 0; i < length; i++) {
                float valf = float(val[i]);
                if (valf < thresdev) {
                        float v = valf/thresdev;
                        v *= v; v = 1-v; v *= v;
                        weights[i] = v;
                } else {
                        weights[i] = 0;
                }
        }
}
*/

__global__ void generateWeights64Kernel(float *residual, int *median64, float *weightsDev, float *extWeightsDev, float *weightedResidual) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    float s = 1.4826f * float(4*median64[0]+3)/255.0f;
    float b = 4.6851f;
    float r = residual[idx];

    float u = fabs(r)/s;
    float w = 0.0f;
    if (u <= b) {
        float ub = u / b; float ub2 = ub*ub;
        w = ( 1.0f - ub2 )*extWeightsDev[idx];
    }
    weightedResidual[idx] = r*w;
    weightsDev[idx] = w;
}

static int *d_Histogram64 = NULL;
static int *discreteResidual = NULL;

//Internal memory allocation
extern "C" void initHistogram64(void){
    if (d_Histogram64 == NULL) {
        checkCudaErrors( cudaMalloc((void **)&d_Histogram64, HISTOGRAM64_BIN_COUNT * sizeof(uint)) );
        checkCudaErrors( cudaMalloc((void **)&discreteResidual, 320*240*sizeof(int)) );
    }
}

//Internal memory deallocation
extern "C" void closeHistogram64(void) {
    if (d_Histogram64 != NULL) {
        checkCudaErrors( cudaFree(d_Histogram64) ); d_Histogram64 = NULL;
        checkCudaErrors( cudaFree(discreteResidual ) ); discreteResidual = NULL;
    }
}

extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev, float *weightedResidualDev, cudaStream_t stream) {
    if (residualDev == NULL || count < 1024 || weightsDev == NULL || weightedResidualDev == NULL) {
        printf("invalid args to generateWeights64 count == %d!\n",count);
        return;
    }
    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("wrong number of selected pixels!\n");
        return;
    }

//    int cnt4 = count/4;
    dim3 cudaBlockSize(1024,1,1);
//    dim3 cudaGridSize(cnt4/cudaBlockSize.x,1,1);
    dim3 cudaGridSize(count/cudaBlockSize.x,1,1);
    while (cudaGridSize.x == 0) {
        cudaBlockSize.x /= 2;
        cudaGridSize.x = count/cudaBlockSize.x;
//        cudaGridSize.x = cnt4/cudaBlockSize.x;
    }
    discretizeResidualKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residualDev,(int*)discreteResidual);

    // Create the necessary function objects and run histogram using them
    // 1 result per input index, 6 bins, 10 inputs
    test_xform xform;
    test_sumfun sum;
    callHistogramKernel<histogram_atomic_inc, 1>(discreteResidual, xform, sum, 0, count, 0, d_Histogram64, 64);

    dim3 cudaBlockSize1(1,1,1); dim3 cudaGridSize1(1,1,1);
    seekThreshold64Kernel<<<cudaGridSize1,cudaBlockSize1,0,stream>>>(d_Histogram64, count/2);
    getLastCudaError("generateWeights64 -- seekThreshold64() execution failed\n");

 /*   unsigned int histCPU[64];
    cudaMemcpy(&histCPU[0],d_Histogram64,64*sizeof(int),cudaMemcpyDeviceToHost);
    FILE *f = fopen("histdata.dat","a");
    for (int i = 0; i < 64; i++) fprintf(f,"%d ",histCPU[i]);
    fprintf(f,"\n");
    fclose(f);*/

    cudaBlockSize.x = 1024;
    cudaGridSize.x = count / cudaBlockSize.x;
    generateWeights64Kernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(residualDev,d_Histogram64,weightsDev,extWeightsDev,weightedResidualDev);
    getLastCudaError("generateWeights64 -- generateWeights64Kernel() execution failed\n");
}
