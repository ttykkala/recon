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
#include <rendering/VertexBuffer2.h>

// this kernel assumes only one thread!
__global__ void seekThreshold256Kernel2(float *d_Histogram, int pixelAmount, int nbins) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // return the bin where pixel amount is exceeded
        uint binThreshold = 0; int selectedPixelAmount = 0;
        for (int i = (nbins-1); i >= 0; i--) {
            selectedPixelAmount += (uint)d_Histogram[i];
            if (selectedPixelAmount > pixelAmount) {
                binThreshold = i;
                break;
            }
        }
//        d_Histogram[0] = (float)binThreshold;
        // how many pixels too much?
//        d_Histogram[1] = (float)(selectedPixelAmount-pixelAmount);

        d_Histogram[0] = (float)binThreshold;
        // how many pixels too much?
        d_Histogram[1] = (float)(selectedPixelAmount-pixelAmount);

    }
}
/*
__global__ void filterIndexKernel(float *vertexData, uint *hist, int *indexPointer, int nthreads, int pointsPerThread, int stride, int slot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startOffset = idx*pointsPerThread;
    int lastOffset = startOffset + (pointsPerThread-1);
 //   if (lastOffset >= 320*240) return;

    __shared__ int countInSegment[1024];
    __shared__ int startIndexOfSegment[1024];
    __shared__ int mediocreCountInSegment[1024];
    __shared__ int mediocreStartIndexOfSegment[1024];

    int packedSegment[1024];
    int mediocrePackedSegment[1024];

    // pack each segment in parallel and store count
    uint threshold = hist[0];

    int cnt = 0, mediocreCnt = 0;
    for (int i = startOffset; i <= lastOffset; i++) {
        uint vertexQuality = (uint)(vertexData[i*stride+slot]);
        if (vertexQuality > threshold) {
            packedSegment[cnt] = i; cnt++;
        } else if (vertexQuality == threshold) {
            mediocrePackedSegment[mediocreCnt] = i; mediocreCnt++;
        }
    }
    countInSegment[idx] = cnt; mediocreCountInSegment[idx] = mediocreCnt;
    __syncthreads();
    // compute start indices in the packed buffer:
    if (idx == 0) {
        int numberOfPixelsTooMuch = int(hist[1]);
        startIndexOfSegment[0] = 0;
        for (int i = 1; i < nthreads; i++) {
            startIndexOfSegment[i] = startIndexOfSegment[i-1] + countInSegment[i-1];
        }
        mediocreStartIndexOfSegment[0] = startIndexOfSegment[nthreads-1]+countInSegment[nthreads-1];
        for (int i = 1; i < nthreads; i++) {
            if (numberOfPixelsTooMuch > 0 && mediocreCountInSegment[i-1] > 0) {
                mediocreCountInSegment[i-1] -= numberOfPixelsTooMuch;
                if (mediocreCountInSegment[i-1] < 0) {
                    numberOfPixelsTooMuch += mediocreCountInSegment[i-1];
                    mediocreCountInSegment[i-1] = 0;
                } else {
                    numberOfPixelsTooMuch = 0;
                }
            }
            mediocreStartIndexOfSegment[i] = mediocreStartIndexOfSegment[i-1] + mediocreCountInSegment[i-1];
        }
    }
    __syncthreads();

    int startPackedOffset = startIndexOfSegment[idx];
    for (int i = 0; i < countInSegment[idx]; i++) {
        indexPointer[startPackedOffset+i] = packedSegment[i];
    }
    int mediocreStartPackedOffset = mediocreStartIndexOfSegment[idx];
    for (int i = 0; i < mediocreCountInSegment[idx]; i++) {
        indexPointer[mediocreStartPackedOffset+i] = mediocrePackedSegment[i];
    }
}
*/

__global__ void filterIndexKernel4(float *gradientData, float *hist, int *indexPointer, int nthreads, int pointsPerThread, int nbins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startOffset = idx*pointsPerThread;
    int lastOffset = startOffset + (pointsPerThread-1);
 //   if (lastOffset >= 320*240) return;

    __shared__ int countInSegment[1024];
    __shared__ int startIndexOfSegment[1024];
    __shared__ int mediocreCountInSegment[1024];
    __shared__ int mediocreStartIndexOfSegment[1024];

    int packedSegment[1024];
    int mediocrePackedSegment[1024];

    // pack each segment in parallel and store count
    uint threshold = (uint)hist[0];

    int cnt = 0, mediocreCnt = 0;
    for (int i = startOffset; i <= lastOffset; i++) {
        uint vertexQuality = (uint)(gradientData[i]*(nbins-1)+0.5f);
        if (vertexQuality > threshold) {
            packedSegment[cnt] = i; cnt++;
        } else if (vertexQuality == threshold) {
            mediocrePackedSegment[mediocreCnt] = i; mediocreCnt++;
        }
    }
    countInSegment[idx] = cnt; mediocreCountInSegment[idx] = mediocreCnt;
    __syncthreads();
    // compute start indices in the packed buffer:
    if (idx == 0) {
        int numberOfPixelsTooMuch = int(hist[1]);
        startIndexOfSegment[0] = 0;
        for (int i = 1; i < nthreads; i++) {
            startIndexOfSegment[i] = startIndexOfSegment[i-1] + countInSegment[i-1];
        }
        mediocreStartIndexOfSegment[0] = startIndexOfSegment[nthreads-1]+countInSegment[nthreads-1];
        for (int i = 1; i < nthreads; i++) {
            if (numberOfPixelsTooMuch > 0 && mediocreCountInSegment[i-1] > 0) {
                mediocreCountInSegment[i-1] -= numberOfPixelsTooMuch;
                if (mediocreCountInSegment[i-1] < 0) {
                    numberOfPixelsTooMuch += mediocreCountInSegment[i-1];
                    mediocreCountInSegment[i-1] = 0;
                } else {
                    numberOfPixelsTooMuch = 0;
                }
            }
            mediocreStartIndexOfSegment[i] = mediocreStartIndexOfSegment[i-1] + mediocreCountInSegment[i-1];
        }
    }
    __syncthreads();

    int startPackedOffset = startIndexOfSegment[idx];
    for (int i = 0; i < countInSegment[idx]; i++) {
        indexPointer[startPackedOffset+i] = packedSegment[i];
    }
    int mediocreStartPackedOffset = mediocreStartIndexOfSegment[idx];
    for (int i = 0; i < mediocreCountInSegment[idx]; i++) {
        indexPointer[mediocreStartPackedOffset+i] = mediocrePackedSegment[i];
    }
}

extern "C" void filterIndices4(VertexBuffer2 *vbuffer, float *gradientData, float *histogramDev, int pixelSelectionAmount, int nbins, cudaStream_t stream) {
    assert(vbuffer != NULL && histogramDev != NULL);
    int *indexPointer = (int*)vbuffer->indexDevPtr;
    int nVertices = vbuffer->getVertexCount();

    if (vbuffer->getStride() != VERTEXBUFFER_STRIDE) {
        printf("filterIndices4: vertexbuffer has illegal stride (%d), must be %d!\n",vbuffer->getStride(),VERTEXBUFFER_STRIDE);
        fflush(stdin); fflush(stdout);
        return;
    }

    dim3 cudaBlockSize1(1,1,1);
    dim3 cudaGridSize1(1,1,1);
    seekThreshold256Kernel2<<<cudaGridSize1,cudaBlockSize1,0,stream>>>(histogramDev, pixelSelectionAmount,nbins);
    char buf[512]; sprintf(buf,"seekThresholdKernel2 execution failed, arguments, ptr: %d, npixels: %d\n",(unsigned int)(histogramDev!=NULL),pixelSelectionAmount);
    getLastCudaError(buf);

    int nthreads = 512;
    int pointsPerThread = nVertices/nthreads;
    dim3 cudaBlockSize(nthreads,1,1);
    dim3 cudaGridSize(1,1,1);
    filterIndexKernel4<<<cudaGridSize,cudaBlockSize,0,stream>>>(gradientData,histogramDev,indexPointer,nthreads,pointsPerThread,nbins);
    vbuffer->setElementsCount(pixelSelectionAmount);
}

