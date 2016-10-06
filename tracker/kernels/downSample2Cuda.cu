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

#include <image2/ImagePyramid2.h>
//#include "Image/ImagePyramid.h"

//#include <types.h>

texture<unsigned char, 2, cudaReadModeNormalizedFloat> texC;

// Kernel that executes on the CUDA device	
__global__ void fastDownSampleKernel(unsigned char *hires, int width2, unsigned char *lowres, int width)
{
	unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yoffset = (blockIdx.y*blockDim.y+threadIdx.y)*width;
	int hiresoffset = (x<<1)+(yoffset<<2);
	int offset = x + yoffset; 
	lowres[offset] = (hires[hiresoffset]+hires[hiresoffset+1]+hires[hiresoffset+width2]+hires[hiresoffset+width2+1])>>2; 
}

// Kernel that executes on the CUDA device	
__global__ void fastDownSampleHdrKernel(float *hires, int width2, float *lowres, int width)
{
	unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yoffset = (blockIdx.y*blockDim.y+threadIdx.y)*width;
	int hiresoffset = (x<<1)+(yoffset<<2);
	int offset = x + yoffset; 
	lowres[offset] = (hires[hiresoffset]+hires[hiresoffset+1]+hires[hiresoffset+width2]+hires[hiresoffset+width2+1])/4.0f; 
}
extern "C" void downSample2Pyramid(ImagePyramid2 &pyramid, cudaStream_t stream)
{
	/*
	cudaArray *hiresArray = (cudaArray*)hires->cArray;
	// set texture parameters (default)
	texC.addressMode[0] = cudaAddressModeClamp;
	texC.addressMode[1] = cudaAddressModeClamp;
	texC.filterMode = cudaFilterModeLinear;
    texC.normalized = false; // do not normalize coordinates
	// bind the array to the texture
//	cudaStreamSynchronize(
	cudaBindTextureToArray(texC, hiresArray);*/
	for (int i = 1; i < pyramid.nLayers; i++) {
		Image2 *hires = pyramid.getImagePtr(i-1);
		Image2 *lowres = pyramid.getImagePtr(i);
		dim3 cudaBlockSize(32,30,1);					// 32x30 tiles produce 10x8 grid evenly for 320x240, and 5x4 for 160x120
		if (lowres->width == 80) { 
			cudaBlockSize.x = 16; cudaBlockSize.y = 12; // 16x12 tiles produce 5x5 grid that match with 80x60
		} else if (lowres->width == 40) {
			cudaBlockSize.x = 8; cudaBlockSize.y = 6; // 8x6 tiles produce 5x5 grid that matches with 40x30
		}
		dim3 cudaGridSize(lowres->width/cudaBlockSize.x,lowres->height/cudaBlockSize.y,1);
		//unsigned int size = lowres->width*lowres->height;
		if (hires->hdr) {
			float *hiresPtr = (float*)hires->devPtr;
			float *lowresPtr = (float*)lowres->devPtr;	
            fastDownSampleHdrKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(hiresPtr,hires->width,lowresPtr,lowres->width);
		} else {
			unsigned char *hiresPtr = (unsigned char *)hires->devPtr;
			unsigned char *lowresPtr = (unsigned char *)lowres->devPtr;	
            fastDownSampleKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(hiresPtr,hires->width,lowresPtr,lowres->width);
		}
	}
//	cudaThreadSynchronize();
}

