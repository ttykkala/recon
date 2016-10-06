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

__global__ void convert2FloatKernel( unsigned char *srcPtr, float *dstPtr, unsigned int pitch)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int rowOffset = yi*pitch;	
	int offsetR = 3*xi+0 + rowOffset;
	int offsetG = 3*xi+1 + rowOffset;
	int offsetB = 3*xi+2 + rowOffset;
	dstPtr[offsetR] = float(srcPtr[offsetR])/255.0f;
	dstPtr[offsetG] = float(srcPtr[offsetG])/255.0f;
	dstPtr[offsetB] = float(srcPtr[offsetB])/255.0f;
}

extern "C" void convert2FloatCuda(Image2 *rgbInput, Image2 *imRGB)
{
	if (rgbInput == 0 || rgbInput->devPtr == NULL || imRGB == 0 || imRGB->devPtr == NULL) return;
	unsigned char *srcPtr = (unsigned char*)rgbInput->devPtr;
	float *dstPtr= (float*)imRGB->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(rgbInput->width/cudaBlockSize.x,rgbInput->height/cudaBlockSize.y,1);
	convert2FloatKernel<<<cudaGridSize,cudaBlockSize,0,rgbInput->cudaStream>>>(srcPtr,dstPtr,(unsigned int)rgbInput->width*3);
}

