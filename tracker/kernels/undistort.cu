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
#include <cwchar>

/*
__global__ void undistortHdrRGBKernel( unsigned char *srcPtr, float *dstPtr, unsigned int width, unsigned int height, float *calibDataDev)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width;
	int widthRGB = width*3; 
	float K[9]; 
	float iK[9]; 
	float kc[5];
	for (int i = 0; i < 9; i++) { 
		K[i]  = calibDataDev[i]; 
		iK[i] = calibDataDev[i+9]; 
	}
	for (int i = 0; i < 5; i++) kc[i] = calibDataDev[i+9*2];
	// convert to float coordinates
	float pu[2]; pu[0] = (float)xi; pu[1] = (float)yi;
	float pd[2];
	pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
	pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
	// distort point
	float r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); float r4 = r2*r2; float r6 = r4 * r2; 
	float radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	pd[0] *= radialDist;
	pd[1] *= radialDist;
	// define sampling point in distorted image
	float p[2]; 
	p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
	p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];
	// bi-linear interpolation
	int xdi = (int)p[0];
	int ydi = (int)p[1];
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offset2 = xdi*3+ydi*widthRGB;
		unsigned short grayVal =
		float v0 = srcPtr[offset2];          float v1 = srcPtr[offset2+3];
		float v2 = srcPtr[offset2+widthRGB]; float v3 = srcPtr[offset2+widthRGB+3];


		float fx = p[0] - xdi;
		float fy = p[1] - ydi;
		dstPtr[offset] = ((1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3)/255.0f; 
	} else	{
		dstPtr[offset] = 0.0f;
	}
}*/

__global__ void undistortHdrKernel( float *srcPtr, float *dstPtr, unsigned int width, unsigned int height, float *calibDataDev)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width; 
	float K[9]; 
	float iK[9]; 
	float kc[5];
	for (int i = 0; i < 9; i++) { 
		K[i]  = calibDataDev[i]; 
		iK[i] = calibDataDev[i+9]; 
	}
	for (int i = 0; i < 5; i++) kc[i] = calibDataDev[i+9*2];
	// convert to float coordinates
	float pu[2]; pu[0] = (float)xi; pu[1] = (float)yi;
	float pd[2];
	pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
	pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
	// distort point
	float r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); float r4 = r2*r2; float r6 = r4 * r2; 
	float radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	pd[0] *= radialDist;
	pd[1] *= radialDist;
	// define sampling point in distorted image
	float p[2]; 
	p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
	p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];
	// bi-linear interpolation
	int xdi = (int)p[0];
	int ydi = (int)p[1];
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offset2 = xdi+ydi*width;
		float v0 = srcPtr[offset2];       float v1 = srcPtr[offset2+1];
		float v2 = srcPtr[offset2+width]; float v3 = srcPtr[offset2+width+1];
		float fx = p[0] - xdi;
		float fy = p[1] - ydi;
		dstPtr[offset] = ((1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3)/255.0f; 
	} else	{
		dstPtr[offset] = 0.0f;
	}
}

__global__ void undistortHdrRGBKernel( unsigned char *srcPtr, float *dstPtr, unsigned int width, unsigned int height, float *calibDataDev)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	float K[9]; 
	float iK[9]; 
	float kc[5];
	int pitch = width*3;
	int offsetR = 3*xi + yi*pitch;
 	int offsetG = offsetR+1;
 	int offsetB = offsetR+2; 

	for (int i = 0; i < 9; i++) { 
		K[i]  = calibDataDev[i]; 
		iK[i] = calibDataDev[i+9]; 
	}
	for (int i = 0; i < 5; i++) kc[i] = calibDataDev[i+9*2];
	// convert to float coordinates
	float pu[2]; pu[0] = (float)xi; pu[1] = (float)yi;
	float pd[2];
	pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
	pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
	// distort point
	float r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); float r4 = r2*r2; float r6 = r4 * r2; 
	float radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	pd[0] *= radialDist;
	pd[1] *= radialDist;
	// define sampling point in distorted image
	float p[2]; 
	p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
	p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];
	// bi-linear interpolation
	int xdi = (int)p[0];
	int ydi = (int)p[1];
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offsetR2 = 3*xdi+0 + ydi*pitch;
		int offsetG2 = offsetR2+1;
		int offsetB2 = offsetR2+2;

		float fx = p[0] - xdi;
		float fy = p[1] - ydi;

		float a = (1-fx)*(1-fy);
		float b = fx*(1-fy);
		float c = (1-fx)*fy;
		float d = fx*fy; 
		
		float v0 = float(srcPtr[offsetR2])/255.0f; float v1 = float(srcPtr[offsetR2+3])/255.0f;
		float v2 = float(srcPtr[offsetR2+pitch])/255.0f; float v3 = float(srcPtr[offsetR2+pitch+3])/255.0f;
		dstPtr[offsetR] = a*v0 + b*v1 + c*v2 + d*v3;

		v0 = float(srcPtr[offsetG2])/255.0f;       v1 = float(srcPtr[offsetG2+3])/255.0f;
		v2 = float(srcPtr[offsetG2+pitch])/255.0f; v3 = float(srcPtr[offsetG2+pitch+3])/255.0f;
		dstPtr[offsetG] = a*v0 + b*v1 + c*v2 + d*v3;

		v0 = float(srcPtr[offsetB2])/255.0f;       v1 = float(srcPtr[offsetB2+3])/255.0f;
		v2 = float(srcPtr[offsetB2+pitch])/255.0f; v3 = float(srcPtr[offsetB2+pitch+3])/255.0f;
		dstPtr[offsetB] = a*v0 + b*v1 + c*v2 + d*v3; 
	} else	{
		dstPtr[offsetR] = 0.0f;
		dstPtr[offsetG] = 0.0f;
		dstPtr[offsetB] = 0.0f;
	}
}

/*
__global__ void undistortKernel( unsigned char *srcPtr, unsigned char *dstPtr, unsigned int width, unsigned int height, float *calibDataDev)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width; 
	__shared__ float K[9]; 
	__shared__ float iK[9]; 
	__shared__ float kc[5];
	for (int i = 0; i < 9; i++) { 
		K[i]  = calibDataDev[i]; 
		iK[i] = calibDataDev[i+9]; 
	}
	for (int i = 0; i < 5; i++) kc[i] = calibDataDev[i+9*2];
	// convert to float coordinates
	float pu[2]; pu[0] = (float)xi; pu[1] = (float)yi;
	float pd[2];
	pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
	pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
	// distort point
	float r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); float r4 = r2*r2; float r6 = r4 * r2; 
	float radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	pd[0] *= radialDist;
	pd[1] *= radialDist;
	// define sampling point in distorted image
	float p[2]; 
	p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
	p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];
	// bi-linear interpolation
	int xdi = (int)p[0];
	int ydi = (int)p[1];
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offset2 = xdi+ydi*width;
		float v0 = (float)srcPtr[offset2];       float v1 = (float)srcPtr[offset2+1];
		float v2 = (float)srcPtr[offset2+width]; float v3 = (float)srcPtr[offset2+width+1];
		float fx = p[0] - xdi;
		float fy = p[1] - ydi;
		dstPtr[offset] = (unsigned char)((1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3); 
	} else	{
		dstPtr[offset] = 0;
	}
}
*/

extern "C" void undistortCuda(Image2 *distorted, Image2 *undistorted, float *calibDataDev)
{
	if (distorted == 0 || distorted->devPtr == NULL || undistorted == 0 || undistorted->devPtr == NULL || calibDataDev == NULL) return;
	float *srcPtr = (float*)distorted->devPtr;
	float *dstPtr= (float*)undistorted->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(distorted->width/cudaBlockSize.x,distorted->height/cudaBlockSize.y,1);
	undistortHdrKernel<<<cudaGridSize,cudaBlockSize,0,distorted->cudaStream>>>(srcPtr,dstPtr,(unsigned int)distorted->width,(unsigned int)distorted->height,calibDataDev);
}


extern "C" void undistortRGBCuda(Image2 *distortedRGB, Image2 *undistortedRGB, float *calibDataDev)
{
	if (distortedRGB == 0 || distortedRGB->devPtr == NULL || undistortedRGB == 0 || undistortedRGB->devPtr == NULL || calibDataDev == NULL) return;
	unsigned char *srcPtr = (unsigned char*)distortedRGB->devPtr;
	float *dstPtr= (float*)undistortedRGB->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(distortedRGB->width/cudaBlockSize.x,distortedRGB->height/cudaBlockSize.y,1);
	undistortHdrRGBKernel<<<cudaGridSize,cudaBlockSize,0,distortedRGB->cudaStream>>>(srcPtr,dstPtr,(unsigned int)distortedRGB->width,(unsigned int)distortedRGB->height,calibDataDev);
}



/*
extern "C" void undistortFromRGBCuda(Image2 *distortedRGB, Image2 *undistorted, float *calibDataDev)
{
	if (distortedRGB == 0 || distortedRGB->devPtr == NULL || undistorted == 0 || undistorted->devPtr == NULL || calibDataDev == NULL) return;
	unsigned char *srcPtr = (unsigned char*)distorted->devPtr;
	float *dstPtr= (float*)undistorted->devPtr;
	dim3 cudaBlockSize(32,32,1);
	dim3 cudaGridSize(undistorted->width/cudaBlockSize.x,undistorted->height/cudaBlockSize.y,1);
	undistortHdrRGBKernel<<<cudaGridSize,cudaBlockSize,0,distortedRGB->cudaStream>>>(srcPtr,dstPtr,(unsigned int)undistorted->width,(unsigned int)undistorted->height,calibDataDev);
}
*/
