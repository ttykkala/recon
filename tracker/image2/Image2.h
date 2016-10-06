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

#pragma once

#define __Image_H__

#include <stdio.h>
#include <string.h>

#define CREATE_GPU_TEXTURE 0xf0f0f0f0
#define NO_GPU_TEXTURE     0xfff0f0f0
#define ONLY_GPU_TEXTURE   0xfffff0f0

// forward declare cuda stream type ( no need for headers)
struct CUstream_st;
typedef CUstream_st *cudaStream_t;
	
class Image2 {
public:
	unsigned char *data;
	unsigned int pitch;
	unsigned int channels;
	unsigned int width;
	unsigned int height;
	unsigned int pbo; // pixel buffer object, cuda interop requires this
	void *cuda_pbo_resource; // cuda handle for texture resource
	unsigned int texID;
	unsigned int type;
	unsigned int shiftIntensity;
	bool hdr;
	bool onlyGPUFlag;
	bool extDataFlag;
	bool showDynamicRange;
	char imageName[512];
	void *cArray; // NOT IN USE: cudaArray is also allocated for temporary copies -> fast interpolation of data possible
	void *devPtr;
    bool renderable; // determines whether OpenGL rendering supported or not -> this saves cuda interop delays when false
	cudaStream_t cudaStream;
	Image2();
	Image2(void *extCudaDevPtr, int width, int height, int pitch, int channels, bool hdr);
	~Image2();
	void bind();
    void createTexture(void *extData = NULL, bool renderable = true);
	void updateTexture(void *extData = NULL);  // copies texture data from host memory (local/extData)
	void updateTextureInternal(void *devData, bool updateArrayFlag=false); // copies texture data from cuda dev ptr
	void lockCudaPtr(void **cudaPtr);
	void setWriteDiscardFlag();
	void unlockCudaPtr();
	void releaseData();
    void releaseGPUData();
	void updateCudaArray();
	void updateCudaArrayInternal();
	void setStream(cudaStream_t stream) { this->cudaStream = stream; }
	void *lock();
	void unlock();
	void setName(const char *name) { strcpy(imageName,name); }
private:
	void createCudaArray();
	void releaseCudaArray();
};

char *loadPNG(const char *name, unsigned int *width, unsigned int *height, unsigned int *nChannels, unsigned int *pitch, bool flipY);
int loadImage(const char *fileName, Image2 *img, unsigned int texID=CREATE_GPU_TEXTURE, bool gray=true, bool flipY=false);
int uploadImage(Image2 *img);
int uploadImage(Image2 *img, unsigned int width, unsigned int height, unsigned int channels, unsigned int pitch, unsigned char *data, bool hdr = false);
int convertToGray(unsigned char **raw, unsigned int width, unsigned int height, unsigned int *channels);
//int downSample3x3(Image *img);
//int downSample3x3( Image *img, Image *img2);
int createImage(unsigned char *data, int width, int height, int channels, int pitch, Image2 *img, unsigned int texID = CREATE_GPU_TEXTURE, bool renderable=true);
int createHdrImage( float *initData,int width, int height, int channels,Image2 *img, unsigned int texID = CREATE_GPU_TEXTURE,bool showDynamicRange=true, bool renderable=true);
int saveImage(const char *fileName, Image2 *img);
int savePNG(unsigned char *data, int width, int height, int nChannels, int pitch, const char *fileName);
unsigned char interpolatePixel(Image2 *img, float x, float y);
unsigned char interpolatePixelFast( Image2 *img, int xf, int yf );
float interpolatePixel2F(Image2 *img, float x, float y);
void interpolateRGBPixel(unsigned char *rgb, int width, int height, float x, float y, unsigned char *colorR, unsigned char *colorG, unsigned char *colorB);
void interpolateRGBAPixel(Image2 *img, float x, float y, unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a);
void interpolateRGBAPixel(Image2 *img, float x, float y, int *r, int *g, int *b, int *a);
float interpolateFloatPixel(Image2 *img, float x, float y);
float interpolateFloatPixel(float *depthMapRow0, float *depthMapRow1, float x, float y);
float interpolateFloatPixelZeroCheck(Image2 *img, float x, float y, bool &validFlag);
float average3x3f(Image2 *img, float x, float y);
void getRGBAPixel(Image2 *img, int xi, int yi, unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a);
void getPixel(Image2 *img, int xi, int yi, unsigned char *v);

int interpolatePixelSSE(unsigned char *data, int width, int height, int pitch, float xx, float yy);
