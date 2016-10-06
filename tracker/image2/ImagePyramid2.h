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

#define __ImagePyramid_H__

#include "Image2.h"
#include <helper_functions.h>
/*#include <windows.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
*/
//#include <opencv2\opencv.hpp>
#include <assert.h>


#define MAX_PYRAMID_LEVELS 10

class ImagePyramid2 {
private:
public:
	int nLayers;
	Image2 pyramid[MAX_PYRAMID_LEVELS];
	ImagePyramid2();
	~ImagePyramid2();
	Image2 *getImagePtr(int index) { return &pyramid[index]; }
	Image2 &getImageRef(int index) { return pyramid[index]; }
	void releaseData();
	void createLayers(int gpuStatus);
	void lock();
	void unlock();
	void setName(const char *name);
	void setStream(cudaStream_t stream);
	virtual void gaussianBlur( Image2 *img, int size);
        void updateCudaArrays();
	void updateLayerHires(Image2 *hires, Image2 *target,int layer);

    virtual void updateLayers(cudaStream_t stream=0);
	//void updateLayersCuda();
    void updateLayersThreshold();

	void thresholdLayers(int threshold);

	int loadPyramid(char *fileName, int nLayers=1, int targetResoX=0, bool colorFlag = false, unsigned int gpuStatus=CREATE_GPU_TEXTURE);
	void correctLighting(Image2 &src, Image2 &dst);

	int updatePyramid(char *fileName, bool lightingCorrectionFlag=false, bool whiteFlag=false);

	int updatePyramid(void *data, bool lightingCorrectionFlag=false, bool whiteFlag=false);

    int createHdrPyramid(unsigned int width, unsigned int height, int nchannels, int nLayers=1, bool showDynamicRange=true,unsigned int gpuStatus=CREATE_GPU_TEXTURE, bool renderable=true);
    int createPyramid(unsigned int width, unsigned int height, int nChannels, int nLayers=1, unsigned int gpuStatus=CREATE_GPU_TEXTURE,bool writeDiscard=false, bool renderable=true);

	virtual int downSample2( Image2 *img );
	void downSampleThreshold2(Image2 *img, Image2 *targetImg);

	void copyPyramid(ImagePyramid2 *srcPyramid);

	void zeroPyramid();

	virtual int downSample2( Image2 *img, Image2 *img2);
};

class DisparityPyramid : public ImagePyramid2
{
public:
	DisparityPyramid() {};
	~DisparityPyramid() {};
    void updateLayers(cudaStream_t /*stream*/) {
		//for (int i = 1; i < nLayers; i++)
		//	downSample2(&pyramid[i-1],&pyramid[i]);
	}
	int downSample2( Image2 *img );

	int downSample2( Image2 *img, Image2 *img2);
};
