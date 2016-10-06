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


#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/glew.h>// GLEW Library
//#include <GL/gl.h>	// OpenGL32 Library
//#include <GL/glu.h>	// GLU32 Library
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_funcs.h>
#include <opencv2/opencv.hpp>
#include "ImagePyramid2.h"

void copyPyramid(ImagePyramid2 *dstPyramid, ImagePyramid2 *srcPyramid) {
	assert(dstPyramid->nLayers == srcPyramid->nLayers);
	if (dstPyramid->nLayers != srcPyramid->nLayers) return;

	for (int i = 0; i < dstPyramid->nLayers; i++) {
		Image2 *src = srcPyramid->getImagePtr(i);
		Image2 *dst = dstPyramid->getImagePtr(i);
		assert(dst->data != NULL);	
		unsigned char *data = NULL;
		src->lockCudaPtr((void**)&data);
		cudaMemcpy(dst->data, data, src->height*src->pitch, cudaMemcpyDeviceToHost);
		src->unlockCudaPtr();
		uploadImage(dst);
	}
//	dstPyramid->updateLayers();
}

void zeroPyramid(ImagePyramid2 *pyramid) {
	for (int i = 0; i < pyramid->nLayers; i++) {
		Image2 *img = pyramid->getImagePtr(i);
		memset(img->data,0,img->pitch*img->height);
		uploadImage(img);
	}
}

void ImagePyramid2::createLayers( int gpuStatus )
{
	if (!pyramid[0].hdr) {
		for (int i = 1; i < nLayers; i++)
			createImage(NULL,pyramid[i-1].width/2,pyramid[i-1].height/2,pyramid[i-1].channels,pyramid[i-1].pitch/2,&pyramid[i],gpuStatus);
	} else {
		for (int i = 1; i < nLayers; i++)
			createHdrImage(NULL,pyramid[i-1].width/2,pyramid[i-1].height/2,pyramid[i-1].channels,&pyramid[i],gpuStatus);		
	}
}

void ImagePyramid2::updateCudaArrays() {
    for (int i = 0; i < nLayers; i++) {
            Image2 *img = getImagePtr(i);
            if (img->devPtr != NULL) img->updateCudaArrayInternal();
            else {
                printf("%s cuda array update failed, because data is not locked!\n",img->imageName);
            }
    }
}

void ImagePyramid2::gaussianBlur( Image2 *img, int size )
{
	if (img->channels > 1) return;
	cv::Mat oldmatA(img->height,img->width,CV_8U);
	cv::Mat oldmatB(img->height,img->width,CV_8U);	
	memcpy(oldmatA.ptr(),img->data,img->pitch*img->height);
	cv::blur(oldmatA,oldmatB,cv::Size(size,size));
	memcpy(img->data,oldmatB.ptr(),img->pitch*img->height);
	//cvReleaseMat(&oldmatA);
	//cvReleaseMat(&oldmatB);
}

void ImagePyramid2::updateLayerHires( Image2 *hires, Image2 *target,int layer )
{
	unsigned char *data = (unsigned char*)hires->data;
	unsigned char *lowresData = target->data;
	float scaleDown = 1.0f/(1<<layer);
	float scaleUp = float(1<<layer);
	int resoX = hires->width>>layer;
	int resoY = hires->height>>layer;
	float a = 1<<layer;
	float b = 0.5*(a-1);

	int imageOffset = 0;
	for (int j = 0; j < resoY; j++) {
		for (int i = 0; i < resoX; i++,imageOffset++) { 
			float xf = a*float(i)+b;
			float yf = a*float(j)+b;
			lowresData[imageOffset] = interpolatePixel(hires, xf,yf);
		}
	}
	uploadImage(target);
}

void ImagePyramid2::updateLayers(cudaStream_t stream)
{
	if (pyramid[0].onlyGPUFlag) {
        downSample2Pyramid(*this,stream);
/*		for (int i = 1; i < nLayers; i++) {
			//downSample2(&pyramid[i-1],&pyramid[i]);			
			downSample2Cuda2(&pyramid[i-1],&pyramid[i]); 
		}*/
	} else {
		for (int i = 1; i < nLayers; i++) {
			//if (pyramid[0].channels > 1) 
			{
                downSample2(&pyramid[i-1],&pyramid[i]);
				uploadImage(&pyramid[i]);
			} //else updateLayerHires(&pyramid[0], &pyramid[i], i);
		}
	}
}

void ImagePyramid2::thresholdLayers( int threshold )
{
	assert((pyramid[0].channels == 1) && (pyramid[0].hdr == false));
	for (int i = 1; i < nLayers; i++) {
		unsigned char *data = pyramid[i].data;
		int size = pyramid[i].width*pyramid[i].height;
		for (int k = 0; k < size; k++) { 
			if (data[k] < threshold) { data[k] = 0; }
			else { data[k] = 255; }
		}
	}
}

void ImagePyramid2::updateLayersThreshold()
{
	for (int i = 1; i < nLayers; i++)
		downSampleThreshold2(&pyramid[i-1],&pyramid[i]);
}

int ImagePyramid2::loadPyramid( char *fileName, int nLayers/*=1*/, int targetResoX/*=0*/, bool colorFlag /*= false*/, unsigned int gpuStatus/*=CREATE_GPU_TEXTURE*/ )
{
	assert(nLayers > 0);
	assert(nLayers < MAX_PYRAMID_LEVELS);
	this->nLayers = nLayers;
	loadImage(fileName,&pyramid[0],gpuStatus,!colorFlag); 
	if (targetResoX > 0)
		while (int(pyramid[0].width) > targetResoX) { downSample2(&pyramid[0]); }
		createLayers(gpuStatus);
		updateLayers();
		return 1;
}

void ImagePyramid2::correctLighting( Image2 &src, Image2 &dst )
{
	unsigned char *srcData = src.data;
	unsigned char *dstData = dst.data;

	unsigned int hist[256]; memset(hist,0,256*sizeof(int));
	unsigned int mass = 0;
	int offset = 0;
	for (int j  = 0; j < src.height; j++) {
		for (int i  = 0; i < src.width; i++,offset++) {
			unsigned char v0 = srcData[offset];
			if (v0 == 0) continue;
			hist[v0]++;
			mass++;
		}
	}
	int desiredMass = mass/2;
	int currentMass = 0;
	int threshold = 0;
	for (int i = 0; i < 256; i++) {
		currentMass += hist[i];
		if (currentMass >= desiredMass) { threshold = i; break;}
	}
	offset = 0;
	for (int j  = 0; j < src.height; j++) {
		for (int i  = 0; i < src.width; i++,offset++) {
			int v0 = srcData[offset];
			if (v0 == 0) dstData[offset] = 0;
			else {
				int v1 = MIN(MAX(v0 - threshold + 128,0),255);
				dstData[offset] = (unsigned char)v1;
			}
		}
	}
	//		memcpy(dst.data,src.data,src.pitch*src.height);
}

void ImagePyramid2::setName(const char *name)
{
	char buf[512];
	for (int i = 0; i < nLayers; i++) {
		sprintf(buf,"%s.%d",name,i);
		pyramid[i].setName(buf);
	}

}

void ImagePyramid2::releaseData()
{
	for (int i = nLayers-1; i >= 0; i--) pyramid[i].releaseData();
}

int ImagePyramid2::updatePyramid( char *fileName, bool lightingCorrectionFlag/*=false*/, bool whiteFlag/*=false*/ )
{
	assert(nLayers > 0);
	assert(nLayers < MAX_PYRAMID_LEVELS);
	int targetResoX = pyramid[0].width;
	bool colorFlag = false;
	if (pyramid[0].channels == 4) colorFlag = true;
	Image2 tmp;
	loadImage(fileName,&tmp,NO_GPU_TEXTURE,!colorFlag); 
	while (int(tmp.width) > targetResoX) { downSample2(&tmp); }
	if (!lightingCorrectionFlag) memcpy(pyramid[0].data,tmp.data,tmp.pitch*tmp.height);
	else {
		correctLighting(tmp,pyramid[0]);
		cv::Mat oldmatA(tmp.height,tmp.width,CV_8U);
		cv::Mat oldmatB(tmp.height,tmp.width,CV_8U);	
		memcpy(oldmatA.ptr(),pyramid[0].data,sizeof(char)*tmp.width*tmp.height);
		cv::medianBlur(oldmatA,oldmatB,3);
		//	cvSmooth(oldmatB,oldmatA,CV_GAUSSIAN/*CV_MEDIAN*/,5);
		memcpy(pyramid[0].data,oldmatB.ptr(),sizeof(char)*tmp.width*tmp.height);
	//	cvReleaseMat(&oldmatA);
	//	cvReleaseMat(&oldmatB);
	}
	if (whiteFlag) {
		int size = pyramid[0].width*pyramid[0].height;
		for (int i = 0; i < size; i++) if (pyramid[0].data[i] != 0) pyramid[0].data[i] = 255;
	}
	uploadImage(&pyramid[0]);
	updateLayers();
	return 1;
}

int ImagePyramid2::updatePyramid( void *data, bool lightingCorrectionFlag/*=false*/, bool whiteFlag/*=false*/ )
{
	assert(nLayers > 0);
	assert(nLayers < MAX_PYRAMID_LEVELS);
	int targetResoX = pyramid[0].width;
	bool colorFlag = false;
	if (pyramid[0].channels == 4) colorFlag = true;
	pyramid[0].updateTexture(data);
	/*		
	uploadImage(pyrami)
	Image tmp; 
	loadImage(fileName,&tmp,NO_GPU_TEXTURE,!colorFlag); 
	while (int(tmp.width) > targetResoX) { downSample2(&tmp); }
	if (!lightingCorrectionFlag) memcpy(pyramid[0].data,tmp.data,tmp.pitch*tmp.height);
	else {
	correctLighting(tmp,pyramid[0]);
	CvMat *oldmatA = cvCreateMat(tmp.height,tmp.width,CV_8U);
	CvMat *oldmatB = cvCreateMat(tmp.height,tmp.width,CV_8U);	
	memcpy(oldmatA->data.ptr,pyramid[0].data,sizeof(char)*tmp.width*tmp.height);
	cvSmooth(oldmatA,oldmatB,CV_MEDIAN,3);
	//	cvSmooth(oldmatB,oldmatA,CV_GAUSSIAN,5);
	memcpy(pyramid[0].data,oldmatB->data.ptr,sizeof(char)*tmp.width*tmp.height);
	cvReleaseMat(&oldmatA);
	cvReleaseMat(&oldmatB);
	}
	if (whiteFlag) {
	int size = pyramid[0].width*pyramid[0].height;
	for (int i = 0; i < size; i++) if (pyramid[0].data[i] != 0) pyramid[0].data[i] = 255;
	}
	uploadImage(&pyramid[0]);*/
	//updateLayers();
	return 1;
}

int ImagePyramid2::createHdrPyramid( unsigned int width, unsigned int height, int nchannels, int nLayers/*=1*/, bool showDynamicRange/*=true*/,unsigned int gpuStatus/*=CREATE_GPU_TEXTURE*/, bool renderableBuffer)
{
	assert(nLayers > 0);
	assert(nLayers < MAX_PYRAMID_LEVELS);
	this->nLayers = nLayers;
	for (int i = 0; i < nLayers; i++)
        createHdrImage(NULL,width>>i,height>>i,nchannels,&pyramid[i],gpuStatus,showDynamicRange, renderableBuffer);
	return 1;
}

int ImagePyramid2::createPyramid( unsigned int width, unsigned int height, int nChannels, int nLayers/*=1*/, unsigned int gpuStatus/*=CREATE_GPU_TEXTURE*/,bool writeDiscard/*=false*/, bool renderableBuffer)
{
	assert(nLayers > 0);
	assert(nLayers < MAX_PYRAMID_LEVELS);
	this->nLayers = nLayers;
	for (int i = 0; i < nLayers; i++) {
        createImage(NULL,width>>i,height>>i,nChannels,(width>>i)*nChannels,&pyramid[i],gpuStatus,renderableBuffer);
		if (writeDiscard) pyramid[i].setWriteDiscardFlag();
	}
	return 1;
}

int ImagePyramid2::downSample2( Image2 *img )
{
	if (img->channels != 1 && img->channels != 4) assert(0);
	int newWidth = img->width/2;
	int newHeight = img->height/2;

	if (img->hdr) {
		float *dst = new float[newWidth*newHeight];
		float *src = (float*)img->data;
		int offset = 0;
		for (int j = 0; j < newHeight; j++) {
			for (int i = 0; i < newWidth; i++,offset++) {
				int offset2 = i*2+j*2*img->width;
				dst[offset] = (src[offset2] + src[offset2+1] + src[offset2+img->width] + src[offset2+1+img->width])/8;
			}
		}
		uploadImage(img, newWidth, newHeight, img->channels, newWidth*sizeof(float), (unsigned char*)dst, true);
		delete[] dst;
	} else {
		unsigned char *data = new unsigned char[newWidth*newHeight*img->channels];
		if (img->channels == 1) {
			int offset = 0;
			for (int j = 0; j < newHeight; j++) {
				for (int i = 0; i < newWidth; i++,offset++) {
					int offset2 = i*2+j*2*img->pitch;
					data[offset] = (unsigned char)((img->data[offset2] + img->data[offset2+1] + img->data[offset2+img->width] + img->data[offset2+1+img->width])/4);
				}
			}
		} else {
			int offset = 0;
			for (int j = 0; j < newHeight; j++) {
				for (int i = 0; i < newWidth; i++,offset+=4) {
					int offset2 = i*4*2+j*2*img->pitch;
					data[offset+0] = (unsigned char)((img->data[offset2] + img->data[offset2+4] + img->data[offset2+img->pitch] + img->data[offset2+4+img->pitch])/4);
					data[offset+1] = (unsigned char)((img->data[offset2+1] + img->data[offset2+4+1] + img->data[offset2+img->pitch+1] + img->data[offset2+4+img->pitch+1])/4);
					data[offset+2] = (unsigned char)((img->data[offset2+2] + img->data[offset2+4+2] + img->data[offset2+img->pitch+2] + img->data[offset2+4+img->pitch+2])/4);
					data[offset+3] = (unsigned char)((img->data[offset2+3] + img->data[offset2+4+3] + img->data[offset2+img->pitch+3] + img->data[offset2+4+img->pitch+3])/4);
				}
			}
		}
		uploadImage(img, newWidth, newHeight, img->channels, newWidth*img->channels, data, false);
		delete[] data;
	}
	return 1;
}

int ImagePyramid2::downSample2( Image2 *img, Image2 *img2 )
{
	if (img->channels != 1 && img->channels != 4) assert(0);

	int newWidth = img->width/2;
	int newHeight = img->height/2;

	if (img->hdr) {
		// NOTICE: disparity image assumed! dvalue divided by 2!
		float *dst = new float[newWidth*newHeight];
		float *src = (float*)img->data;
		assert(img->channels==1);
		int offset = 0;
		for (int j = 0; j < newHeight; j++) {
			for (int i = 0; i < newWidth; i++,offset++) {
				int offset2 = i*2+j*2*img->width;
				dst[offset] = (src[offset2] + src[offset2+1] + src[offset2+img->width] + src[offset2+1+img->width])/4;
			}
		}
		if (img2->width == newWidth && img2->height == newHeight && img2->channels == img->channels && img2->hdr == img->hdr) {
			memcpy(img2->data,dst,img2->pitch*img2->height);
		} else {
			assert(0);
		}
		delete[] dst;
	} else {
		unsigned char *data = new unsigned char[newWidth*newHeight*img->channels];
		if (img->channels == 1) {
			int offset = 0;
			for (int j = 0; j < newHeight; j++) {
				for (int i = 0; i < newWidth; i++,offset++) {
					int offset2 = i*2+j*2*img->width;
					data[offset] = (unsigned char)((img->data[offset2] + img->data[offset2+1] + img->data[offset2+img->width] + img->data[offset2+1+img->width])/4);
				}
			}
		}else {
			int offset = 0;
			for (int j = 0; j < newHeight; j++) {
				for (int i = 0; i < newWidth; i++,offset+=4) {
					int offset2 = i*4*2+j*2*img->pitch;
					data[offset+0] = (unsigned char)((img->data[offset2] + img->data[offset2+4] + img->data[offset2+img->pitch] + img->data[offset2+4+img->pitch])/4);
					data[offset+1] = (unsigned char)((img->data[offset2+1] + img->data[offset2+4+1] + img->data[offset2+img->pitch+1] + img->data[offset2+4+img->pitch+1])/4);
					data[offset+2] = (unsigned char)((img->data[offset2+2] + img->data[offset2+4+2] + img->data[offset2+img->pitch+2] + img->data[offset2+4+img->pitch+2])/4);
					data[offset+3] = (unsigned char)((img->data[offset2+3] + img->data[offset2+4+3] + img->data[offset2+img->pitch+3] + img->data[offset2+4+img->pitch+3])/4);
				}
			}
		}
		if (img2->width == newWidth && img2->height == newHeight && img2->channels == img->channels) {
			memcpy(img2->data,data,img2->pitch*img2->height);
		} else {
			assert(0);
			//				unsigned int gpuStatus = CREATE_GPU_TEXTURE;
			//				if (img2->onlyGPUFlag) gpuStatus = ONLY_GPU_TEXTURE;						
			//				createImage(data,newWidth,newHeight,img->channels,newWidth*img->channels,img2,gpuStatus);
		}
		delete[] data;
	}
	return 1;
}
void ImagePyramid2::downSampleThreshold2( Image2 *img, Image2 *targetImg )
{
	if (img->channels != 1 && img->channels != 4) assert(0);

	int newWidth = img->width/2;
	int newHeight = img->height/2;
	assert(img->channels == 1);
	if (img->hdr) {
		// NOTICE: disparity image assumed! dvalue divided by 2!
		float *dst = new float[newWidth*newHeight];
		float *src = (float*)img->data;
		assert(img->channels==1);
		int offset = 0;
		for (int j = 0; j < newHeight; j++) {
			for (int i = 0; i < newWidth; i++,offset++) {
				int offset2 = i*2+j*2*img->width;
				float cnt = src[offset2]+src[offset2+1]+src[offset2+img->width]+src[offset2+1+img->width];
				dst[offset] = (cnt == 4.0f);
			}
		}
		if (targetImg->width == newWidth && targetImg->height == newHeight && targetImg->channels == img->channels && targetImg->hdr == img->hdr) {
			memcpy(targetImg->data,dst,targetImg->pitch*targetImg->height);
			uploadImage(targetImg);
		} else {
			assert(0);
		}
		delete[] dst;
	} else {
		unsigned char *dst = new unsigned char[newWidth*newHeight*img->channels];
		unsigned char *src = img->data;
		int offset = 0;
		for (int j = 0; j < newHeight; j++) {
			for (int i = 0; i < newWidth; i++,offset++) {
				int offset2 = i*2+j*2*img->width;
				int cnt = src[offset2]+src[offset2+1]+src[offset2+img->width]+src[offset2+1+img->width];
				if (cnt > 255) dst[offset] = 255;
				else dst[offset] = 0;
			}
		}
		if (targetImg->width == newWidth && targetImg->height == newHeight && targetImg->channels == img->channels) {
			memcpy(targetImg->data,dst,targetImg->pitch*targetImg->height);
			uploadImage(targetImg);
		} else {
			assert(0);
		}
		delete[] dst;
	}
}

void ImagePyramid2::copyPyramid( ImagePyramid2 *srcPyramid )
{
	ImagePyramid2 *dstPyramid = this; 
	assert(dstPyramid->nLayers == srcPyramid->nLayers);
	if (dstPyramid->nLayers != srcPyramid->nLayers) return;

	for (int i = 0; i < dstPyramid->nLayers; i++) {
		Image2 *src = srcPyramid->getImagePtr(i);
		Image2 *dst = dstPyramid->getImagePtr(i);
		assert(dst->data != NULL);	
		unsigned char *data = NULL;
		src->lockCudaPtr((void**)&data);
		cudaMemcpy(dst->data, data, src->height*src->pitch, cudaMemcpyDeviceToHost);
		src->unlockCudaPtr();
		uploadImage(dst);
	}
	//	dstPyramid->updateLayers();
}

void ImagePyramid2::zeroPyramid()
{
	ImagePyramid2 *pyramid = this;	
	for (int i = 0; i < pyramid->nLayers; i++) {
		Image2 *img = pyramid->getImagePtr(i);
		memset(img->data,0,img->pitch*img->height);
		uploadImage(img);
	}
}

ImagePyramid2::ImagePyramid2()
{
	nLayers = 0; 
	for (int i = 0; i < MAX_PYRAMID_LEVELS; i++) { 
		pyramid[i].data = NULL; 
	}
}

ImagePyramid2::~ImagePyramid2()
{
	//note: release has to be done manually, while cuda is still alive!
}

void ImagePyramid2::lock()
{
	for (int i = 0; i < nLayers; i++) pyramid[i].lock();
}

void ImagePyramid2::unlock()
{
	for (int i = 0; i < nLayers; i++) pyramid[i].unlock();
}

void ImagePyramid2::setStream( cudaStream_t stream )
{
	for (int i = 0; i < nLayers; i++) pyramid[i].setStream(stream);
}

int DisparityPyramid::downSample2( Image2 *img )
{
	if (img->channels != 1) assert(0);
	if (!img->hdr) assert(0);
	int newWidth = img->width/2;
	int newHeight = img->height/2;

	// NOTICE: disparity image assumed! dvalue divided by 2!
	float *dst = new float[newWidth*newHeight];
	float *src = (float*)img->data;
	int offset = 0;
	for (int j = 0; j < newHeight; j++) {
		for (int i = 0; i < newWidth; i++,offset++) {
			int offset2 = i*2+j*2*img->width;
			dst[offset] = (src[offset2] + src[offset2+1] + src[offset2+img->width] + src[offset2+1+img->width])/8;
		}
	}
	uploadImage(img, newWidth, newHeight, img->channels, newWidth*sizeof(float), (unsigned char*)dst, true);
	delete[] dst;
	return 1;
}

int DisparityPyramid::downSample2( Image2 *img, Image2 *img2 )
{
	if (img->channels != 1) assert(0);
	assert(img->hdr);
	int newWidth = img->width/2;
	int newHeight = img->height/2;

	// NOTICE: disparity image assumed! dvalue divided by 2!
	float *dst = new float[newWidth*newHeight];
	float *src = (float*)img->data;
	assert(img->channels==1);
	int offset = 0;
	for (int j = 0; j < newHeight; j++) {
		for (int i = 0; i < newWidth; i++,offset++) {
			int offset2 = i*2+j*2*img->width;
			dst[offset] = (src[offset2] + src[offset2+1] + src[offset2+img->width] + src[offset2+1+img->width])/8;
		}
	}
	if (img2->width == newWidth && img2->height == newHeight && img2->channels == img->channels && img2->hdr == img->hdr) {
		memcpy(img2->data,dst,img2->pitch*img2->height);
		uploadImage(img2);
	} else {
		assert(0);
		//			unsigned int gpuStatus = CREATE_GPU_TEXTURE;
		//			if (img2->onlyGPUFlag) gpuStatus = ONLY_GPU_TEXTURE;					
		//			createHdrImage(dst,newWidth,newHeight,img2,gpuStatus,img->showDynamicRange);
	}
	delete[] dst;
	return 1;
}
