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
#include <GL/glew.h>
// CUDA utilities and system includes
// CUDA Runtime and Interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// Helper functions
#include <helper_functions.h>
#include <helper_timer.h>
// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>
#include <hostUtils.h>
#include "/usr/include/png.h"
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include "Image2.h"
#include <math.h>
#include <xmmintrin.h> //SSE2

#define PNG_SIG_BYTES 8

void loadPNGHeader(const char *name,unsigned int *width, unsigned int *height, unsigned int *nChannels, unsigned int *pitch)
{
	FILE *png_file = fopen(name, "rb");
	if (png_file == NULL) {
	    printf("png file %s did not exist or not readable\n",name);
	}
	assert(png_file);
	unsigned char header[PNG_SIG_BYTES*4];

	int ret = fread(&header[0], 1, PNG_SIG_BYTES, png_file);
	if (!png_sig_cmp((png_bytep)&header[0], 0, PNG_SIG_BYTES) == 0) {
		printf("%s header was corrupted!\n",name);
		assert(0);
	}

	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) printf("failed to create png read struct!\n");
	assert(png_ptr);

	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);

	png_infop end_info = png_create_info_struct(png_ptr);
	assert(end_info);

        //assert(!setjmp(png_jmpbuf(png_ptr)));
	png_init_io(png_ptr, png_file);
	png_set_sig_bytes(png_ptr, PNG_SIG_BYTES);
	png_read_info(png_ptr, info_ptr);

	png_uint_32 bit_depth, color_type;
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	
	//TODO: new libpng does not support this call -> gray PNG images are not supported at the moment?
	//if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
	//	png_set_gray_1_2_4_to_8(png_ptr);

	if (bit_depth == 16)
			png_set_strip_16(png_ptr);
			
	if(color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	else if(color_type == PNG_COLOR_TYPE_GRAY ||
			color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		{
			png_set_gray_to_rgb(png_ptr);
		}

	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png_ptr);
	else
		png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);

	png_read_update_info(png_ptr, info_ptr);

	*nChannels = info_ptr->channels;
	*pitch = png_get_rowbytes(png_ptr, info_ptr);
	*width = png_get_image_width(png_ptr, info_ptr);
	*height = png_get_image_height(png_ptr, info_ptr);
	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(png_file);
}

char *loadPNG(const char *name, unsigned int *width, unsigned int *height, unsigned int *nChannels, unsigned int *pitch, bool flipY)
{
    printf("loading image %s\n",name);
	FILE *png_file = fopen(name, "rb");
	assert(png_file);

	unsigned char header[PNG_SIG_BYTES];

	int ret = fread(&header[0], 1, PNG_SIG_BYTES, png_file);
	if (!png_sig_cmp((png_bytep)&header[0], 0, PNG_SIG_BYTES) == 0) {
		printf("png file %s did not exist or not readable!\n",name);
		assert(0);
	}
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(png_ptr);

	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);

	png_infop end_info = png_create_info_struct(png_ptr);
	assert(end_info);

        //assert(!setjmp(png_jmpbuf(png_ptr)));
	png_init_io(png_ptr, png_file);
	png_set_sig_bytes(png_ptr, PNG_SIG_BYTES);
	png_read_info(png_ptr, info_ptr);

	png_uint_32 bit_depth, color_type;
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	
	//TODO: new libpng does not support this call -> gray PNG images are not supported at the moment?
	//if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
	//	png_set_gray_1_2_4_to_8(png_ptr);

	if (bit_depth == 16)
			png_set_strip_16(png_ptr);
			
	if(color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	else if(color_type == PNG_COLOR_TYPE_GRAY ||
			color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		{
			png_set_gray_to_rgb(png_ptr);
		}

	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png_ptr);
	else
		png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);

	png_read_update_info(png_ptr, info_ptr);

	*nChannels = info_ptr->channels;
	*pitch = png_get_rowbytes(png_ptr, info_ptr);
	*width = png_get_image_width(png_ptr, info_ptr);
	*height = png_get_image_height(png_ptr, info_ptr);

	png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
	png_uint_32 numbytes = rowbytes*(*height);
	//*nChannels = rowbytes/(*width);
	png_byte* pixels = new png_byte[numbytes];
	png_byte** row_ptrs = (png_byte**)malloc((*height) * sizeof(png_byte*));

    if (!flipY) {
        for (unsigned int i=0; i<(*height); i++)
            row_ptrs[i] = pixels + i*rowbytes;
    } else {
        for (int i=0; i<(*height); i++)
            row_ptrs[i] = pixels + ((*height) - 1 - i)*rowbytes;
    }

	png_read_image(png_ptr, row_ptrs);

	free(row_ptrs);
	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(png_file);

	return (char *)pixels;
}

/* Attempts to save PNG to file; returns 0 on success, non-zero on error. */
int savePNG(unsigned char *data, int width, int height, int nChannels, int pitch, const char *fileName)
{
	FILE *fp = fopen(fileName, "wb");
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	int y;

	if (fp == NULL) return -1;

	/* Initialize the write struct. */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fclose(fp);
		return -1;
	}

	/* Initialize the info struct. */
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		png_destroy_write_struct(&png_ptr, NULL);
		fclose(fp);
		return -1;
	}

	/* Set up error handling. */
/*	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(fp);
		return -1;
        }*/

	unsigned int colortype = PNG_COLOR_TYPE_RGBA;
	if (nChannels == 1) colortype = PNG_COLOR_TYPE_GRAY;
	else if (nChannels == 3) colortype = PNG_COLOR_TYPE_RGB;
	else if (nChannels == 4) colortype = PNG_COLOR_TYPE_RGBA;
	else assert(0);

	/* Set Image attributes. */
	png_set_IHDR(png_ptr,
		info_ptr,
		width,
		height,
		8,
		colortype,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);

	png_byte** row_ptrs = (png_byte**)malloc(height * sizeof(png_byte*));

	for (y = 0; y < height; ++y) {
		//png_byte *pixels = (png_byte*)malloc(pitch * sizeof(png_byte));
		/*if (nChannels == 4) {
			for (int x = 0; x < width; x++) {
				pixels[x*nChannels+0] = data[(height-1-y)*pitch+x*nChannels+0];
				pixels[x*nChannels+1] = data[(height-1-y)*pitch+x*nChannels+1];
				pixels[x*nChannels+2] = data[(height-1-y)*pitch+x*nChannels+2];
				pixels[x*nChannels+3] = data[(height-1-y)*pitch+x*nChannels+3];
			}
		}*/
//		row_ptrs[y] = data + (height - 1 - y)*pitch;
		row_ptrs[y] = data + y*pitch;

	}

	/* Actually write the Image data. */
	png_init_io(png_ptr, fp);
	png_set_rows(png_ptr, info_ptr, row_ptrs);
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
//	for (int y = 0; y < height; y++){
//		free(row_ptrs[y]);
//	}
	free(row_ptrs);

	/* Finish writing. */
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
	return 0;
}

int saveImage( const char *fileName, Image2 *img )
{
	savePNG(img->data,img->width,img->height,img->channels,img->pitch,fileName);
	return 1;
}

int uploadImage( Image2 *img )
{
    if (img->data == NULL) return 1;
	if (img->texID == NO_GPU_TEXTURE) return 1;
	if (img->hdr) {
		float *src = (float*)img->data;
		if (img->showDynamicRange) {
            float *dVisu = new float[img->width*img->height*img->channels];
            memset(dVisu,0,sizeof(float)*img->width*img->height*img->channels);

            float maxD = 1.0f; float minD = FLT_MAX;
			// abs value for gradient visualization
			for (unsigned int offset = 0; offset < img->height*img->width*img->channels; offset++) dVisu[offset] = src[offset];//fabsf(src[offset]);
		
			for (unsigned int offset = 0; offset < img->height*img->width*img->channels; offset++) {
				//if (dVisu[offset] > 50) dVisu[offset] = 50;
				//if (dVisu[offset] < 36) dVisu[offset] = 36;
				if (dVisu[offset] > maxD) maxD = dVisu[offset];
				if (dVisu[offset] > 0 && dVisu[offset] < minD) minD = dVisu[offset];
			}
			for (unsigned int offset = 0; offset < img->height*img->width; offset++) {
				if (dVisu[offset]>0)
//					dVisu[offset] = (dVisu[offset]-0)/();
					dVisu[offset] = (dVisu[offset]-minD)/(maxD-minD);	
			}
            img->updateTexture(dVisu);
            delete[] dVisu;
		} else {
            img->updateTexture(src);
            //for (unsigned int offset = 0; offset < img->height*img->width*img->channels; offset++) { float v = src[offset]/255.0f; if (v>1.0f) v=1.0f; dVisu[offset]=v;}//fabsf(src[offset]);
		}

	} else img->updateTexture(0);
	return 1;
}

int uploadImage( Image2 *img, unsigned int width, unsigned int height, unsigned int channels, unsigned int pitch, unsigned char *data, bool hdr)
{
    if (width == img->width && height == img->height && channels == img->channels && pitch == img->pitch && img->hdr == hdr && img->data != NULL && data != NULL) {
		memcpy(img->data,data,height*pitch);
		uploadImage(img);
	} else {
		img->releaseData();
		unsigned int gpuStatus = CREATE_GPU_TEXTURE;
		if (img->onlyGPUFlag) gpuStatus = ONLY_GPU_TEXTURE;
		if (!hdr) createImage(data,width,height,channels,pitch,img,gpuStatus);
        else createHdrImage((float*)data,width,height,channels,img,gpuStatus,img->showDynamicRange,img->renderable);
	}
	return 1;
}

void getRawFileName(const char *fileName, char *rawFileName) {
	strcpy(rawFileName,fileName);
	int len = strlen(rawFileName);
	rawFileName[len-3] = 'r';
	rawFileName[len-2] = 'a';
	rawFileName[len-1] = 'w';	
}

unsigned char *loadRawFile(const char *fileName, unsigned int *width, unsigned int *height, unsigned int *nChannels, unsigned int *pitch) {
	loadPNGHeader(fileName,width,height,nChannels,pitch);
	int correctSize = (*height)*(*pitch);
	char rawFileName[512];
	getRawFileName(fileName,rawFileName);
	FILE *f = fopen(rawFileName,"rb");
	// no cache file found!
	if (f == NULL) return NULL;
	// read file length
	fseek(f,0,SEEK_END);
	size_t len = ftell(f);
	fseek(f,0,SEEK_SET);
	assert(len == correctSize);
	unsigned char *raw = new unsigned char[len];
	int ret = fread(raw,1,len,f);
	fclose(f);
	return raw;
}

void saveRawFile(const char *fileName, int fileSize, unsigned char *raw) {
	char rawFileName[512];
	getRawFileName(fileName,rawFileName);
	FILE *f = fopen(rawFileName,"wb");
	fwrite(raw,1,fileSize,f);
	fclose(f);
}

//TODO: should this have "renderable" flag too?
int loadImage( const char *fileName, Image2 *img, unsigned int texID, bool gray, bool flipY)
{
    unsigned char *raw = (unsigned char*)loadPNG(fileName,&img->width,&img->height,&img->channels,&img->pitch,flipY);
	if (raw == NULL) {
		fprintf(stderr,"Image data null!\n");
		return 0;
	}
    if (gray) {
		convertToGray(&raw,img->width, img->height, &img->channels);
		img->pitch = img->width;
	}
	if (img->channels == 4)
		img->type = GL_RGBA;
	else if (img->channels == 3)
		img->type = GL_RGB;
	else if (img->channels == 1) {
		img->type = GL_LUMINANCE;
	} else {
		assert(0);
	}
	img->hdr = false;
	img->data = raw;

	// note: both rgb and rgba Images have 4 channels, rgb-Images have a=255!
	// this is what adobe gives
	if (texID == CREATE_GPU_TEXTURE) {
        img->createTexture();
		img->onlyGPUFlag = false;
	} else if (texID == ONLY_GPU_TEXTURE) {
        img->createTexture();
		img->onlyGPUFlag = true;
	} else {
		img->texID = texID;
		if (img->texID != NO_GPU_TEXTURE) {
			img->updateTexture(0);
		}
    }
    //saveImage("scratch/testikuva.png", img );

    return 1;
}

int createImage( unsigned char *initData,int width, int height, int channels, int pitch, Image2 *img, unsigned int texFlag, bool renderable)
{
	img->width = width;
	img->height = height;
	img->channels = channels;
	img->pitch = pitch;
	img->hdr = false;
	img->data = new unsigned char[height*pitch];	

	if (initData == NULL) {
		memset(img->data,0,height*pitch);
	} else {
		memcpy(img->data,initData,height*pitch);
	}

	if (channels == 1)
		img->type = GL_LUMINANCE;
	else if (channels == 3)
		img->type = GL_RGB;
	else if (channels == 4)
		img->type = GL_RGBA;
	else assert(0);

	if (texFlag == CREATE_GPU_TEXTURE) {
        img->createTexture(NULL,renderable);
		img->onlyGPUFlag = false;
	} else if (texFlag == ONLY_GPU_TEXTURE) {
        img->createTexture(NULL,renderable);
		img->onlyGPUFlag = true;
	} else if (texFlag == NO_GPU_TEXTURE) { 
		img->texID = NO_GPU_TEXTURE; 
		return 1;
	} else {
		printf("unknown gpu status!\n");
		assert(0);
	}
	return 1;
}

int createHdrImage( float *initData,int width, int height, int nchannels, Image2 *img, unsigned int texFlag,bool showDynamicRange, bool renderable)
{
    img->width = width;
	img->height = height;
	img->channels = nchannels;
	img->pitch = sizeof(float)*width*nchannels;
	img->hdr = true;
	img->showDynamicRange=showDynamicRange;
	float *data = new float[height*width*nchannels];

	if (initData == NULL) {
		memset(data,0,height*img->pitch);
	} else 
		memcpy(data,initData,height*img->pitch);
	img->data = (unsigned char*)(data);
	if (nchannels == 1) img->type = GL_LUMINANCE;
	else if (nchannels == 3) img->type = GL_RGB;
	else { assert(0); }
 
	if (texFlag == CREATE_GPU_TEXTURE) {
        img->createTexture(NULL,renderable);
		img->onlyGPUFlag = false;
	} else if (texFlag == ONLY_GPU_TEXTURE) {
        img->createTexture(NULL,renderable);
		img->onlyGPUFlag = true;
	} else if (texFlag == NO_GPU_TEXTURE) { 
		img->texID = NO_GPU_TEXTURE; 
		return 1;
	} else {
		printf("unknown gpu status!\n");	
		assert(0);
	}
	return 1;
}

int convertToGray( unsigned char **rawdata, unsigned int width,unsigned int height,unsigned int *nchannels )
{
	unsigned int channels = *nchannels;
	if (channels == 1) {return 1;}

	if (channels == 4) {
		// convert to gray scale
		int size = width*height;
		unsigned char *dst = new unsigned char[size];
		unsigned char *src = *rawdata;
		for (int i = 0; i < size; i++) {
			dst[i] = (unsigned char)(float(src[i*4+2])*0.11f+float(src[i*4+1])*0.59f+float(src[i*4+0])*0.3f);
		}
		delete[] src;
		*rawdata = dst;
		*nchannels = 1;
	} else assert(0);
	return 1;
}

unsigned char blur3x3(unsigned char *data, int width) {
	unsigned int v = data[-width-1]*1+data[-width]*2+data[-width+1]+data[-1]*2+data[0]*4+data[1]*2+data[width-1]*1+data[width]*2+data[width+1];
	return (unsigned char)(v >> 4);
}
/*
int downSample3x3( Image *img )
{
	if (img->channels != 1) assert(0);
	if (img->hdr) assert(0);

	int newWidth = img->width/2;
	int newHeight = img->height/2;

	unsigned char *data = new unsigned char[newWidth*newHeight];
	memset(data,0,sizeof(char)*newWidth*newHeight);
	for (int j = 1; j < newHeight-1; j++) {
		for (int i = 1; i < newWidth-1; i++) {
			int offset = i+j*newWidth;
			int offset2 = i*2+j*2*img->width;
			data[offset] = blur3x3(&img->data[offset2],img->width);				
				//unsigned char((img->data[offset2] + img->data[offset2+1] + img->data[offset2+img->width] + img->data[offset2+1+img->width])/4);
		}
	}
	uploadImage(img, newWidth, newHeight, 1, newWidth, data, false);
	delete[] data;
	return 1;
}

int downSample3x3( Image *img, Image *img2)
{
	if (img->channels != 1) assert(0);
	if (img->hdr) assert(0);

	int newWidth = img->width/2;
	int newHeight = img->height/2;

	unsigned char *data = new unsigned char[newWidth*newHeight];
	for (int j = 1; j < newHeight-1; j++) {
		for (int i = 1; i < newWidth-1; i++) {
			int offset = i+j*newWidth;
			int offset2 = i*2+j*2*img->width;
			data[offset] = blur3x3(&img->data[offset2],img->width);//unsigned char((img->data[offset2] + img->data[offset2+1] + img->data[offset2+img->width] + img->data[offset2+1+img->width])/4);
		}
	}
	uploadImage(img2,newWidth,newHeight,img2->channels,img2->pitch,data,img->hdr);
	delete[] data;
	return 1;
}
*/


/*
DWORD GetSubTexel( int x, int y )
{
	const int h = (x & 0xff00) / 255;
	const int i = (y & 0xff00) / 255;

	x = x >> 16;
	y = y >> 16;

	const COLORREF cr1 = GetTexel( x + 0, y + 0 );
	const COLORREF cr2 = GetTexel( x + 1, y + 0 );
	const COLORREF cr3 = GetTexel( x + 1, y + 1 );
	const COLORREF cr4 = GetTexel( x + 0, y + 1 );

	const int a = (0x100 - h) * (0x100 - i);
	const int b = (0x000 + h) * (0x100 - i);
	const int c = (0x000 + h) * (0x000 + i);
	const int d = 65536 - a - b - c;

	const unsigned int R = 0x00ff0000 & (((cr1 >> 16)      * a) + ((cr2 >> 16)      * b) + ((cr3 >> 16)      * c) + ((cr4 >> 16)      * d));
	const unsigned int G = 0xff000000 & (((cr1 & 0x00ff00) * a) + ((cr2 & 0x00ff00) * b) + ((cr3 & 0x00ff00) * c) + ((cr4 & 0x00ff00) * d));
	const unsigned int B = 0x00ff0000 & (((cr1 & 0x0000ff) * a) + ((cr2 & 0x0000ff) * b) + ((cr3 & 0x0000ff) * c) + ((cr4 & 0x0000ff) * d));

	return R|((G|B)>>16);    
}
*/


unsigned char interpolatePixelFast( Image2 *img, int xf, int yf )
{	
	unsigned int x = (unsigned int)(xf * 65536.0f);
	unsigned int y = (unsigned int)(yf * 65536.0f);

	const int h = (x & 0xff00) / 255;//>> 8;/// 255;
	const int i = (y & 0xff00) / 255;//>> 8;/// 255;
	x = x >> 16;
	y = y >> 16;

	if (x >= img->width-2 || y >= img->height-2) return 0;

	unsigned char *I = img->data;
	unsigned int pitch = img->pitch;
 	unsigned char cr1 = I[x+y*pitch];
	unsigned char cr2 = I[x+1+y*pitch];
	unsigned char cr3 = I[(x+1)+(y+1)*pitch];
	unsigned char cr4 = I[x+(y+1)*pitch];
	
	const int a = (0x100 - h) * (0x100 - i);
	const int b = (0x000 + h) * (0x100 - i);
	const int c = (0x000 + h) * (0x000 + i);
	const int d = 65536 - a - b - c;

	const unsigned int C = 0x00ff0000 & ((cr1 * a) + (cr2 * b) + (cr3 * c) + (cr4 * d) + 32760);
	return (unsigned char)(C>>16);    
}


float interpolatePixel2F(Image2 *img, float x, float y)
{
	if (x < 0 || y < 0) return 0;
	if (x >= img->width-2 || y > img->height-2) return 0;

	unsigned char *I = img->data;
	unsigned int pitch = img->width;
	int xi = int(x);
	int yi = int(y);
	float fracX = x-xi;
	float fracY = y-yi;

	float i1 = (float)I[xi+yi*pitch];
	float i2 = (float)I[xi+1+yi*pitch];
	float i3 = (float)I[xi+(yi+1)*pitch];
	float i4 = (float)I[(xi+1)+(yi+1)*pitch];

	return (1-fracX)*(1-fracY)*i1 + fracX*(1-fracY)*i2 + (1-fracX)*fracY*i3 + fracX*fracY*i4;
}

unsigned char interpolatePixel(Image2 *img, float x, float y)
{
	unsigned int xi = (unsigned int)x;
	unsigned int yi = (unsigned int)y;

	if (xi >= img->width-2 || yi >= img->height-2) return 0;

	unsigned int fracX = (unsigned int)((x-xi)*256.0f);
	unsigned int fracY = (unsigned int)((y-yi)*256.0f);

	unsigned int pitch = img->pitch;
	unsigned char *ptr = &img->data[xi+yi*pitch];
 	unsigned char i1 = ptr[0]; unsigned char i2 = ptr[1]; ptr += pitch;
	unsigned char i4 = ptr[0]; unsigned char i3 = ptr[1];

	const unsigned int c = fracX * fracY;
	const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
	const unsigned int b = (fracX<<8) - c;	
	const unsigned int d = 65536 - a - b - c;

	return (a*i1 + b*i2 + c*i3 + d*i4)>>16;
}

void interpolateRGBAPixel(Image2 *img, float x, float y, unsigned char *colorR, unsigned char *colorG, unsigned char *colorB, unsigned char *colorA)
{
	unsigned int xi = (unsigned int)x;
	unsigned int yi = (unsigned int)y;

	if (xi >= img->width-2 || yi >= img->height-2) return;

	unsigned int fracX = (unsigned int)((x-xi)*256.0f);
	unsigned int fracY = (unsigned int)((y-yi)*256.0f);
	unsigned int pitch = img->pitch;
	unsigned char *ptr = &img->data[(xi<<2)+yi*pitch];

	const unsigned int c = fracX * fracY;
	const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
	const unsigned int b = (fracX<<8) - c;	
	const unsigned int d = 65536 - a - b - c;

	unsigned char i1,i2,i3,i4;

	i1 = ptr[0]; i2 = ptr[4];
	i4 = ptr[pitch]; i3 = ptr[pitch+4];
	*colorR = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[1]; i2 = ptr[5];
	i4 = ptr[pitch+1]; i3 = ptr[pitch+5];
	*colorG = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[2]; i2 = ptr[6];
	i4 = ptr[pitch+2]; i3 = ptr[pitch+6];
	*colorB = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[3]; i2 = ptr[7];
	i4 = ptr[pitch+3]; i3 = ptr[pitch+7];
	*colorA = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
}

void interpolateRGBPixel(unsigned char *rgb, int width, int height, float x, float y, unsigned char *colorR, unsigned char *colorG, unsigned char *colorB)
{
    unsigned int xi = (unsigned int)x;
    unsigned int yi = (unsigned int)y;

    if (xi >= width-2 || yi >= height-2) return;

    unsigned int fracX = (unsigned int)((x-xi)*256.0f);
    unsigned int fracY = (unsigned int)((y-yi)*256.0f);
    unsigned int pitch = width*3;
    unsigned char *ptr = &rgb[xi*3+yi*pitch];

    const unsigned int c = fracX * fracY;
    const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
    const unsigned int b = (fracX<<8) - c;
    const unsigned int d = 65536 - a - b - c;

    unsigned char i1,i2,i3,i4;

    i1 = ptr[0]; i2 = ptr[3];
    i4 = ptr[pitch]; i3 = ptr[pitch+3];
    *colorR = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

    i1 = ptr[1]; i2 = ptr[4];
    i4 = ptr[pitch+1]; i3 = ptr[pitch+4];
    *colorG = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

    i1 = ptr[2]; i2 = ptr[5];
    i4 = ptr[pitch+2]; i3 = ptr[pitch+5];
    *colorB = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
}

void interpolateRGBAPixel(Image2 *img, float x, float y, int *colorR, int *colorG, int *colorB, int *colorA)
{
	unsigned int xi = (unsigned int)x;
	unsigned int yi = (unsigned int)y;

	if (xi >= img->width-2 || yi >= img->height-2) return;

	unsigned int fracX = (unsigned int)((x-xi)*256.0f);
	unsigned int fracY = (unsigned int)((y-yi)*256.0f);
	unsigned int pitch = img->pitch;
	unsigned char *ptr = &img->data[(xi<<2)+yi*pitch];

	const unsigned int c = fracX * fracY;
	const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
	const unsigned int b = (fracX<<8) - c;	
	const unsigned int d = 65536 - a - b - c;

	unsigned char i1,i2,i3,i4;

	i1 = ptr[0]; i2 = ptr[4];
	i4 = ptr[pitch]; i3 = ptr[pitch+4];
	*colorR = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[1]; i2 = ptr[5];
	i4 = ptr[pitch+1]; i3 = ptr[pitch+5];
	*colorG = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[2]; i2 = ptr[6];
	i4 = ptr[pitch+2]; i3 = ptr[pitch+6];
	*colorB = (a*i1 + b*i2 + c*i3 + d*i4)>>16;

	i1 = ptr[3]; i2 = ptr[7];
	i4 = ptr[pitch+3]; i3 = ptr[pitch+7];
	*colorA = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
}

int interpolatePixelSSE(unsigned char *data, int width, int height, int pitch, float xx, float yy) {
	const float x = xx * 128.0f;
	const float y = yy * 128.0f;
	int retIntensity = 0;
// TODO: how does __asm work in linux?
#if 0
	unsigned int widthBound = (width-2)<<7;
	unsigned int heightBound = (height-2)<<7;
	__asm {
		xor eax,eax // out of bounds -> 0
		// check bounds
		cvtss2si ebx,x // ebx = xi*128
		cvtss2si ecx,y // ecx = yi*128
		cmp ebx, widthBound
		ja outOfRange
		cmp ecx, heightBound
		ja outOfRange
		// edi <- base offset 
		mov	edi,data
		mov edx,ebx
		mov eax,ecx //eax available again
		shr edx,7 // edx = xi
		shr eax,7 // eax = yi
		add edi,edx
		mul pitch

		// compute fracX,fracY
		and ebx, 0x7F
		and ecx, 0x7F
		mov edx, ebx // edx = fracX
		mov esi, ecx // esi = fracY

		add edi,eax // edi = base offset

		mov  ebx, edx//fracX
		imul bx, esi//fracY // ebx = c	
		// store a
		mov eax, 16384
		shl esi,7
		shl edx, 7 // dx = fracX<<7
		sub ax,si
 		add ax,bx
		sub ax,dx 
		shl eax,16
		// store b
		sub dx,bx
		mov ax,dx
		movd mm6, eax
	
		// store d
		mov ecx,16384
		sub cx,ax //-b
		sub cx,bx //-c
		shr eax,16
		sub cx,ax
		mov eax,ecx
		shl eax,16
		// save c
		mov ax,bx
		movd mm4, eax	
		psllq mm6, 32
		
		// fetch 4 pixel values and store to mm4 as packed words
		movzx ax,[edi]
		movzx bx,[edi+1]
		por mm6,mm4

		add edi, pitch
		shl eax, 16
		mov ax,bx
		movd mm4, eax
		movzx ax,[edi]
		movzx bx,[edi+1]
		psllq mm4, 32
		shl eax, 16
		mov ax,bx
		movd mm5, eax
		por mm4,mm5
		// a=(w1*p1 + w2*p2),b=(w3*p3 + w4*p4) 
		pmaddwd mm4, mm6
		// move a -> eax
		movd eax, mm4
		psrlq mm4, 32
		// move b -> ecx
		movd ecx, mm4
		// c=a+b
		add eax, ecx
		// c = c>>14
		shr eax, 14
outOfRange:
		mov retIntensity,eax
		// flush mmx after use
		emms
	}
#endif
	return retIntensity;
}

/*
int interpolatePixelSSE(unsigned char *data, int width, int height, int pitch, float xx, float yy)
{
	const float x = xx * 128.0f;
	const float y = yy * 128.0f;

	unsigned int widthBound = (width-2)<<7;
	unsigned int heightBound = (height-2)<<7;
	int retIntensity;
	
	__asm {
		xor eax,eax // out of bounds -> 0
		// check bounds
		cvtss2si ebx,x // ebx = xi*128
		cmp ebx, widthBound
		ja outOfRange
		cvtss2si ecx,y // ecx = yi*128
		cmp ecx, heightBound
		ja outOfRange
		// edi <- base offset 
		mov	edi,data
		mov eax,ebx
		shr eax,7 // edx = xi
		add edi,eax
		mov eax,ecx //eax available again
		shr eax,7 // eax = yi
		mul pitch
		add edi,eax // edi = base offset

		// compute fracX,fracY
		and ebx, 0x7F
		mov edx, ebx // edx = fracX
		and ecx, 0x7F
		mov esi, ecx // esi = fracY

		mov  ebx, edx//fracX
		imul bx, esi//fracY // ebx = c	
		// store a
		mov eax, 16384
		mov ecx, esi//fracY
		shl ecx, 7
		sub ax,cx
		mov ecx, edx//fracX
		shl ecx, 7 // cx = fracX<<7
 		sub ax,cx 
		add ax,bx
		shl eax,16
		// store b
		sub cx,bx
		mov ax,cx
		movd mm6, eax
		psllq mm6, 32

		// store d
		mov ecx,16384
		sub cx,bx //-c
		sub cx,ax //-b
		shr eax,16
		sub cx,ax
		mov eax,ecx
		shl eax,16
		// save c
		mov ax,bx
		movd mm4, eax
		por mm6,mm4

		// fetch 4 pixel values and store to mm4 as packed words
		movzx ax,[edi]
		shl eax, 16
		movzx ax,[edi+1]
		movd mm4, eax
		psllq mm4, 32
		add edi, pitch
		movzx ax,[edi]
		shl eax, 16
		movzx ax,[edi+1]
		movd mm5, eax
		por mm4,mm5
		// a=(w1*p1 + w2*p2),b=(w3*p3 + w4*p4) 
		pmaddwd mm4, mm6
		// move a -> eax
		movd eax, mm4
		psrlq mm4, 32
		// move b -> ecx
		movd ecx, mm4
		// c=a+b
		add eax, ecx
		// c = c>>14
		shr eax, 14
outOfRange:
		mov retIntensity,eax
		// flush mmx after use
		emms
	}
	return retIntensity;
}*/

float interpolateFloatPixel(float *depthMapRow0, float *depthMapRow1, float x, float y) {
	int xi = int(x);
	int yi = int(y);
	float fracX = x-xi;
	float fracY = y-yi;
	float i1 = depthMapRow0[xi];
	float i2 = depthMapRow0[xi+1];
	float i3 = depthMapRow1[xi];
	float i4 = depthMapRow1[xi+1];
	return (1-fracX)*(1-fracY)*i1 + fracX*(1-fracY)*i2 + (1-fracX)*fracY*i3 + fracX*fracY*i4;
}


void getRGBAPixel(Image2 *img, int xi, int yi, unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a)
{
	unsigned char *I = img->data;
	unsigned int pitch = img->pitch;
	if (xi < 0 || yi < 0) return;
	if (xi >= img->width-1 || yi > img->height-1) return;

	int offset = xi*img->channels+yi*pitch;
	*r = I[offset];
	*g = I[offset+1];
	*b = I[offset+2];
	*a = I[offset+3];
}

void getPixel(Image2 *img, int xi, int yi, unsigned char *v)
{
	unsigned char *I = img->data;
	unsigned int pitch = img->pitch;
	if (xi < 0 || yi < 0) return;
	if (xi >= img->width-1 || yi > img->height-1) return;
	*v = I[xi+yi*pitch];
}

float interpolateFloatPixel(Image2 *img, float x, float y)
{
	if (x < 0 || y < 0) return 0;
	if (x >= img->width-2 || y > img->height-2) return 0;

	float *I = (float*)img->data;
	unsigned int pitch = img->width;
	int xi = int(x);
	int yi = int(y);
	float fracX = x-xi;
	float fracY = y-yi;

	float i1 = I[xi+yi*pitch];
	float i2 = I[xi+1+yi*pitch];
	float i3 = I[xi+(yi+1)*pitch];
	float i4 = I[(xi+1)+(yi+1)*pitch];
	return (1-fracX)*(1-fracY)*i1 + fracX*(1-fracY)*i2 + (1-fracX)*fracY*i3 + fracX*fracY*i4;
}

float interpolateFloatPixelZeroCheck(Image2 *img, float x, float y, bool &validFlag)
{
//	validFlag = true;
	if (x < 0 || y < 0) return 0;
	if (x >= img->width-2 || y > img->height-2) return 0;

	float *I = (float*)img->data;
	unsigned int pitch = img->width;
	int xi = int(x);
	int yi = int(y);
	float fracX = x-xi;
	float fracY = y-yi;

	float i1 = I[xi+yi*pitch];
	float i2 = I[xi+1+yi*pitch];
	float i3 = I[xi+(yi+1)*pitch];
	float i4 = I[(xi+1)+(yi+1)*pitch];
	if (i1*i2*13*i4 == 0.0f) { validFlag = false; return 0; }
	// add almost half for rounding up!
	// note: have to prevent overflow at 255	
	return (1-fracX)*(1-fracY)*i1 + fracX*(1-fracY)*i2 + (1-fracX)*fracY*i3 + fracX*fracY*i4;
}
float average3x3f(Image2 *img, float x, float y) {
	float a = 0;
	a += interpolateFloatPixel(img,x-1,y-1);
	a += interpolateFloatPixel(img,x,y-1);
	a += interpolateFloatPixel(img,x+1,y-1);
	a += interpolateFloatPixel(img,x-1,y);
	a += interpolateFloatPixel(img,x,y);
	a += interpolateFloatPixel(img,x+1,y);
	a += interpolateFloatPixel(img,x-1,y+1);
	a += interpolateFloatPixel(img,x,y+1);
	a += interpolateFloatPixel(img,x+1,y+1);
	return a/9.0f;
}

void Image2::bind()
{
    if (extDataFlag || !renderable) return;
	assert(texID != CREATE_GPU_TEXTURE && texID != NO_GPU_TEXTURE);
	if (texID == NO_GPU_TEXTURE || pbo == 0) return;
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, texID);
	if (!hdr) glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, type, GL_UNSIGNED_BYTE, 0);
	else glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, type, GL_FLOAT, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
//	cudaThreadSynchronize();
}

void Image2::createTexture(void *extData, bool renderable)
{
    if (extDataFlag) return;
    assert(pbo==0);
    assert(cuda_pbo_resource == 0);

    cArray = NULL;
    this->renderable = renderable;

    if (renderable) {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

        if (extData == 0)
            glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch*height, data, GL_STREAM_COPY);
        else
            glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch*height, extData, GL_STREAM_COPY);

        assert(cuda_pbo_resource == 0);
        //While a PBO is registered to CUDA, it can't be used
        //as the destination for OpenGL drawing calls.
        //But in our particular case OpenGL is only used
        //to display the content of the PBO, specified by CUDA kernels,
        //so we need to register/unregister it only once.
        checkCudaErrors(cudaGraphicsGLRegisterBuffer((struct cudaGraphicsResource**)&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsNone));
        //createCudaArray();

        // create texture for display
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE); // auto-mipmapping
        //	glTexEnvi(GL_TEXTURE_FILTER_CONTROL,GL_TEXTURE_LOD_BIAS, 3);
        if (!hdr) {
            unsigned int internal = GL_RGBA8;
            if (type == GL_RGB) internal = GL_RGB8;
            if (type == GL_LUMINANCE) internal = GL_LUMINANCE8;
            glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, type, GL_UNSIGNED_BYTE, NULL);
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, GL_FLOAT, NULL);

        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    } else {
        // allocate non-renderable image buffer from GPU memory
        pbo    = 0;
        cuda_pbo_resource = 0;
        checkCudaErrors(cudaMalloc((void **)&devPtr, pitch*height));
        if (extData == 0) {
            checkCudaErrors(cudaMemcpy(devPtr,data,pitch*height, cudaMemcpyHostToDevice));
        } else {
            checkCudaErrors(cudaMemcpy(devPtr,extData,pitch*height, cudaMemcpyHostToDevice));
        }
        checkCudaError("Image2 init");
    }
}

void Image2::setWriteDiscardFlag() {
    if (extDataFlag || !renderable) return;
	if (cuda_pbo_resource == 0) return;
	cudaGraphicsResourceSetMapFlags((struct cudaGraphicsResource*)cuda_pbo_resource,cudaGraphicsMapFlagsWriteDiscard);
}

Image2::~Image2()
{
	//note: releaseData() must be done manually, while cuda is still alive!
}

void Image2::releaseGPUData()
{
    setStream(0);
    unlock();
//	printf("releasing image \"%s\"\n",imageName);
    //if (extDataFlag) { releaseCudaArray(); return; }
    if (data != NULL) {
        if (texID != NO_GPU_TEXTURE) {
            //assert(pbo != 0);
            releaseCudaArray();

            if (cuda_pbo_resource != 0) {
                 checkCudaErrors(cudaGraphicsUnregisterResource((struct cudaGraphicsResource*)cuda_pbo_resource));
                 glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
                 cuda_pbo_resource = 0;
            }
            if (pbo != 0) {
                glDeleteBuffers(1, &pbo); pbo = 0;
            }
            if (texID != 0) {
                glDeleteTextures(1, &texID); texID = 0;
            }
            if (!renderable && devPtr != NULL) cudaFree(devPtr); devPtr = NULL;
            //checkCudaError("Image2 destro");
        }
    }
}
void Image2::releaseData()
{
    releaseGPUData();
    if (data != NULL) delete[] data; data = NULL;
}

void Image2::updateTexture(void *extData)
{
    if (renderable) {
        if (texID == NO_GPU_TEXTURE || pbo == 0) { assert(0); return; }
        unsigned char *devPtr = NULL;
        this->lockCudaPtr((void**)&devPtr);
        if (extData == NULL) cudaMemcpy(devPtr, data, height*pitch, cudaMemcpyHostToDevice);
        else cudaMemcpy(devPtr, extData, height*pitch, cudaMemcpyHostToDevice );
        this->unlockCudaPtr();
    } else {
        if (texID == NO_GPU_TEXTURE || devPtr == NULL) { assert(0); return; }
        if (extData == NULL) cudaMemcpy(devPtr, data, height*pitch, cudaMemcpyHostToDevice);
        else cudaMemcpy(devPtr, extData, height*pitch, cudaMemcpyHostToDevice );
    }
}

void Image2::updateTextureInternal(void *devData, bool updateArrayFlag)
{
    if (renderable) {
        if (texID == NO_GPU_TEXTURE || pbo == 0 || devPtr == NULL) { assert(0); return; }
        cudaMemcpyAsync(devPtr, devData, height*pitch, cudaMemcpyDeviceToDevice,cudaStream);
    } else {
        if (texID == NO_GPU_TEXTURE || devPtr == NULL) { printf("devPtr==NULL while attempting to copy texture data!\n"); assert(0); return; }
        cudaMemcpyAsync(devPtr, devData, height*pitch, cudaMemcpyDeviceToDevice,cudaStream);
    }
    if (updateArrayFlag) updateCudaArrayInternal();
}

void *Image2::lock() {
    if (data == NULL) return NULL;
    if (extDataFlag || !renderable) return devPtr;
	// already locked?
	if (devPtr != NULL) return devPtr; 
	lockCudaPtr(&devPtr);
	return devPtr;
}
void Image2::unlock() {
    if (data == NULL) return;
    if (extDataFlag || !renderable) return;
	if (devPtr != NULL) { unlockCudaPtr(); devPtr = NULL; }
}

void Image2::lockCudaPtr( void **cudaPtr)
{
    if (data == NULL) { *cudaPtr = NULL; return; }
    if (extDataFlag || !renderable) { *cudaPtr = devPtr; return;}
	//printf("locking image!\n");
    checkCudaErrors(cudaGraphicsMapResources(1, (struct cudaGraphicsResource**)&cuda_pbo_resource, cudaStream));
	size_t numBytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(cudaPtr, &numBytes, (struct cudaGraphicsResource*)cuda_pbo_resource));
}

void Image2::unlockCudaPtr()
{
    if (data == NULL) { return; }

    if (extDataFlag || !renderable) return;
	if (cuda_pbo_resource == 0) return;
	assert(cuda_pbo_resource != 0);
	//printf("unlocking image!\n");

	//struct cudaGraphicsResource *cuda_pbo = (struct cudaGraphicsResource*)cuda_pbo_resource;
    checkCudaErrors(cudaGraphicsUnmapResources(1, (struct cudaGraphicsResource**)&cuda_pbo_resource, cudaStream));
}

Image2::Image2()
{
	data = NULL; 
	pitch = 0; 
	channels = 0; 
	width = 0; 
	height = 0; 
	texID = 0; 
	pbo = 0; 
	cuda_pbo_resource = 0; 
	type = 0; 
	hdr = false;
	onlyGPUFlag = false;
	shiftIntensity = 0; 
	showDynamicRange=true; 
	cArray = NULL;
	cudaStream = 0;
	devPtr = NULL;
	extDataFlag = false;
    renderable = true;
	sprintf(imageName,"default");
}

Image2::Image2(void *extCudaDevPtr, int width, int height, int pitch, int channels, bool hdr) {
	this->data = NULL; 
	this->pitch = pitch; 
	this->channels = channels; 
	this->width = width; 
	this->height = height; 
	texID = 0; 
	pbo = 0; 
	cuda_pbo_resource = 0; 
	if (channels == 4) type = GL_RGBA;
	else if (channels == 3) type = GL_RGB;
	else if (channels == 1) type = GL_LUMINANCE;
	else assert(0);
	this->hdr = hdr;
	onlyGPUFlag = true;
	shiftIntensity = 0; 
	showDynamicRange=false; 
	cArray = NULL;
	cudaStream = 0;
	devPtr = extCudaDevPtr;
	extDataFlag = true;
    renderable = true;
	sprintf(imageName,"extAllocated");
}


void Image2::createCudaArray()
{
    if (cArray == NULL && channels == 1) {
        if (!hdr) {
            cudaChannelFormatDesc description = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
            cudaMallocArray((cudaArray**)&cArray,&description,width,height);
        } else {
            cudaChannelFormatDesc description = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
            cudaMallocArray((cudaArray**)&cArray,&description,width,height);
        }
    }
}

void Image2::releaseCudaArray() {
    cudaArray *arr = (cudaArray*)cArray;
    if (arr != NULL) {
        cudaFreeArray(arr);
    }
}

void Image2::updateCudaArray() {
    if (cArray != NULL && channels == 1) {
        unsigned char *ptr;
        lockCudaPtr((void**)&ptr);
        cudaMemcpyToArrayAsync((cudaArray*)cArray,0,0,ptr,pitch*height,cudaMemcpyDeviceToDevice);
        unlockCudaPtr();
    }
}

void Image2::updateCudaArrayInternal() {
    if (cArray != NULL && channels == 1) {
        cudaMemcpyToArrayAsync((cudaArray*)cArray,0,0,devPtr,pitch*height,cudaMemcpyDeviceToDevice,cudaStream);
    }
}
