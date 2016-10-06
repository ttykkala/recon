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

#include "KinectDisparityCompressor.h"

// compress 2x2 disparity blocks using max filter
// rationale: in run time 640->320 compression is done in anycase with 2x2 block max filter
// the 2x2 blocks here must match runtime 2x2 blocks!
void compressDisparity2(Mat &hiresD, Mat &loresD)
{
	unsigned short *dPtr = (unsigned short*)hiresD.ptr();
	unsigned short *dLowPtr = (unsigned short*)loresD.ptr();
	int widthLow = loresD.cols; int heightLow = loresD.rows;
	assert(widthLow*2 == hiresD.cols && heightLow*2 == hiresD.rows);

	int offset = 0;
	// fill first row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int i2 = 2*i;
		dLowPtr[offset] = MAX(dPtr[i2],dPtr[i2+1]);
	}
	int width = 2*widthLow;
	// fill full 2x2 blocks by max value
	for (int j = 1; j < heightLow-1; j++) {
		for (int i = 0; i < widthLow; i++, offset++) {
			int i2 = 2*i; int j2 = 2*j-1;
			int offset2 = i2 + j2 * width;
			unsigned short m1 = MAX(dPtr[offset2],dPtr[offset2+1]);
			unsigned short m2 = MAX(dPtr[offset2+width],dPtr[offset2+1+width]);
			dLowPtr[offset] = MAX(m1,m2);
		}
	}
	// fill last row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int offset2 = 2*i + (2*heightLow-1)*width;
		dLowPtr[offset] = MAX(dPtr[offset2],dPtr[offset2+1]);
	}
}


// decompress 2x2 disparity blocks using 2x2 replication
// rationale: in run time 640->320 compression is done in anycase with 2x2 block max filter
// the 2x2 blocks here must match runtime 2x2 blocks!
void decompressDisparity2(Mat &loresD, Mat &hiresD)
{
	unsigned short *dPtr = (unsigned short*)hiresD.ptr();
	unsigned short *dLowPtr = (unsigned short*)loresD.ptr();
	int widthLow = loresD.cols; int heightLow = loresD.rows;
	assert(widthLow*2 == hiresD.cols && heightLow*2 == hiresD.rows);

	int offset = 0;
	// fill first row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int i2 = 2*i;
		dPtr[i2] = dLowPtr[offset];
		dPtr[i2+1] = dLowPtr[offset];
	}
	int width = 2*widthLow;
	// fill full 2x2 blocks by max value
	for (int j = 1; j < heightLow-1; j++) {
		for (int i = 0; i < widthLow; i++, offset++) {
			int i2 = 2*i; int j2 = 2*j-1;
			int offset2 = i2 + j2 * width;
			dPtr[offset2] = dLowPtr[offset];
			dPtr[offset2+1] = dLowPtr[offset];
			dPtr[offset2+width] = dLowPtr[offset];
			dPtr[offset2+width+1] = dLowPtr[offset];
		}
	}
	// fill last row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int offset2 = 2*i + (2*heightLow-1)*width;
		dPtr[offset2] = dLowPtr[offset];
		dPtr[offset2+1] = dLowPtr[offset];
	}
}
