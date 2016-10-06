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

// TODO: test usleep !
/*
current driver features:
	- timestamp order is respected
	- the two most recent rgb and depth frames are put forward
		- in case of recording mode
			- assumed even callback number for rgb and depth
			- frame number difference is maintained for depth and rgb
			- only frames in correct timestamp order are counted
			- recordingFrame++ will be done in the delayed callback
*/
#include <opencv2/opencv.hpp>
#include <libfreenect.h>
#include <unistd.h>
//#include <SDL.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include "KinectDisparityCompressor.h"
#include "kinect.h"
#include <reconstruct/basic_math.h>
#include <timer/performanceCounter.h>

using namespace std;
using namespace cv;

static Mat rgbHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC3);
static Mat rgbHeaderSmall(RGB_HEIGHT_SMALL, RGB_WIDTH_SMALL, CV_8UC3);

freenect_context *f_ctx = NULL;
freenect_device *f_dev = NULL;
int freenect_angle = 0;
int freenect_led;

pthread_t fnkt_thread;
pthread_mutex_t callbackMutex;
pthread_cond_t newDepthMapAvailable = PTHREAD_COND_INITIALIZER;

const int kinectFrameSizeCompressed = kinectBayerSize+compressedKinectDepthSize;
const int kinectFrameSize = kinectBayerSize+kinectDepthSize;
const int scratchMemorySizeMB = 2000;
const int maxNumberOfFramesCompressed = (scratchMemorySizeMB*1024*1024)/kinectFrameSizeCompressed;
const int maxNumberOfFrames = (scratchMemorySizeMB*1024*1024)/kinectFrameSize;
static unsigned char *videoRecordingScratch = NULL;
static int frameRecordingIndex = 0;
static int finalFrameIndex = 0;//maxNumberOfFrames-1;

static bool filledRGBBackbuffer = false;
static bool filledDepthBackbuffer = false;
static bool controlThreadRunning = false;
static bool compressDepthFrames = true;

static unsigned int rgbStamp = 0;
static unsigned int depthStamp = 0;
static unsigned int depthCounter = 0;
static unsigned int rgbCounter = 0;
 
// storage for recent depth frames
const int nRingBuffer = 5;
int rgbBufferIndex = 0;
int depthBufferIndex = 0;
static unsigned char dMapCur[nRingBuffer*kinectDepthSize];
static unsigned char rgbCur[nRingBuffer*kinectRgbSizeSmall];
static unsigned int  dMapStamps[nRingBuffer];
static unsigned int  rgbStamps[nRingBuffer];
PerformanceCounter callbackTimer;

// for recording raw data
static unsigned char dMapRec[kinectDepthSize];

static bool recordingOn = false;

unsigned int stampDiff(unsigned int stampA, unsigned int stampB) {
	unsigned int diff = 0;
	if (stampA < stampB) {
	     if (stampB - stampA > 0xfffff) diff = (UINT_MAX - stampB) + stampA;
	     else diff = stampB-stampA;
	} else {
	     if (stampA - stampB > 0xfffff) diff = (UINT_MAX - stampA) + stampB;
	     else diff = stampA-stampB;
	}
	return diff;
}

//TODO:
// Should image synchronization be used also in real-time Kinect mode?
// Currently images are synchronized only when recording frames.
void depth_cb(freenect_device *dev, void *depth, uint32_t timeStamp)
{
    bool problemFrame = false;
    // if stamp order does not match without uint32 overflow -> problem
    if (timeStamp < depthStamp && (depthStamp - timeStamp) < 0xffffff) problemFrame = true;
    if (!controlThreadRunning) problemFrame = true;

    if (problemFrame) { printf("problem depthframe!\n"); return; }

    pthread_mutex_lock(&callbackMutex);
/*
    callbackTimer.StopCounter();
    printf("depth: %3.1fms\n",callbackTimer.GetElapsedTime()*1000.0f);
    callbackTimer.StartCounter();
*/

    Mat depthHeader(DISPARITY_HEIGHT, DISPARITY_WIDTH, CV_16UC1, depth);
    memcpy(&dMapCur[depthBufferIndex*kinectDepthSize],depthHeader.ptr(),kinectDepthSize);
    dMapStamps[depthBufferIndex] = timeStamp;
    depthBufferIndex = (depthBufferIndex+1)%nRingBuffer;

    if (recordingOn) {
        if (compressDepthFrames) {
            Mat depthLowHeader(COMPRESSED_DISPARITY_HEIGHT, COMPRESSED_DISPARITY_WIDTH, CV_16UC1, &dMapRec[0]);
            compressDisparity2(depthHeader,depthLowHeader);
            memcpy(&videoRecordingScratch[frameRecordingIndex*kinectFrameSizeCompressed+kinectBayerSize],&dMapRec[0],compressedKinectDepthSize);
        } else {
            memcpy(&videoRecordingScratch[frameRecordingIndex*kinectFrameSize+kinectBayerSize],depthHeader.ptr(),kinectDepthSize);
        }
        // frameRecordingIndex updated by "slower callback"
        if (depthCounter < rgbCounter) {
            // skip missing frames
            while (rgbCounter - depthCounter > 1) {
                depthCounter++;
            }
            if (frameRecordingIndex >= finalFrameIndex) recordingOn = false;
            else  frameRecordingIndex++;
        }
    }
    //printf("depth image callback %u!\n",timeStamp);
    depthStamp = timeStamp;
    depthCounter++;
    filledDepthBackbuffer = true;

    pthread_mutex_unlock(&callbackMutex);
}

void rgb_cb(freenect_device *dev, void *rgb, uint32_t timeStamp)
{
    bool problemFrame = false;
    // if stamp order does not match without uint32 overflow -> problem
    if (timeStamp < rgbStamp && (rgbStamp - timeStamp) < 0xffffff) problemFrame = true;
    if (!controlThreadRunning) problemFrame = true;

    if (problemFrame) { printf("problem rgbframe!\n"); return; }

    pthread_mutex_lock(&callbackMutex);
  /*  callbackTimer.StopCounter();
    printf("rgb: %3.1fms\n",callbackTimer.GetElapsedTime()*1000.0f);
    callbackTimer.StartCounter();*/

    Mat bayerHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC1, rgb);
    cvtColor(bayerHeader,rgbHeader,CV_BayerGB2RGB);
    pyrDown(rgbHeader,rgbHeaderSmall);
    memcpy(&rgbCur[rgbBufferIndex*kinectRgbSizeSmall],rgbHeaderSmall.ptr(),kinectRgbSizeSmall);
    rgbStamps[rgbBufferIndex] = timeStamp;
    rgbBufferIndex = (rgbBufferIndex+1)%nRingBuffer;

    if (recordingOn) {
        if (compressDepthFrames) memcpy(&videoRecordingScratch[frameRecordingIndex*kinectFrameSizeCompressed],bayerHeader.ptr(),kinectBayerSize);
        else memcpy(&videoRecordingScratch[frameRecordingIndex*kinectFrameSize],bayerHeader.ptr(),kinectBayerSize);
        //              printf("save rgb %d\n",frameRecordingIndex);

        // frameRecordingIndex updated by "slower callback"
        if (rgbCounter < depthCounter) {
            // skip missing frames
            while (depthCounter - rgbCounter > 1) {
                rgbCounter++;
            }
            if (frameRecordingIndex >= finalFrameIndex) recordingOn = false;
            else  frameRecordingIndex++;
        }
    }

    rgbStamp = timeStamp;
    filledRGBBackbuffer=true;
    rgbCounter++;
    pthread_mutex_unlock(&callbackMutex);
}

void *freenect_threadfunc(void* arg) {
	controlThreadRunning = true;
	while(controlThreadRunning && (freenect_process_events(f_ctx) >= 0) ) { /*SDL_Delay(1);*/ }
	return NULL;
}


Kinect::Kinect(const char *baseDir)
{
	frameRecordingIndex = 0;
	recordingPathStr = baseDir;
	pauseFlag = false;
	initFailed = false;
	capturingFlag = false;
	saveToDisk = false;
	//printf("recording path set to %s\n",recordingPathStr.c_str());
	pthread_mutex_init(&callbackMutex,NULL);

	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("freenect_init() failed\n");
		return;
	}

	freenect_set_log_level(f_ctx, FREENECT_LOG_INFO);
    freenect_select_subdevices(f_ctx,(freenect_device_flags)(FREENECT_DEVICE_CAMERA));
	int nr_devices = freenect_num_devices(f_ctx);
	printf ("Number of devices found: %d\n", nr_devices);
	int user_device_number = 0;
	//	if (argc > 1)
	//		user_device_number = atoi(argv[1]);

	memset(&dMapCur[0],0,kinectDepthSize*nRingBuffer);
        memset(&rgbCur[0],0,kinectRgbSizeSmall*nRingBuffer);
        memset(&rgbStamps[0],0,sizeof(int)*nRingBuffer);
        memset(&dMapStamps[0],0,sizeof(int)*nRingBuffer);

	// for recording depth maps
	memset(&dMapRec[0],0,kinectDepthSize);

	if (nr_devices < 1) { initFailed = true; return; }

    int ret = 0;
    if ((ret=freenect_open_device(f_ctx, &f_dev, user_device_number)) < 0) {
        printf("Could not open device (%d)\n",ret);
		return;
	}
//	freenect_set_tilt_degs(f_dev,freenect_angle);
	freenect_set_led(f_dev,LED_RED);
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_video_callback(f_dev, rgb_cb);
    ret=0;
	ret = freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_BAYER));
	printf("set color mode ok: %d\n",ret>=0);
	ret = freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
	printf("set 11bit ok: %d\n",ret>=0);
}

Kinect::~Kinect()
{	
	stopKinect();

	freenect_close_device(f_dev);
	freenect_shutdown(f_ctx);
	pthread_mutex_destroy(&callbackMutex);

	if (videoRecordingScratch != NULL) delete[] videoRecordingScratch;



/*
    static unsigned char *mappedRGBPtr = NULL;
    static unsigned char *mappedRGBDevPtr = NULL;
    static cudaStream_t driverStreamRGB;
    static cudaStream_t driverStreamRGBASync;

    static Mat rgbHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC3);
    static Mat rgbHeaderSmall(RGB_HEIGHT_SMALL, RGB_WIDTH_SMALL, CV_8UC3);


    static unsigned short *mappedDepthPtr = NULL;
    static unsigned short *mappedDepthDevPtr = NULL;
    static cudaStream_t driverStreamDepth;
    static cudaStream_t driverStreamDepthASync;

    freenect_context *f_ctx = NULL;
    freenect_device *f_dev = NULL;
    int freenect_angle = 0;
    int freenect_led;

    pthread_t fnkt_thread;
    pthread_mutex_t callbackMutex;
    pthread_cond_t newDepthMapAvailable = PTHREAD_COND_INITIALIZER;
  */
}

int Kinect::getWidth()
{
	return RGB_WIDTH_SMALL;
}

int Kinect::getHeight()
{
	return RGB_HEIGHT_SMALL;
}


int Kinect::getDisparityWidth() {
        return DISPARITY_WIDTH;
}

int Kinect::getDisparityHeight() {
    return DISPARITY_HEIGHT;
}


int Kinect::fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int /*frameIndex*/)
{  
    // define previously filled ring buffer indices
    int rgbBufferIndex2 = (rgbBufferIndex+nRingBuffer-1)%nRingBuffer;
    int depthBufferIndex2 = (depthBufferIndex+nRingBuffer-1)%nRingBuffer;

    *rgbCPU   = &rgbCur[rgbBufferIndex2*kinectRgbSizeSmall];
    *depthCPU = (unsigned short*)&dMapCur[depthBufferIndex2*kinectDepthSize];

     // automatically save scratch buffer to disk if saveToDisk=true
     if (!recordingOn && frameRecordingIndex > 0 && saveToDisk) {
         saveToDisk = false;
         saveScratchBuffer();
     }
     return 1;
}

void Kinect::startKinect() {
	printf("starting kinect!\n");
	if (initFailed || capturingFlag) return;

	filledDepthBackbuffer = false;
	filledRGBBackbuffer = false;	
	frameRecordingIndex = 0;
	rgbStamp = 0;
	depthStamp = 0;
	depthCounter = 0;
	rgbCounter = 0;
    rgbBufferIndex = 0;
    depthBufferIndex = 0;

    // reset depth time stamps
    for (int i = 0; i < nRingBuffer; i++) {
        dMapStamps[i] = 0;
        rgbStamps[i] = 0;
        memset(&rgbCur[i*kinectRgbSizeSmall],0,kinectRgbSizeSmall);
        memset(&dMapCur[i*kinectDepthSize],0,kinectDepthSize);
    }
    printf("pertti.\n");
    //freenect_exposure_control(f_dev,0x028E); // set 33ms exposure

    freenect_start_video(f_dev);
    freenect_start_depth(f_dev);

    //freenect_disable_automatics(f_dev);

    callbackTimer.StartCounter();

	int res = pthread_create(&fnkt_thread, NULL, freenect_threadfunc, NULL);
	if (res) {
		printf("pthread_create failed\n");
		return;
	}

	// capture few test frames
	capturingFlag = true;
	int nFrames = nRingBuffer;
	while (nFrames > 0) {
		// read first frames and swap buffers -> ready to serve first images immidiately
		while(!filledRGBBackbuffer || !filledDepthBackbuffer) { 
            usleep(1000);
//            SDL_Delay(1);
		}
		filledDepthBackbuffer = false;
		filledRGBBackbuffer = false;
		nFrames--;
	}
}

void Kinect::stopKinect() {
	if (initFailed || !capturingFlag) return;
	capturingFlag = false;
	printf("stopping control thread\n");
	controlThreadRunning = false; 	
//	pthread_mutex_lock(&callbackMutex);
	printf("stopping depth\n"); freenect_stop_depth(f_dev);
	printf("stopping video\n"); freenect_stop_video(f_dev);
//	pthread_mutex_unlock(&callbackMutex);
	pthread_join(fnkt_thread, NULL);
    callbackTimer.StopCounter();
    usleep(33000);

//        SDL_Delay(33);
}

void filterMedianStructure(int nImages, unsigned char *data, int stride, int dw, int dh, cv::Mat &depthMat) {

    int imageSize = dw*dh;
    float *sampleData = new float[nImages*imageSize]; memset(sampleData,0,sizeof(float)*imageSize*nImages);
    int *counterImage  = new int[imageSize];          memset(counterImage,0,sizeof(int)*imageSize);
    unsigned short *tempMap = new unsigned short[imageSize]; memset(tempMap,0,sizeof(short)*imageSize);

    for (int k = 0; k < nImages; k++) {
        Mat depthHeader(dh, dw, CV_16UC1, &data[k*stride]);
        unsigned short *dptr = (unsigned short*)depthHeader.ptr();
        for (int i = 0; i < imageSize; i++) {
            if ((dptr[i] > 0) && (dptr[i] < 2047)) {
                int sampleIndex = counterImage[i];
                sampleData[i*nImages+sampleIndex] = float(dptr[i]);
                counterImage[i]++;
            }
        }
    }
    // median structure per pixel
    for (int i = 0; i < imageSize; i++) {
        float median = quickMedian(&sampleData[i*nImages],counterImage[i]);
        tempMap[i] = (unsigned short)median;
    }
    Mat depthResult(dh, dw, CV_16UC1, tempMap);
    if (dw == depthMat.cols) {
        memcpy(depthMat.ptr(),depthResult.ptr(),sizeof(short)*dw*dh);
    } else if (2*dw == depthMat.cols) {
        decompressDisparity2(depthResult,depthMat);
    } else {
        assert(0);
    }

    delete[] tempMap;
    delete[] sampleData;
    delete[] counterImage;
}


void Kinect::saveScratchBuffer()
{
    Mat depthMat(DISPARITY_HEIGHT, DISPARITY_WIDTH, CV_16UC1);
    if (!averageSavedFrames) {
        stopKinect();

        for (int i = 0; i <= frameRecordingIndex; i++) {
            if (compressDepthFrames) {
                Mat bayerHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC1, &videoRecordingScratch[i*kinectFrameSizeCompressed]);
                Mat depthHeader(COMPRESSED_DISPARITY_HEIGHT, COMPRESSED_DISPARITY_WIDTH, CV_16UC1, &videoRecordingScratch[i*kinectFrameSizeCompressed+kinectBayerSize]);
                int saveIndex = i+1;
                char buf[512];
                sprintf(buf,"%s/bayer_rgbimage%04d.ppm",recordingPathStr.c_str(),saveIndex);
                printf("saving %s\n",buf);
                bool rgbWriteOk = imwrite(buf, bayerHeader);
                decompressDisparity2(depthHeader,depthMat);
                sprintf(buf,"%s/rawdepth%04d.ppm",recordingPathStr.c_str(),saveIndex);
                printf("saving %s\n",buf);
                bool depthWriteOk = imwrite(buf, depthMat);
                if (!rgbWriteOk || !depthWriteOk) printf("error in saving image %d!\n",saveIndex);
            } else {
                Mat bayerHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC1, &videoRecordingScratch[i*kinectFrameSize]);
                Mat depthHeader(DISPARITY_HEIGHT, DISPARITY_WIDTH, CV_16UC1, &videoRecordingScratch[i*kinectFrameSize+kinectBayerSize]);
                int saveIndex = i+1;
                char buf[512];
                sprintf(buf,"%s/bayer_rgbimage%04d.ppm",recordingPathStr.c_str(),saveIndex);
                printf("saving %s\n",buf);
                bool rgbWriteOk = imwrite(buf, bayerHeader);
                sprintf(buf,"%s/rawdepth%04d.ppm",recordingPathStr.c_str(),saveIndex);
                printf("saving %s\n",buf);
                bool depthWriteOk = imwrite(buf, depthHeader);
                if (!rgbWriteOk || !depthWriteOk) printf("error in saving image %d!\n",saveIndex);
            }
        }
        startKinect();
    } else {
        char buf[512];
        bool depthWriteOk = false;
        bool rgbWriteOk = false;
        // scratch rgb image
        Mat rgbMatHdr = Mat::zeros(RGB_HEIGHT,RGB_WIDTH, CV_32FC3);
        Mat rgbMatHdrCur(RGB_HEIGHT,RGB_WIDTH, CV_32FC3);
        Mat rgbImage(RGB_HEIGHT, RGB_WIDTH, CV_8UC3);

        if (compressDepthFrames) {

            filterMedianStructure(frameRecordingIndex+1,&videoRecordingScratch[kinectBayerSize],kinectFrameSizeCompressed, COMPRESSED_DISPARITY_WIDTH, COMPRESSED_DISPARITY_HEIGHT, depthMat);
            for (int i = 0; i <= frameRecordingIndex; i++) {
                // average rgb images
                Mat bayerHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC1, &videoRecordingScratch[i*kinectFrameSizeCompressed]);
                cvtColor(bayerHeader,rgbImage,CV_BayerGB2BGR);
                rgbImage.convertTo(rgbMatHdrCur,CV_32FC3,1.0f,0);
                add(rgbMatHdr,rgbMatHdrCur,rgbMatHdr);
            }
            sprintf(buf,"%s_disp.ppm",recordingPathStr.c_str());
            printf("saving %s\n",buf);
            depthWriteOk = imwrite(buf, depthMat);
        } else {
            filterMedianStructure(frameRecordingIndex+1,&videoRecordingScratch[kinectBayerSize],kinectFrameSize, DISPARITY_WIDTH, DISPARITY_HEIGHT, depthMat);
            for (int i = 0; i <= frameRecordingIndex; i++) {
                // average rgb images
                Mat bayerHeader(RGB_HEIGHT, RGB_WIDTH, CV_8UC1, &videoRecordingScratch[i*kinectFrameSize]);
                cvtColor(bayerHeader,rgbImage,CV_BayerGB2BGR);
                rgbImage.convertTo(rgbMatHdrCur,CV_32FC3,1.0f,0);
                add(rgbMatHdr,rgbMatHdrCur,rgbMatHdr);
                //  printf("avg rgb %d\n",i);
            }
            sprintf(buf,"%s_disp.ppm",recordingPathStr.c_str());
            printf("saving %s\n",buf);
            depthWriteOk = imwrite(buf, depthMat);
        }
        rgbMatHdr.convertTo(rgbImage,CV_8UC3,1.0f/float(frameRecordingIndex+1),0);
        sprintf(buf,"%s_rgb.ppm",recordingPathStr.c_str());
        printf("saving %s\n",buf);
        rgbWriteOk = imwrite(buf, rgbImage);
        if (!rgbWriteOk || !depthWriteOk) printf("error in saving keyframe images to %s_{rgb,disp}.ppm!\n",recordingPathStr.c_str());
    }
    frameRecordingIndex = 0;
}

void Kinect::setRecording(const char *recordingPath, bool flag, bool saveToDiskFlag, int nFrames, bool averageFrames, bool compressedDepthFrames)
{
	if (!recordingOn && flag) { 
		saveToDisk = saveToDiskFlag;
		averageSavedFrames = averageFrames;
		recordingPathStr = recordingPath;
		compressDepthFrames = compressedDepthFrames;
		if (nFrames == 0) {
			if (compressDepthFrames) finalFrameIndex = maxNumberOfFramesCompressed-1;
			else finalFrameIndex = maxNumberOfFrames-1;
		} else {
			finalFrameIndex = MAX(nFrames-1,0);
			if (compressDepthFrames) finalFrameIndex = MIN(finalFrameIndex,maxNumberOfFramesCompressed-1);
			else MIN(finalFrameIndex,maxNumberOfFrames-1);
		}

		if (videoRecordingScratch == NULL) {
			videoRecordingScratch = new unsigned char[kinectFrameSize*maxNumberOfFrames]; 
			if (videoRecordingScratch == NULL) printf("scratch allocation failed for %d frames!\n",maxNumberOfFrames);
			else { 
				printf("scratch buffer allocated, compressed: %3.1fs, uncompressed: %3.1fs\n",float(maxNumberOfFramesCompressed)/30.0f,float(maxNumberOfFrames)/30.0f);
			}
		}
		pthread_mutex_lock(&callbackMutex);
		filledDepthBackbuffer = false;
		filledRGBBackbuffer = false;
                rgbStamp = 0;
                depthStamp = 0;
                depthCounter = 0;
                rgbCounter = 0;
                rgbBufferIndex = 0;
                depthBufferIndex = 0;
                frameRecordingIndex = 0;
                recordingOn = flag;
		pthread_mutex_unlock(&callbackMutex);
	}
	if (recordingOn && !flag) { 
		recordingOn = false; 
		if (saveToDisk) {
			saveToDisk = false;
			saveScratchBuffer();
		}
	}
}

bool Kinect::isRecording()
{
	return recordingOn;
}

bool Kinect::isPaused() {
    return !controlThreadRunning;
}

float Kinect::getSecondsRemaining()
{
	return float(finalFrameIndex-frameRecordingIndex)/30.0f;
}


void Kinect::pause()
{
	pauseFlag = !pauseFlag;
	if (pauseFlag) stopKinect();
	if (!pauseFlag) startKinect();
}

void Kinect::record()
{
	saveScratchBuffer();
}

void Kinect::setExposure(float exposureVal) {
    if (f_dev == NULL) return;
    //freenect_exposure_control_raw(f_dev,350+int(exposureVal*float(654-350)));
   // freenect_dump_rgb_bits(f_dev);
   // fflush(stdout);
}

