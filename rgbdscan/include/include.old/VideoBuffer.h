#pragma once

class VideoBuffer {
public:
	VideoBuffer();
	bool init(const char *videoname, int nRingBuffer);
	~VideoBuffer();
	void preloadFirstFrames();
	bool getFrameData(int frameNumber, int &bytesCopied, unsigned char **retPtr);
	unsigned int getWidth();
	unsigned int getHeight();
	unsigned char *getData();
	unsigned int getFrameCount();
	unsigned int getFPS();
	void processQueue(bool &runningFlag);
	bool isInitialized();
	bool isReady();
private:
	bool fileExists(const char *fn);
	int frameToRing(int frame);
	void sanityCheck(const char *str);
	void preloadFrame(int frameNumber, int ringbufferIndex);
	bool allLoaded();
	int closestLoaded(int frame);
	int getSlot(unsigned int frame);
	void rotateRingBuffer(int newPivotFrame);
	unsigned int nframes,w,h,nRingBuffer,maxCompressedBytes;
	unsigned long totalSize;
	unsigned char *data;
	unsigned char *compressedFrame;
	unsigned int *frameIndex;
	unsigned int fps;
	bool initialized;
	bool *frameLoaded;
	unsigned int *offset;
	FILE *uc3File;
	char videoname[512];
	unsigned int *framesize;
	int pivotFrame;
	bool m_ready;
};
