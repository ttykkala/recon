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

#include <string>
using namespace std;

#include "videoSource.h"

class Kinect : public VideoSource {
private:
	string recordingPathStr;
	void saveScratchBuffer();
	bool pauseFlag;
	bool initFailed;
	bool capturingFlag;
	bool saveToDisk;
	bool averageSavedFrames;
public:
    Kinect(const char *baseDir);
    ~Kinect();
    int getWidth();
    int getHeight();
    int getDisparityWidth();
    int getDisparityHeight();

    int fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int frameIndex);
    // nFrames = 0 <-> max amount of frames
    void setRecording(const char *recordingPath, bool flag, bool saveToDisk=false, int nFrames=0, bool averageFrames = false, bool compressedDepthFrames=true);
    bool isRecording();
    bool isPaused();
    float getSecondsRemaining();
    void pause();
    void startKinect();
    void stopKinect();
    void start() { startKinect(); };
    void stop() { stopKinect(); };
    void record();
    void setExposure(float exposureVal);
};
