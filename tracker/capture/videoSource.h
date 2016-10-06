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

class Calibration;

class VideoSource {
private:
public:
  /*  virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual int getDisparityWidth() = 0;
    virtual int getDisparityHeight() = 0;*/
    virtual int getFrame() { return 0; }
    virtual void setFrame(int frame) {};
    virtual int fetchRawImages(unsigned char **rgbCPU, float **depthCPU, int frameIndex, int rgbTargetResoX, Calibration *calib) = 0;
    // dummy calls by default
    virtual void setRecording(const char *, bool, bool saveToDisk=false, int nFrames = 0, bool averageFrames = false, bool compressedDepthFrames=true) { };
    virtual bool isRecording() { return false; }
    virtual bool isPaused() { return false; }
    virtual float getSecondsRemaining() { return 0.0f; }
    virtual void pause() {};
    virtual void start() {};
    virtual void stop() {};
    virtual void record() {};
    virtual void reset() {};
    virtual void setExposure(float exposureVal)  {};
};
