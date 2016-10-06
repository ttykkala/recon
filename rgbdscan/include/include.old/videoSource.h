/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
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
