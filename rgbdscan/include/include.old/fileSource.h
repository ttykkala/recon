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

#include "videoSource.h"

class Calibration;

class FileSource : public VideoSource {
private:
    string loadingPathStr;
    int prevLoadIndex;
    string fileRGB;
    string fileDepth;
    bool singleFileMode;
    bool loadSingleRGBD(Calibration *calib,int targetResoX, int targetResoY);
public:
    FileSource(const char *baseDir, bool flipY=false);
    FileSource(const char *filenameRGB, const char *filenameDepth);
    ~FileSource();
  /*  int getWidth();
    int getHeight();
    int getDisparityWidth();
    int getDisparityHeight();*/
    int fetchRawImages(unsigned char **rgbCPU, float **depthCPU, int frameIndex, int rgbTargetResoX, Calibration *calib);
    void reset();
};
