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

#include <calib/calib.h>
#include <multicore/multicore.h>
#include <vector>

class ZConv {
private:
public:
        ZConv();
        ~ZConv();
        int convert(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib);
        int d2z(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFilter=false);
        int d2zHdr(float *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering=false);
        int d2zGPU(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib);
        void baselineTransform(float *depthImageL, float *depthImageR, int zwidth, int zheight, Calibration *calib);
        void baselineWarp(float *depthImageL,unsigned char *grayImageR, ProjectData *fullPointSet, int width, int height, Calibration *calib);
        void baselineWarpRGB(float *depthImageL,unsigned char *rgbDataR, ProjectData *fullPointSet, int width, int height, Calibration &calib);
        // this method is currently deprecated.
        // TODO: use polyCoeffs for undistortion!
        void undistortDisparityMap(unsigned short* disp16, float *udisp, int widht, int height, Calibration* calib);
        void dumpDepthRange(float *depthMap, int width, int height);
        void mapDisparityRange(unsigned short* ptr, int w, int h, int minD,int maxD);
        void increaseDynamics(float *dispImage,int w, int h, float scale);
        void setRange(float*ptr, int len, float minZ, float maxZ, float z);
        void mapFromRGBToDepth(ProjectData *pointCloud,  Calibration &calib, std::vector<cv::Point2f> &rgbPoints, std::vector<cv::Point3f> &depthPoints3D, std::vector<cv::Point2f> &depthPoints2D, std::vector<int> &mask);
};
