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

#define __CALIBRATION_H__

#define  KR_OFFSET   0
#define iKR_OFFSET   9
#define KcR_OFFSET  18
#define TLR_OFFSET  26
#define C1_OFFSET    42
#define KL_OFFSET   43
#define iKL_OFFSET  52
#define C0_OFFSET    61
#define MIND_OFFSET 62
#define MAXD_OFFSET 63
#define TRL_OFFSET  64
//#define ALPHA0_OFFSET 77
//#define ALPHA1_OFFSET 78
//#define BETA_OFFSET 79 // 640*480 static disparity distortion function
#define ZPOLY_DATA  80 //307279 // 16xfloats to determine a zmap correction polynomial
#define CALIB_SIZE  96 //307295

class Calibration {
private:
    double KL[9];
	double KR[9];
    double TLR[16];
	double TRL[16];
    double kcR[8];
    double kcL[8];
    //double B;
    //double b;
    float *zpolyCoeffs;
    int numZPolyCoeffs;
  //  float *beta;
    double c0,c1;
   // double alpha0,alpha1;
    double minDist, maxDist;
    float *calibData; //KR,iKRL,kcR,TLR,B,KL,iKL,b,minDist,maxDist
    int width,height;
    bool useXYOffset; // are the 2d points defined based on IR or disparity image grid?
    char *m_filename;
    //int initOulu(const char *fileName, bool silent);
    bool fileExists(const char *fileName);
    void resetVariables();
public:
    Calibration(const char *fn, bool silent = false);
    Calibration();
    ~Calibration();
    void copyCalib(Calibration &extCalib);
    double *getKR() { return &KR[0]; }
    double *getKL() { return &KL[0]; }
    double *getKcR() { return &kcR[0]; }
    double *getKcL() { return &kcL[0]; }
    double *getTLR() { return &TLR[0]; }
    //double  getB() { return B; }
    double getC0() { return c0; }
    double getC1() { return c1; }
    //double getAlpha0() {return alpha0; }
    //double getAlpha1() {return alpha1; }
    double getMinDist() { return minDist; }
    double getMaxDist() { return maxDist; }
    //double  getKinectBaseline() { return b; }
    void setMinDist(double minDist) { this->minDist = minDist; }
    void setMaxDist(double maxDist) { this->maxDist = maxDist; }
    void setupCalibDataBuffer(int width, int height);
    float *getCalibData() { return &calibData[0]; }
    float getFovX_R();
    float getFovY_R();
    float getFovX_L();
    float getFovY_L();
    int init(const char *fn, bool silent = true);
    int getWidth() { return width; }
    int getHeight() { return height; }
    bool isOffsetXY() { return useXYOffset; }
    void saveZPolynomial(int polyOrder, float *coeffs);
    int getNumPolyCoeffs();
    float *getPolyCoeffs();
};
