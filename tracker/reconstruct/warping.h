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

#define __WARPING_H__

class Warping
{
private:
	double KLo[9],KRo[9],K3o[9];
	double KL[9],KR[9],K3[9];
	double iKL[9],iKR[9],iK3[9];
	double tKL[9],tKR[9],tK3[9];
	double T1[9],T2[9],T3[9];
	double dTK1_1[9],dTK2_1[9],dTK3_1[9];
	double dTK1_2[9],dTK2_2[9],dTK3_2[9];
	double dTK1_3[9],dTK2_3[9],dTK3_3[9];
	double dTK1_4[9],dTK2_4[9],dTK3_4[9];
	double dTK1_5[9],dTK2_5[9],dTK3_5[9];
	double dTK1_6[9],dTK2_6[9],dTK3_6[9];
	// storage for extrinsic diff params
	double dT1_1[9],dT2_1[9],dT3_1[9];
	double dT1_2[9],dT2_2[9],dT3_2[9];
	double dT1_3[9],dT2_3[9],dT3_3[9];
	double dT1_4[9],dT2_4[9],dT3_4[9];
	double dT1_5[9],dT2_5[9],dT3_5[9];
	double dT1_6[9],dT2_6[9],dT3_6[9];

	double base1[16],base2[16],base3[16];
	double postT[16];
	double origBase3[16];
	double Mrel[16];
	double invMarker[16],invTmp[16];
	double MrelMarker[16];
	// matrix structure for cuda
	float matrixData[54];
	// CUDA device memory pointers
	float *matrixDataDev;
	int prevResoUpdatedDev;
//	float *tKRDev,*T1Dev,*T2Dev,*T3Dev,*iKLDev,*K3Dev;
	
	void copy3x3(double *M3x4, double *R3x3);
	void copy3x3(float *M3x4, float *R3x3);
	void copy3x3T(double *M3x4, int cols, double *R3x3);
	void transpose3x3(double *M3x3, double *R3x3);
	void transpose3x3(float *M3x3, float *R3x3);
	void zeroMatrix3x3(double *M3x3);
	void identityMatrix4x4(double *M4x4);
	double det3x3(double *mat);
	void inverse3x3( double *ma, double *mr);
	void matrixMult3x3(double *M1, double *M2, double *R);
	void matrixMultVec2(double *M1, float *V, double *R);
	void matrixMultVec3(float *M1, float *V, float *R);
	void generateTensor(double *R2c, double *C2c, double *R3c, double *C3c, double *T1, double *T2, double *T3);
	void canonizeTransformation(double *R1, double *C1, double *R2, double *C2, double *R3, double *C3, double *R2c, double *C2c, double *R3c, double *C3c);
	void invertRT(double *R, double *t, double *Ri, double *ti);
	void invertRT(float *R, float *t, float *Ri, float *ti);
	void verticalLine(float *P, double *L);
	void verticalLineDiff(double *dL);
	void setupExtrinsic(double *CM1, double *CM2, double *CM3);
	void precomputeTrifocalDiff();
	void precomputeTrifocalDiffIntrinsic();
	void updateCudaMatrixData();
public:
	// postT is the post transform applied after motion increments
	// using this its possible to specify motion purely for the right view
	// and postT transform when warping to left.
	Warping(int resoX, double *K1, double *CM1, double *K2, double *CM2, double *K3, double *CM3, double *postT);
	~Warping();
	void warp(float *points1, float *points2, int nPoints, float *outPoints);
	void warpCuda(float *points1, float *points2, int nPoints, float *outPoints);
	void warpPointLineCuda(float *pointsR0Dev, float *linesL0Dev, int nPoints, float *linesL1Dev);

	void warp(float *points1, float *points2, int nPoints, float *outPoints, float *outPoints3);
	void warpDiff( float *pointsR, float *pointsL, int nPoints, float *out1, float *out2, float *out3, float *out4, float *out5, float *out6, float *outD);

	void setRelativeTransform(double *M);
	double *getRelativeTransform();
	double *getRelativeTransformInv();
	double *getRelativeTransformToMarker();
	void setTransformSE3(double *x);
	void mulTransformSE3(float *x);
	void setImageResolution(int resoX, bool updateDeviceFlag = false);
	void resetBase();
	void setMarker();
	void invertRT4( double *M, double *Mi );
	void invertRT4( float *M, float *Mi );
	void matrixMultVec2(double *M1, double *V, double *R);
	void matrixMultVec3(double *M1, double *V, double *R);
	void matrixMultVec4(double *M1, double *V, double *R);
	void transformRT3(double *M1, float *V, float *R);
	void matrixMult4x4(double *M1, double *M2, double *R);
	void matrixMult4x4(float *M1, float *M2, float *R);
	double *getKR(); double *getNormalizedKR();
	double *getKL();

	double cudaTransfer0;
	double cudaProcess;
	double cudaTransfer1;
	int nCudaMeasurements;

	double cpuProcess;
	int nCpuMeasurements;
};
