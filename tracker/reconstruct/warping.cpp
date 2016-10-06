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
#include "warping.h"
#include <string.h>
#include <expm.h>
#include <stdio.h>
// CUDA
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <timer/performanceCounter.h>
#include <../cudakernels/cuda_funcs.h>
//PerformanceCounter warpTimer;

void Warping::copy3x3(double *M3x4, double *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[1]; R3x3[2] = M3x4[2];
	R3x3[3] = M3x4[4]; R3x3[4] = M3x4[5]; R3x3[5] = M3x4[6];
	R3x3[6] = M3x4[8]; R3x3[7] = M3x4[9]; R3x3[8] = M3x4[10];
}

void Warping::copy3x3(float *M3x4, float *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[1]; R3x3[2] = M3x4[2];
	R3x3[3] = M3x4[4]; R3x3[4] = M3x4[5]; R3x3[5] = M3x4[6];
	R3x3[6] = M3x4[8]; R3x3[7] = M3x4[9]; R3x3[8] = M3x4[10];
}

void Warping::copy3x3T(double *M3x4, int cols, double *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[0+cols]; R3x3[2] = M3x4[0+cols*2];
	R3x3[3] = M3x4[1]; R3x3[4] = M3x4[1+cols]; R3x3[5] = M3x4[1+cols*2];
	R3x3[6] = M3x4[2]; R3x3[7] = M3x4[2+cols]; R3x3[8] = M3x4[2+cols*2];
}    

void Warping::transpose3x3(double *M3x3, double *R3x3)
{
	R3x3[0] = M3x3[0]; R3x3[1] = M3x3[3]; R3x3[2] = M3x3[6];
	R3x3[3] = M3x3[1]; R3x3[4] = M3x3[4]; R3x3[5] = M3x3[7];
	R3x3[6] = M3x3[2]; R3x3[7] = M3x3[5]; R3x3[8] = M3x3[8];
}
void Warping::transpose3x3(float *M3x3, float *R3x3)
{
	R3x3[0] = M3x3[0]; R3x3[1] = M3x3[3]; R3x3[2] = M3x3[6];
	R3x3[3] = M3x3[1]; R3x3[4] = M3x3[4]; R3x3[5] = M3x3[7];
	R3x3[6] = M3x3[2]; R3x3[7] = M3x3[5]; R3x3[8] = M3x3[8];
}
/*
void Warping::transpose4x4(double *M4x4, double *R4x4)
{
	R4x4[0] = M4x4[0]; R4x4[1] = M4x4[4]; R4x4[2] = M4x4[6];
	R4x4[3] = M4x4[1]; R4x4[4] = M4x4[4]; R4x4[5] = M4x4[7];
	R4x4[6] = M4x4[2]; R4x4[7] = M4x4[5]; R4x4[8] = M4x4[8];
} */

void Warping::matrixMult3x3(double *M1, double *M2, double *R)
{
	R[0] = M1[0]*M2[0]+M1[1]*M2[3]+M1[2]*M2[6];
	R[3] = M1[3]*M2[0]+M1[4]*M2[3]+M1[5]*M2[6];
	R[6] = M1[6]*M2[0]+M1[7]*M2[3]+M1[8]*M2[6];

	R[1] = M1[0]*M2[1]+M1[1]*M2[4]+M1[2]*M2[7];
	R[4] = M1[3]*M2[1]+M1[4]*M2[4]+M1[5]*M2[7];
	R[7] = M1[6]*M2[1]+M1[7]*M2[4]+M1[8]*M2[7];

	R[2] = M1[0]*M2[2]+M1[1]*M2[5]+M1[2]*M2[8];
	R[5] = M1[3]*M2[2]+M1[4]*M2[5]+M1[5]*M2[8];
	R[8] = M1[6]*M2[2]+M1[7]*M2[5]+M1[8]*M2[8];
}

void Warping::matrixMult4x4(double *M1, double *M2, double *R)
{
	double Rtmp[16];
	Rtmp[0] = M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12];
	Rtmp[1] = M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13];
	Rtmp[2] = M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14];
	Rtmp[3] = M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15];

	Rtmp[4] = M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12];
	Rtmp[5] = M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13];
	Rtmp[6] = M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14];
	Rtmp[7] = M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15];

	Rtmp[8]  = M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12];
	Rtmp[9]  = M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13];
	Rtmp[10] = M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14];
	Rtmp[11] = M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15];

	Rtmp[12] = M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12];
	Rtmp[13] = M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13];
	Rtmp[14] = M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14];
	Rtmp[15] = M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15];
	memcpy(R,Rtmp,sizeof(double)*16);
}

void Warping::matrixMult4x4(float *M1, float *M2, float *R)
{
	float Rtmp[16];
	Rtmp[0] = M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12];
	Rtmp[1] = M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13];
	Rtmp[2] = M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14];
	Rtmp[3] = M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15];

	Rtmp[4] = M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12];
	Rtmp[5] = M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13];
	Rtmp[6] = M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14];
	Rtmp[7] = M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15];

	Rtmp[8]  = M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12];
	Rtmp[9]  = M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13];
	Rtmp[10] = M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14];
	Rtmp[11] = M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15];

	Rtmp[12] = M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12];
	Rtmp[13] = M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13];
	Rtmp[14] = M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14];
	Rtmp[15] = M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15];
	memcpy(R,Rtmp,sizeof(float)*16);
}
void Warping::matrixMultVec2(double *M1, double *V, double *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2];
	R[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5];
	R[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8];
}

void Warping::matrixMultVec2(double *M1, float *V, double *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2];
	R[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5];
	R[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8];
}

void Warping::matrixMultVec3(double *M1, double *V, double *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
	R[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5]*V[2];
	R[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8]*V[2];
}

void Warping::matrixMultVec3(float *M1, float *V, float *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
	R[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5]*V[2];
	R[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8]*V[2];
}

void Warping::transformRT3(double *M1, float *V, float *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2]+M1[3];
	R[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2]+M1[7];
	R[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2]+M1[11];
}

void Warping::zeroMatrix3x3(double *M3x3)
{
	M3x3[0] = 0; M3x3[1] = 0; M3x3[2] = 0;
	M3x3[3] = 0; M3x3[4] = 0; M3x3[5] = 0;
	M3x3[6] = 0; M3x3[7] = 0; M3x3[8] = 0;
}

void Warping::identityMatrix4x4(double *M4x4)
{
	M4x4[0] = 1; M4x4[1] = 0; M4x4[2] = 0; M4x4[3] = 0;
	M4x4[4] = 0; M4x4[5] = 1; M4x4[6] = 0; M4x4[7] = 0; 
	M4x4[8] = 0; M4x4[9] = 0; M4x4[10] = 1; M4x4[11] = 0;
	M4x4[12] = 0; M4x4[13] = 0; M4x4[14] = 0; M4x4[15] = 1;
}

void Warping::generateTensor(double *R2c, double *C2c, double *R3c, double *C3c, double *T1, double *T2, double *T3)
{
	double S1[9],S2[9],S3[9];

	S1[0] = R2c[0]*C3c[0]-C2c[0]*R3c[0]; S1[1] = R2c[0]*C3c[1]-C2c[0]*R3c[3]; S1[2] = R2c[0]*C3c[2]-C2c[0]*R3c[6];
	S1[3] = R2c[3]*C3c[0]-C2c[1]*R3c[0]; S1[4] = R2c[3]*C3c[1]-C2c[1]*R3c[3]; S1[5] = R2c[3]*C3c[2]-C2c[1]*R3c[6];
	S1[6] = R2c[6]*C3c[0]-C2c[2]*R3c[0]; S1[7] = R2c[6]*C3c[1]-C2c[2]*R3c[3]; S1[8] = R2c[6]*C3c[2]-C2c[2]*R3c[6];

	S2[0] = R2c[1]*C3c[0]-C2c[0]*R3c[1]; S2[1] = R2c[1]*C3c[1]-C2c[0]*R3c[4]; S2[2] = R2c[1]*C3c[2]-C2c[0]*R3c[7];
	S2[3] = R2c[4]*C3c[0]-C2c[1]*R3c[1]; S2[4] = R2c[4]*C3c[1]-C2c[1]*R3c[4]; S2[5] = R2c[4]*C3c[2]-C2c[1]*R3c[7];
	S2[6] = R2c[7]*C3c[0]-C2c[2]*R3c[1]; S2[7] = R2c[7]*C3c[1]-C2c[2]*R3c[4]; S2[8] = R2c[7]*C3c[2]-C2c[2]*R3c[7];

	S3[0] = R2c[2]*C3c[0]-C2c[0]*R3c[2]; S3[1] = R2c[2]*C3c[1]-C2c[0]*R3c[5]; S3[2] = R2c[2]*C3c[2]-C2c[0]*R3c[8];
	S3[3] = R2c[5]*C3c[0]-C2c[1]*R3c[2]; S3[4] = R2c[5]*C3c[1]-C2c[1]*R3c[5]; S3[5] = R2c[5]*C3c[2]-C2c[1]*R3c[8];
	S3[6] = R2c[8]*C3c[0]-C2c[2]*R3c[2]; S3[7] = R2c[8]*C3c[1]-C2c[2]*R3c[5]; S3[8] = R2c[8]*C3c[2]-C2c[2]*R3c[8];

	transpose3x3(S1,T1);
	transpose3x3(S2,T2);
	transpose3x3(S3,T3);
}

Warping::Warping( int resoX, double *K1, double *CM1, double *K2, double *CM2, double *K3, double *CM3, double *postT)
{
	// save base transforms
	memcpy(this->base1,CM1,sizeof(double)*16);
	memcpy(this->base2,CM2,sizeof(double)*16);
	memcpy(this->base3,CM3,sizeof(double)*16);
	memcpy(this->origBase3,CM3,sizeof(double)*16);
	memcpy(this->postT,postT,sizeof(double)*16);

	// setup mappings
	identityMatrix4x4(this->Mrel);
	
	double CM3new[16];
	matrixMult4x4(this->postT,this->base3,CM3new);
	cudaMalloc( (void**)&matrixDataDev,6*9*sizeof(float));

	memcpy(this->KLo, K1, sizeof(double)*9);
	memcpy(this->KRo, K2, sizeof(double)*9);
	memcpy(this->K3o, K3, sizeof(double)*9);
	prevResoUpdatedDev = 0;
	setImageResolution(resoX,false);
	//memset(matrixData,0,sizeof(float)*6*9);
	setupExtrinsic(base1,base2,CM3new);//base3);
	precomputeTrifocalDiff();
	// stop timer
	cudaTransfer0 = 0;
	cudaProcess = 0;
	cudaTransfer1 = 0;
	nCudaMeasurements = 0;

	cpuProcess = 0;
	nCpuMeasurements = 0;
}

double Warping::det3x3(double *mat)
{
	double det;
	det = mat[0] * ( mat[4]*mat[8] - mat[7]*mat[5] )
		- mat[1] * ( mat[3]*mat[8] - mat[6]*mat[5] )
		+ mat[2] * ( mat[3]*mat[7] - mat[6]*mat[4] );
	return det;
}

void Warping::inverse3x3( double *ma, double *mr)
{
	double det = det3x3(ma);
	double t[9];

	t[0] =  ( ma[4]*ma[8] - ma[5]*ma[7] ) / det;
	t[1] = -( ma[1]*ma[8] - ma[7]*ma[2] ) / det;
	t[2] =  ( ma[1]*ma[5] - ma[4]*ma[2] ) / det;
	t[3] = -( ma[3]*ma[8] - ma[5]*ma[6] ) / det;
	t[4] =  ( ma[0]*ma[8] - ma[6]*ma[2] ) / det;
	t[5] = -( ma[0]*ma[5] - ma[3]*ma[2] ) / det;
	t[6] =  ( ma[3]*ma[7] - ma[6]*ma[4] ) / det;
	t[7] = -( ma[0]*ma[7] - ma[6]*ma[1] ) / det;
	t[8] =  ( ma[0]*ma[4] - ma[1]*ma[3] ) / det;

	memcpy(mr,t,sizeof(double)*9);
}


void Warping::verticalLine(float *P2, double *LN2)
{		
	LN2[0] = 1;
	LN2[1] = 0;
	LN2[2] = -P2[0];
}

void Warping::verticalLineDiff(double *dL)
{		
	dL[0] = 0;
	dL[1] = 0;
	dL[2] = -1;
}

inline void setCol3(double *M, double *col, int index)
{
	M[index+0*3] = col[0];
	M[index+1*3] = col[1];
	M[index+2*3] = col[2];
}

void Warping::warp( float *pointsR, float *pointsL, int nPoints, float *outPoints, float *outPoints3)
{
	for (int i = 0; i < nPoints; i++) {
		double h1[3] = {0, 0, 0}, h2[3] = {0,0,0}, h3[3]={0,0,0};
		double l[3],ltmp[3];
		verticalLine(&pointsR[i*2],ltmp);
		matrixMultVec3(tKR, ltmp, l);

		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);

		double H[9];
		setCol3(H,h1,0);
		setCol3(H,h2,1);
		setCol3(H,h3,2);
		double P[3],R[3];
		matrixMultVec2(iKL, &pointsL[i*2], P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);
		
		outPoints3[i*3+0] = float(P[0]);
		outPoints3[i*3+1] = float(P[1]);
		outPoints3[i*3+2] = float(P[2]);

		outPoints[i*2+0] = float(P[0]/P[2]);
		outPoints[i*2+1] = float(P[1]/P[2]);
	} 
}

void Warping::warp( float *pointsR, float *pointsL, int nPoints, float *outPoints)
{
	for (int i = 0; i < nPoints; i++) {
		double h1[3] = {0, 0, 0}, h2[3] = {0,0,0}, h3[3]={0,0,0};
		double l[3],ltmp[3];
		verticalLine(&pointsR[i*2],ltmp);
		matrixMultVec3(tKR, ltmp, l);

		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);

		double H[9];
		setCol3(H,h1,0);
		setCol3(H,h2,1);
		setCol3(H,h3,2);
		double P[3],R[3];
		matrixMultVec2(iKL, &pointsL[i*2], P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);

		outPoints[i*2+0] = float(P[0]/P[2]);
		outPoints[i*2+1] = float(P[1]/P[2]);
	} 
}

void convertDouble2Float(double *dbl, float *flt, int cnt) {
	for (int i = 0; i < cnt; i++) flt[i] = float(dbl[i]);
}

void Warping::warpPointLineCuda(float *pointsR0Dev, float *linesL0Dev, int nPoints, float *linesL1Dev) {
	if (nPoints <= 0) return;
	cudaPointLineWarp(pointsR0Dev,linesL0Dev,nPoints,matrixDataDev,linesL1Dev);
}


void Warping::warpCuda( float *pointsRDev, float *pointsLDev, int nPoints, float *outPointsDev)
{	
	if (nPoints <= 0) return;
	cudaWarp(pointsRDev,pointsLDev,nPoints,matrixDataDev,outPointsDev);
//	warpTimer.StopCounter();
//	cudaProcess += 1000*warpTimer.GetElapsedTime();
//	nCudaMeasurements++;
}

void Warping::invertRT( double *R, double *t, double *Ri, double *ti )
{
	transpose3x3(R,Ri);
	matrixMultVec3(Ri,t,ti); ti[0] = -ti[0]; ti[1] = -ti[1]; ti[2] = -ti[2];
}

void Warping::invertRT( float *R, float *t, float *Ri, float *ti )
{
	transpose3x3(R,Ri);
	matrixMultVec3(Ri,t,ti); ti[0] = -ti[0]; ti[1] = -ti[1]; ti[2] = -ti[2];
}

void Warping::invertRT4( double *M, double *Mi )
{
	double R[9],Ri[9];
	double t[3],ti[3];
	// extract R,t
	copy3x3(M,R); t[0]  = M[3]; t[1] = M[7]; t[2] = M[11];
	invertRT(R,t,Ri,ti);
	
	Mi[0]  = Ri[0]; Mi[1]  = Ri[1]; Mi[2]  = Ri[2];  Mi[3]  = ti[0];
	Mi[4]  = Ri[3]; Mi[5]  = Ri[4]; Mi[6]  = Ri[5];  Mi[7]  = ti[1];
	Mi[8]  = Ri[6]; Mi[9]  = Ri[7]; Mi[10] = Ri[8];  Mi[11] = ti[2];
	Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;      Mi[15] = 1;
}

void Warping::invertRT4( float *M, float *Mi )
{
	float R[9],Ri[9];
	float t[3],ti[3];
	// extract R,t
	copy3x3(M,R); t[0]  = M[3]; t[1] = M[7]; t[2] = M[11];
	invertRT(R,t,Ri,ti);

	Mi[0]  = Ri[0]; Mi[1]  = Ri[1]; Mi[2]  = Ri[2];  Mi[3]  = ti[0];
	Mi[4]  = Ri[3]; Mi[5]  = Ri[4]; Mi[6]  = Ri[5];  Mi[7]  = ti[1];
	Mi[8]  = Ri[6]; Mi[9]  = Ri[7]; Mi[10] = Ri[8];  Mi[11] = ti[2];
	Mi[12] = 0;    Mi[13] = 0;    Mi[14] = 0;     Mi[15] = 1;
}

void Warping::canonizeTransformation(double *R1, double *C1, double *R2, double *C2, double *R3, double *C3, double *R2c, double *C2c, double *R3c, double *C3c) {
	// invert R1,t1
	double R1i[9],C1i[3];
	invertRT(R1,C1,R1i,C1i);
	// canonize R2 and R3
	matrixMult3x3(R1i,R2,R2c);
	matrixMult3x3(R1i,R3,R3c);
	// canonize t2 and t2
	matrixMultVec3(R1i,C2,C2c); C2c[0] += C1i[0]; C2c[1] += C1i[1]; C2c[2] += C1i[2];
	matrixMultVec3(R1i,C3,C3c); C3c[0] += C1i[0]; C3c[1] += C1i[1]; C3c[2] += C1i[2];
}

bool isIdentity(double *M) {
	double sumVal = 0;
	for (int i = 0; i < 16; i++) sumVal += M[i];
	if (sumVal != 4.0f) return false;
	if (M[0] != 1.0f) return false;
	if (M[5] != 1.0f) return false;
	if (M[10] != 1.0f) return false;
	if (M[15] != 1.0f) return false;
	return true;
}

void Warping::setupExtrinsic( double *CM1, double *CM2, double *CM3 )
{
	double R1[9],R2[9],R3[9];
	double C1[3],C2[3],C3[3];
	// extract R,t
	copy3x3(CM1,R1);     C1[0]  = CM1[3]; C1[1] = CM1[7]; C1[2] = CM1[11];
	copy3x3(CM2,R2);	 C2[0]  = CM2[3]; C2[1] = CM2[7]; C2[2] = CM2[11];
	copy3x3(CM3,R3);	 C3[0]  = CM3[3]; C3[1] = CM3[7]; C3[2] = CM3[11];
	// canonize and invert R t
	double iR2c[9],iR3c[9],iC2c[3],iC3c[3];
	if (!isIdentity(CM1)) {
		double R2c[9],R3c[9],C2c[3],C3c[3];	
		canonizeTransformation(R1,C1,R2,C2,R3,C3,R2c,C2c,R3c,C3c);
		invertRT(R2c,C2c,iR2c,iC2c);
		invertRT(R3c,C3c,iR3c,iC3c);
	} else {
		invertRT(R2,C2,iR2c,iC2c);
		invertRT(R3,C3,iR3c,iC3c);		
	}
	// generate tensor slice transposes
	generateTensor(iR2c,iC2c,iR3c,iC3c,this->T1,this->T2,this->T3);
	updateCudaMatrixData();
}

void Warping::setRelativeTransform( double *M )
{
	memcpy(this->Mrel,M,sizeof(double)*16);
	double tmp[16],CM3new[16];
	matrixMult4x4(this->base3,this->Mrel,tmp);
	matrixMult4x4(tmp,this->postT,CM3new);
	setupExtrinsic(base1,base2,CM3new);
}

void Warping::setTransformSE3( double *x )
{
	double A[16];
	A[0] = 0;		  A[1] = -x[2];   A[2] = x[1];	A[3]  = x[3];
	A[4] = x[2];	  A[5] = 0;		  A[6] =-x[0];	A[7]  = x[4];
	A[8] =-x[1];	  A[9] = x[0];	  A[10] = 0;	A[11] = x[5];
	A[12] = 0;		  A[13] = 0;	  A[14] = 0;	A[15] = 0;
	double Md[16];
	expm(4,A,1.0,Md,6);
	setRelativeTransform(Md);
}

void Warping::resetBase()
{
	identityMatrix4x4(this->Mrel);
	
	double CM3new[16];
	matrixMult4x4(this->base3,this->postT,CM3new);
	setupExtrinsic(base1,base2,CM3new);
	setMarker();
}

void Warping::mulTransformSE3( float *xf )
{
	double x[6]; for (int i = 0; i < 6; i++) x[i] = double(xf[i]);
	double A[16];

	A[0]  =    0;	A[1] = -x[2];   A[2]  = x[1]; A[3]  = x[3];
	A[4]  = x[2];	A[5] =     0;   A[6]  =-x[0]; A[7]  = x[4];
	A[8]  =-x[1];	A[9] =  x[0];	A[10] =    0; A[11] = x[5];
	A[12] =    0;	A[13] =    0;	A[14] =    0; A[15] =    0;

	double Md[16];
	expm(4,A,1.0,Md,6);
	double Mf[16];
	for (int i = 0; i < 16; i++) Mf[i] = double(Md[i]);
	matrixMult4x4(this->Mrel,Mf,this->Mrel);

	// re-generate CM3 using new base
	double tmp[16],CM3new[16];
	matrixMult4x4(this->base3,this->Mrel,tmp);
	matrixMult4x4(tmp,this->postT,CM3new);
	setupExtrinsic(base1,base2,CM3new);
}

void Warping::precomputeTrifocalDiffIntrinsic()
{
	// precompute multiplication by K'
	matrixMult3x3(dT1_1,tKR,dTK1_1); matrixMult3x3(dT2_1,tKR,dTK2_1); matrixMult3x3(dT3_1,tKR,dTK3_1);
	matrixMult3x3(dT1_2,tKR,dTK1_2); matrixMult3x3(dT2_2,tKR,dTK2_2); matrixMult3x3(dT3_2,tKR,dTK3_2);
	matrixMult3x3(dT1_3,tKR,dTK1_3); matrixMult3x3(dT2_3,tKR,dTK2_3); matrixMult3x3(dT3_3,tKR,dTK3_3);
	matrixMult3x3(dT1_4,tKR,dTK1_4); matrixMult3x3(dT2_4,tKR,dTK2_4); matrixMult3x3(dT3_4,tKR,dTK3_4);
	matrixMult3x3(dT1_5,tKR,dTK1_5); matrixMult3x3(dT2_5,tKR,dTK2_5); matrixMult3x3(dT3_5,tKR,dTK3_5);
	matrixMult3x3(dT1_6,tKR,dTK1_6); matrixMult3x3(dT2_6,tKR,dTK2_6); matrixMult3x3(dT3_6,tKR,dTK3_6);
}

void Warping::precomputeTrifocalDiff()
{
	double R1[9],R2[9],R3[9];
	double C1[3],C2[3],C3[3];
	// extract R,t
	copy3x3(this->base1,R1); C1[0]  = this->base1[3]; C1[1] = this->base1[7]; C1[2] = this->base1[11];
	copy3x3(this->base2,R2); C2[0]  = this->base2[3]; C2[1] = this->base2[7]; C2[2] = this->base2[11];
	copy3x3(this->base3,R3); C3[0]  = this->base3[3]; C3[1] = this->base3[7]; C3[2] = this->base3[11];
	// canonize and invert [R2 t2]
	double iR2c[9],iC2c[3];
	if (!isIdentity(this->base1)) {
		double R2c[9],R3c[9],C2c[3],C3c[3];	
		canonizeTransformation(R1,C1,R2,C2,R3,C3,R2c,C2c,R3c,C3c);
		invertRT(R2c,C2c,iR2c,iC2c);
	} else {
		invertRT(R2,C2,iR2c,iC2c);
	}
	/*
	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
	*/
	double dA1[16],dA2[16],dA3[16],dA4[16],dA5[16],dA6[16];
	// x is inverted here! 
	// dexp(A(x))/dx = A(x) around zero
	// exp(A(x))*exp(-A(x)) = I, thus exp(-A(x)) is the inverse
	// A(x) is linear function -> -A(x) = A(-x)
	memset(dA1,0,sizeof(double)*16); dA1[6]  =  1; dA1[9] = -1;
	memset(dA2,0,sizeof(double)*16); dA2[2]  = -1; dA2[8] =  1;
	memset(dA3,0,sizeof(double)*16); dA3[1]  =  1; dA3[4] = -1;
	memset(dA4,0,sizeof(double)*16); dA4[3]  = -1;
	memset(dA5,0,sizeof(double)*16); dA5[7]  = -1;
	memset(dA6,0,sizeof(double)*16); dA6[11] = -1;
	// generate inv([dR3(0),dt3(0)]) for each param x
	double dR1[9],dR2[9],dR3[9],dR4[9],dR5[9],dR6[9];
	double dC1[3],dC2[9],dC3[3],dC4[3],dC5[3],dC6[3];
	copy3x3(dA1,dR1); dC1[0] = dA1[3]; dC1[1] = dA1[7]; dC1[2] = dA1[11];
	copy3x3(dA2,dR2); dC2[0] = dA2[3]; dC2[1] = dA2[7]; dC2[2] = dA2[11];
	copy3x3(dA3,dR3); dC3[0] = dA3[3]; dC3[1] = dA3[7]; dC3[2] = dA3[11];
	copy3x3(dA4,dR4); dC4[0] = dA4[3]; dC4[1] = dA4[7]; dC4[2] = dA4[11];
	copy3x3(dA5,dR5); dC5[0] = dA5[3]; dC5[1] = dA5[7]; dC5[2] = dA5[11];
	copy3x3(dA6,dR6); dC6[0] = dA6[3]; dC6[1] = dA6[7]; dC6[2] = dA6[11];
	// generate tensor differentials
	generateTensor(iR2c,iC2c,dR1,dC1,dT1_1,dT2_1,dT3_1);
	generateTensor(iR2c,iC2c,dR2,dC2,dT1_2,dT2_2,dT3_2);
	generateTensor(iR2c,iC2c,dR3,dC3,dT1_3,dT2_3,dT3_3);
	generateTensor(iR2c,iC2c,dR4,dC4,dT1_4,dT2_4,dT3_4);
	generateTensor(iR2c,iC2c,dR5,dC5,dT1_5,dT2_5,dT3_5);
	generateTensor(iR2c,iC2c,dR6,dC6,dT1_6,dT2_6,dT3_6);

	precomputeTrifocalDiffIntrinsic();
}

//void Warping::warpDiff( float *pointsR, float *pointsL, int nPoints, float *out1, float *out2, float *out3, float *out4, float *out5, float *out6)
void Warping::warpDiff( float *pointsR, float *pointsL, int nPoints, float *out1, float *out2, float *out3, float *out4, float *out5, float *out6, float *outD)
{
	double N[9];
	double h1[3] = {0, 0, 0},    h2[3] = {0,0,0},    h3[3]={0,0,0};
	double dh1_d[3] = {0, 0, 0}, dh2_d[3] = {0,0,0}, dh3_d[3]={0,0,0};
	double dh1_1[3] = {0, 0, 0}, dh2_1[3] = {0,0,0}, dh3_1[3]={0,0,0};
	double dh1_2[3] = {0, 0, 0}, dh2_2[3] = {0,0,0}, dh3_2[3]={0,0,0};
	double dh1_3[3] = {0, 0, 0}, dh2_3[3] = {0,0,0}, dh3_3[3]={0,0,0};
	double dh1_4[3] = {0, 0, 0}, dh2_4[3] = {0,0,0}, dh3_4[3]={0,0,0};
	double dh1_5[3] = {0, 0, 0}, dh2_5[3] = {0,0,0}, dh3_5[3]={0,0,0};
	double dh1_6[3] = {0, 0, 0}, dh2_6[3] = {0,0,0}, dh3_6[3]={0,0,0};

	for (int i = 0; i < nPoints; i++) {
		double l[3],ltmp[3],ldtmp[3],ld[3];
		verticalLine(&pointsR[i*2],ltmp);
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);

		verticalLineDiff(ldtmp);
		matrixMultVec3(tKR, ldtmp, ld);
		matrixMultVec3(T1, ld, dh1_d);
		matrixMultVec3(T2, ld, dh2_d);
		matrixMultVec3(T3, ld, dh3_d);

		double H[9];
		setCol3(H,h1,0);
		setCol3(H,h2,1);
		setCol3(H,h3,2);

		double dHd[9];
		setCol3(dHd,dh1_d,0);
		setCol3(dHd,dh2_d,1);
		setCol3(dHd,dh3_d,2);

		// compute P3
		double P3[3],PG[3],tmp1[3],tmp2[3];
		matrixMultVec2(iKL, &pointsL[i*2], PG);
		matrixMultVec3(H, PG, tmp2);
		matrixMultVec3(K3, tmp2, P3);
		// compute linearized normalization
		double x1 = P3[0]; double x2 = P3[1]; double x3 = P3[2];
		N[0] = 1.0f/x3; N[1] =       0; N[2] = -x1/(x3*x3);
		N[3] =       0; N[4] = 1.0f/x3; N[5] = -x2/(x3*x3);
		N[6] =       0; N[7] =       0; N[8] =           1;
		
		//////////////////////////////
		// COMPUTE point differentials
		//////////////////////////////

		// generate dH/dx1
		matrixMultVec3(dTK1_1, ltmp, dh1_1);
		matrixMultVec3(dTK2_1, ltmp, dh2_1);
		matrixMultVec3(dTK3_1, ltmp, dh3_1);
		// generate dH/dx2
		matrixMultVec3(dTK1_2, ltmp, dh1_2);
		matrixMultVec3(dTK2_2, ltmp, dh2_2);
		matrixMultVec3(dTK3_2, ltmp, dh3_2);
		// generate dH/dx3
		matrixMultVec3(dTK1_3, ltmp, dh1_3);
		matrixMultVec3(dTK2_3, ltmp, dh2_3);
		matrixMultVec3(dTK3_3, ltmp, dh3_3);
		// generate dH/dx4
		matrixMultVec3(dTK1_4, ltmp, dh1_4);
		matrixMultVec3(dTK2_4, ltmp, dh2_4);
		matrixMultVec3(dTK3_4, ltmp, dh3_4);
		// generate dH/dx5
		matrixMultVec3(dTK1_5, ltmp, dh1_5);
		matrixMultVec3(dTK2_5, ltmp, dh2_5);
		matrixMultVec3(dTK3_5, ltmp, dh3_5);
		// generate dH/dx6
		matrixMultVec3(dTK1_6, ltmp, dh1_6);
		matrixMultVec3(dTK2_6, ltmp, dh2_6);
		matrixMultVec3(dTK3_6, ltmp, dh3_6);

		// compute dP/dx1
		double dP[3]; double dH[9]; 
		setCol3(dH,dh1_1,0); setCol3(dH,dh2_1,1); setCol3(dH,dh3_1,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out1[i*2+0] = float(dP[0]);
		out1[i*2+1] = float(dP[1]);
		
		// compute dP/dx2
		setCol3(dH,dh1_2,0); setCol3(dH,dh2_2,1); setCol3(dH,dh3_2,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out2[i*2+0] = float(dP[0]);
		out2[i*2+1] = float(dP[1]);

		// compute dP/dx3
		setCol3(dH,dh1_3,0); setCol3(dH,dh2_3,1); setCol3(dH,dh3_3,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out3[i*2+0] = float(dP[0]);
		out3[i*2+1] = float(dP[1]);

		// compute dP/dx4
		setCol3(dH,dh1_4,0); setCol3(dH,dh2_4,1); setCol3(dH,dh3_4,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out4[i*2+0] = float(dP[0]);
		out4[i*2+1] = float(dP[1]);

		// compute dP/dx5
		setCol3(dH,dh1_5,0); setCol3(dH,dh2_5,1); setCol3(dH,dh3_5,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out5[i*2+0] = float(dP[0]);
		out5[i*2+1] = float(dP[1]);
		
		// compute dP/dx6
		setCol3(dH,dh1_6,0); setCol3(dH,dh2_6,1); setCol3(dH,dh3_6,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		out6[i*2+0] = float(dP[0]);
		out6[i*2+1] = float(dP[1]);

		// compute dP/dd
		setCol3(dH,dh1_d,0); setCol3(dH,dh2_d,1); setCol3(dH,dh3_d,2);
		matrixMultVec3(dH, PG, tmp2);
		matrixMultVec3(K3, tmp2, tmp1);
		matrixMultVec3(N, tmp1, dP);
		outD[i*2+0] = float(dP[0]);
		outD[i*2+1] = float(dP[1]);
	} 
}

void Warping::setMarker()
{
	invertRT4(Mrel,invMarker);
}

double *Warping::getRelativeTransformToMarker()
{
	matrixMult4x4(invMarker,Mrel,MrelMarker);
	return (double*)MrelMarker;
}

double *Warping::getRelativeTransformInv()
{
	invertRT4(Mrel,invTmp);
	return &invTmp[0];
}

double *Warping::getRelativeTransform()
{
	return &Mrel[0];
}

double *Warping::getKR()
{
	return &KR[0];
}

double *Warping::getKL()
{
	return &KL[0];
}

Warping::~Warping()
{
	// Free device memory
	cudaFree ( matrixDataDev);
}

void Warping::updateCudaMatrixData()
{
	convertDouble2Float(tKR,&matrixData[0*9],9);
	convertDouble2Float(T1,&matrixData[1*9],9);
	convertDouble2Float(T2,&matrixData[2*9],9);
	convertDouble2Float(T3,&matrixData[3*9],9);
	convertDouble2Float(iKL,&matrixData[4*9],9);
	convertDouble2Float(K3,&matrixData[5*9],9);
	cudaMemcpy( matrixDataDev, matrixData, 6*9*sizeof(float), cudaMemcpyHostToDevice );
}

double *Warping::getNormalizedKR() {
	return &KRo[0];
}

void Warping::setImageResolution( int resoX, bool updateDeviceFlag)
{
	for (int i = 0; i < 3; i++) {
		KL[i] = KLo[i]*double(resoX);
		KR[i] = KRo[i]*double(resoX);
		K3[i] = K3o[i]*double(resoX);
		KL[i+3] = KLo[i+3]*double(resoX);
		KR[i+3] = KRo[i+3]*double(resoX);
		K3[i+3] = K3o[i+3]*double(resoX);
	}
	KL[6] = 0; KL[7] = 0;  KL[8] = 1;
	KR[6] = 0; KR[7] = 0;  KR[8] = 1;
	K3[6] = 0; K3[7] = 0;  K3[8] = 1;

	transpose3x3(this->KL,this->tKL);
	transpose3x3(this->KR,this->tKR);
	transpose3x3(this->K3,this->tK3);
	inverse3x3(this->KL,this->iKL);
	inverse3x3(this->KR,this->iKR);
	inverse3x3(this->K3,this->iK3);
	precomputeTrifocalDiffIntrinsic();
	if (updateDeviceFlag && resoX != prevResoUpdatedDev) { updateCudaMatrixData(); prevResoUpdatedDev=resoX; }
}
