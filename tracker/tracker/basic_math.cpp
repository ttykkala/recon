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


//#include "math.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <tracker/basic_math.h>
#include <float.h>
#include <vector>
#include <cmath>
//#include <ctgmath>

using namespace std;

#if !defined(M_PI)
#define M_PI 3.141592653f
#endif

float rad2deg(float x) {
    return (180.0*(x)/M_PI);
}
float deg2rad(float x) { return M_PI*x/180.0f; }
void dumpMatrix(const char *str, const float *M, int rows, int cols) {
	printf("%s:\n",str);
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++)
			printf("%e ",M[i+j*cols]);
		printf("\n");
	}
    fflush(stdout);
}
void dumpMatrix(const char *str, const double *M, int rows, int cols) {
	printf("%s:\n",str);
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++)
			printf("%e ",M[i+j*cols]);
		printf("\n");
	}
    fflush(stdout);
}

void identity3x4(double *CM)
{
	CM[0] = 1; CM[1] = 0; CM[2] = 0;  CM[3] = 0;
	CM[4] = 0; CM[5] = 1; CM[6] = 0;  CM[7] = 0;
	CM[8] = 0; CM[9] = 0; CM[10] = 1;  CM[11] = 0;
}

void identity4x4(double *M4x4)
{
	M4x4[0] = 1; M4x4[1] = 0; M4x4[2] = 0; M4x4[3] = 0;
	M4x4[4] = 0; M4x4[5] = 1; M4x4[6] = 0; M4x4[7] = 0; 
	M4x4[8] = 0; M4x4[9] = 0; M4x4[10] = 1; M4x4[11] = 0;
	M4x4[12] = 0; M4x4[13] = 0; M4x4[14] = 0; M4x4[15] = 1;
}

void identity4x4(float *M4x4)
{
	M4x4[0] = 1; M4x4[1] = 0; M4x4[2] = 0; M4x4[3] = 0;
	M4x4[4] = 0; M4x4[5] = 1; M4x4[6] = 0; M4x4[7] = 0; 
	M4x4[8] = 0; M4x4[9] = 0; M4x4[10] = 1; M4x4[11] = 0;
	M4x4[12] = 0; M4x4[13] = 0; M4x4[14] = 0; M4x4[15] = 1;
}

float length3(float *vec) {
    return sqrtf(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]+1e-8f);
}

void matrixMult4x4(float *M1, float *M2, float *R)
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

void matrixMult4x4(double *M1, float *M2, float *R)
{
    float Rtmp[16];
    Rtmp[0] = (float)(M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12]);
    Rtmp[1] = (float)(M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13]);
    Rtmp[2] = (float)(M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14]);
    Rtmp[3] = (float)(M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15]);

    Rtmp[4] = (float)(M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12]);
    Rtmp[5] = (float)(M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13]);
    Rtmp[6] = (float)(M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14]);
    Rtmp[7] = (float)(M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15]);

    Rtmp[8]  = (float)(M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12]);
    Rtmp[9]  = (float)(M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13]);
    Rtmp[10] = (float)(M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14]);
    Rtmp[11] = (float)(M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15]);

    Rtmp[12] = (float)(M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12]);
    Rtmp[13] = (float)(M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13]);
    Rtmp[14] = (float)(M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14]);
    Rtmp[15] = (float)(M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15]);
    memcpy(R,Rtmp,sizeof(float)*16);
}

void matrixMult4x4(float *M1, double *M2, float *R)
{
    float Rtmp[16];
    Rtmp[0] = (float)(M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12]);
    Rtmp[1] = (float)(M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13]);
    Rtmp[2] = (float)(M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14]);
    Rtmp[3] = (float)(M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15]);

    Rtmp[4] = (float)(M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12]);
    Rtmp[5] = (float)(M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13]);
    Rtmp[6] = (float)(M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14]);
    Rtmp[7] = (float)(M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15]);

    Rtmp[8]  = (float)(M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12]);
    Rtmp[9]  = (float)(M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13]);
    Rtmp[10] = (float)(M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14]);
    Rtmp[11] = (float)(M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15]);

    Rtmp[12] = (float)(M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12]);
    Rtmp[13] = (float)(M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13]);
    Rtmp[14] = (float)(M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14]);
    Rtmp[15] = (float)(M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15]);
    memcpy(R,Rtmp,sizeof(float)*16);
}

void matrixMult4x4RT(float *M1, float *M2, float *R)
{
	R[0] = M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8];
	R[1] = M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9];
	R[2] = M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10];
	R[3] = M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11] + M1[3];

	R[4] = M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8];
	R[5] = M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9];
	R[6] = M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10];
	R[7] = M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11] + M1[7];

	R[8]  = M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8];
	R[9]  = M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9];
	R[10] = M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10];
	R[11] = M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11] + M1[11];

	R[12] = 0;
	R[13] = 0;
	R[14] = 0;
	R[15] = 1;
}

void matrixMult4x4(double *M1, double *M2, double *R)
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

void transpose3x3(double *M3x3, double *R3x3)
{
	R3x3[0] = M3x3[0]; R3x3[1] = M3x3[3]; R3x3[2] = M3x3[6];
	R3x3[3] = M3x3[1]; R3x3[4] = M3x3[4]; R3x3[5] = M3x3[7];
	R3x3[6] = M3x3[2]; R3x3[7] = M3x3[5]; R3x3[8] = M3x3[8];
}
void transpose3x3(float *M3x3, float *R3x3)
{
	R3x3[0] = M3x3[0]; R3x3[1] = M3x3[3]; R3x3[2] = M3x3[6];
	R3x3[3] = M3x3[1]; R3x3[4] = M3x3[4]; R3x3[5] = M3x3[7];
	R3x3[6] = M3x3[2]; R3x3[7] = M3x3[5]; R3x3[8] = M3x3[8];
}

void transpose4x4(float *M4x4, float *R4x4) {
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            R4x4[i+j*4] = M4x4[j+i*4];
        }
    }

}

void transpose4x4(double *M4x4, double *R4x4) {
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            R4x4[i+j*4] = M4x4[j+i*4];
        }
    }

}


void copy3x3(double *M3x4, double *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[1]; R3x3[2] = M3x4[2];
	R3x3[3] = M3x4[4]; R3x3[4] = M3x4[5]; R3x3[5] = M3x4[6];
	R3x3[6] = M3x4[8]; R3x3[7] = M3x4[9]; R3x3[8] = M3x4[10];
}

void copy3x3f(double *M3x4, float *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[1]; R3x3[2] = M3x4[2];
	R3x3[3] = M3x4[4]; R3x3[4] = M3x4[5]; R3x3[5] = M3x4[6];
	R3x3[6] = M3x4[8]; R3x3[7] = M3x4[9]; R3x3[8] = M3x4[10];
}

void copyT3x1(float *M4, float *T3x1) {
	T3x1[0] = M4[3];
	T3x1[1] = M4[7];
	T3x1[2] = M4[11];
}

void copy3x3(float *M3x4, float *R3x3)
{
	R3x3[0] = M3x4[0]; R3x3[1] = M3x4[1]; R3x3[2] = M3x4[2];
	R3x3[3] = M3x4[4]; R3x3[4] = M3x4[5]; R3x3[5] = M3x4[6];
	R3x3[6] = M3x4[8]; R3x3[7] = M3x4[9]; R3x3[8] = M3x4[10];
}

void matrixMultVec3(float *M1, float *V, float *R)
{
	float Rt[3];
	Rt[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
	Rt[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5]*V[2];
	Rt[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8]*V[2];
	R[0] = Rt[0];
	R[1] = Rt[1];
	R[2] = Rt[2];
}

void matrixMultVec4(double *M1, float *V, float *R)
{
	float Rt[3];
	Rt[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2]+M1[3];
	Rt[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2]+M1[7];
	Rt[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2]+M1[11];
	R[0] = Rt[0];
	R[1] = Rt[1];
	R[2] = Rt[2];
}

void matrixMultVec3(double *M1, double *V, double *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
	R[1] = M1[3]*V[0]+M1[4]*V[1]+M1[5]*V[2];
	R[2] = M1[6]*V[0]+M1[7]*V[1]+M1[8]*V[2];
}

void transformRT3(double *M1, float *V, float *R)
{
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2]+M1[3];
	R[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2]+M1[7];
	R[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2]+M1[11];
}

void transformRT3(float *M1, float *V, float *R) {
	R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2]+M1[3];
	R[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2]+M1[7];
	R[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2]+M1[11];
}


void rotate3(double *M1, double *V, double *R) {
    R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
    R[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2];
    R[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2];
}

void rotate3(float *M1, float *V, float *R) {
    R[0] = M1[0]*V[0]+M1[1]*V[1]+M1[2]*V[2];
    R[1] = M1[4]*V[0]+M1[5]*V[1]+M1[6]*V[2];
    R[2] = M1[8]*V[0]+M1[9]*V[1]+M1[10]*V[2];
}


void invertRT( double *R, double *t, double *Ri, double *ti )
{
	transpose3x3(R,Ri);
	matrixMultVec3(Ri,t,ti); ti[0] = -ti[0]; ti[1] = -ti[1]; ti[2] = -ti[2];
}

void invertRT( float *R, float *t, float *Ri, float *ti )
{
	transpose3x3(R,Ri);
	matrixMultVec3(Ri,t,ti); ti[0] = -ti[0]; ti[1] = -ti[1]; ti[2] = -ti[2];
}

void invertRT4( double *M, double *Mi )
{
	Mi[0]  = M[0]; Mi[1]  = M[4]; Mi[2]  = M[8];   Mi[3]  = -(M[0]*M[3]+M[4]*M[7]+M[8]*M[11]);
	Mi[4]  = M[1]; Mi[5]  = M[5]; Mi[6]  = M[9];   Mi[7]  = -(M[1]*M[3]+M[5]*M[7]+M[9]*M[11]);
	Mi[8]  = M[2]; Mi[9]  = M[6]; Mi[10] = M[10];  Mi[11] = -(M[2]*M[3]+M[6]*M[7]+M[10]*M[11]);
	Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;    Mi[15] = 1;
/*
	double R[9],Ri[9];
	double t[3],ti[3];
	// extract R,t
	copy3x3(M,R); t[0]  = M[3]; t[1] = M[7]; t[2] = M[11];
	invertRT(R,t,Ri,ti);
	
	Mi[0]  = Ri[0]; Mi[1]  = Ri[1]; Mi[2]  = Ri[2];  Mi[3]  = ti[0];
	Mi[4]  = Ri[3]; Mi[5]  = Ri[4]; Mi[6]  = Ri[5];  Mi[7]  = ti[1];
	Mi[8]  = Ri[6]; Mi[9]  = Ri[7]; Mi[10] = Ri[8];  Mi[11] = ti[2];
	Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;      Mi[15] = 1;
*/
}

void invertRT4( float *M, float *Mi )
{
/*	float R[9],Ri[9];
	float t[3],ti[3];
	// extract R,t
	copy3x3(M,R); t[0]  = M[3]; t[1] = M[7]; t[2] = M[11];
	invertRT(R,t,Ri,ti);
	*/
//	transpose3x3(R,Ri);
//	matrixMultVec3(Ri,t,ti); ti[0] = -ti[0]; ti[1] = -ti[1]; ti[2] = -ti[2];
	Mi[0]  = M[0]; Mi[1]  = M[4]; Mi[2]  = M[8];   Mi[3]  = -(M[0]*M[3]+M[4]*M[7]+M[8]*M[11]);
	Mi[4]  = M[1]; Mi[5]  = M[5]; Mi[6]  = M[9];   Mi[7]  = -(M[1]*M[3]+M[5]*M[7]+M[9]*M[11]);
	Mi[8]  = M[2]; Mi[9]  = M[6]; Mi[10] = M[10];  Mi[11] = -(M[2]*M[3]+M[6]*M[7]+M[10]*M[11]);
	Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;    Mi[15] = 1;
/*
	Mi[0]  = Ri[0]; Mi[1]  = Ri[1]; Mi[2]  = Ri[2];  Mi[3]  = ti[0];
	Mi[4]  = Ri[3]; Mi[5]  = Ri[4]; Mi[6]  = Ri[5];  Mi[7]  = ti[1];
	Mi[8]  = Ri[6]; Mi[9]  = Ri[7]; Mi[10] = Ri[8];  Mi[11] = ti[2];
	Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;      Mi[15] = 1;
*/
}

/*
float dotProduct6(float *a, float *b) {
	float dot = 0;
	for (int i = 0; i < 6; i++) dot += a[i]*b[i];
	return dot;
}
*/

void transformVector(double *CM, double *v, int nRows, int nCols, double *w)
{
	for (int i = 0; i < nRows; i++) {
		w[i] = 0;
		for (int j = 0; j < nCols; j++) w[i] += CM[i*nCols+j]*v[j];
	}
}
void matrixMult(double *K, double *CM, double *P, int nRows, int nCols)
{
	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++) {
			P[i*nCols+j] = 0;
			for (int k = 0; k < nRows; k++)
				P[i*nCols+j] += K[k+i*nRows]*CM[j+k*nCols];
		}
}



void matrixMultMxN(double *A, double *B, double *R, int nRows1, int nCols1, int nCols2)
{
	for (int i = 0; i < nRows1; i++)
		for (int j = 0; j < nCols2; j++) {
			R[i*nCols2+j] = 0;
			for (int k = 0; k < nCols1; k++)
				R[i*nCols2+j] += A[k+i*nCols1]*B[j+k*nCols2];
		}
}

void matrixMultMxN_diagonal(double *A, double *B, double *R, int nRows1, int nCols1, int nCols2) {
	assert(nRows1 == nCols2);
	for (int i = 0; i < nRows1; i++) {
			int j = i;
			R[i] = 0;
			for (int k = 0; k < nCols1; k++)
				R[i] += A[k+i*nCols1]*B[j+k*nCols2];
		}
}

void matrixMultMxN_diagonal(float *A, float *B, float *R, int nRows1, int nCols1, int nCols2) {
	assert(nRows1 == nCols2);
	for (int i = 0; i < nRows1; i++) {
		int j = i;
		R[i] = 0;
		for (int k = 0; k < nCols1; k++)
			R[i] += A[k+i*nCols1]*B[j+k*nCols2];
	}
}

void projectionMatrix(double *K, double *cameraMatrix, double *P)
{
	double CM[12];
	identity3x4(CM);
	// invert camera matrix
	// R = R'
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 3; i++)
			CM[i+j*4] = double(cameraMatrix[j+i*4]);
	// t = -R'C
	CM[3] = -double(CM[0]*cameraMatrix[3]+CM[1]*cameraMatrix[7]+CM[2]*cameraMatrix[11]);
	CM[3+4] = -double(CM[4]*cameraMatrix[3]+CM[5]*cameraMatrix[7]+CM[6]*cameraMatrix[11]);
	CM[3+4*2] = -double(CM[8]*cameraMatrix[3]+CM[9]*cameraMatrix[7]+CM[10]*cameraMatrix[11]);

	// generate full projection matrix
	matrixMult(K,CM,P,3,4);
}


void capZero(float *A, int cnt) {
	for (int i = 0; i < cnt; i++) {
		if (A[i] < 0) A[i] = 0;
	}
}

void capZero(double *A, int cnt) {
	for (int i = 0; i < cnt; i++) {
		if (A[i] < 0) A[i] = 0;
	}
}

void normalize(double a[3])
{
	double len = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]+1e-8f);
	a[0] /= len;
	a[1] /= len;
	a[2] /= len;
}

void scaleVector3(double v[3],double scale)
{
	v[0] *= scale;
	v[1] *= scale;
	v[2] *= scale;
}

double dot3(double a[3], double b[3])
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

double lensq(double v[3])
{
	return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
}

void get3x3(double matrix4x3[3*4], double matrix3x3[3*3])
{
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			matrix3x3[c+r*3] = matrix4x3[c+r*4];
	return;
}

double det3x3(double mat[9] )
{
	double det;
    det = mat[0] * ( mat[4]*mat[8] - mat[7]*mat[5] )
        - mat[1] * ( mat[3]*mat[8] - mat[6]*mat[5] )
        + mat[2] * ( mat[3]*mat[7] - mat[6]*mat[4] );
    return det;
}

float det3x3(float mat[9] )
{
	float det;
    det = mat[0] * ( mat[4]*mat[8] - mat[7]*mat[5] )
        - mat[1] * ( mat[3]*mat[8] - mat[6]*mat[5] )
        + mat[2] * ( mat[3]*mat[7] - mat[6]*mat[4] );
    return det;
}

void normalize(float *vec) {
	float len = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]+1e-8f);
	vec[0] /= len;
	vec[1] /= len;
	vec[2] /= len; 
}


void identity3x3(float m[9])
{
	memset(m,0,sizeof(float)*9);
	m[0] = m[4] = m[8] = 1;
}


void identity3x3(double m[9])
{
	memset(m,0,sizeof(double)*9);
	m[0] = m[4] = m[8] = 1;
}

int inverse3x3( double ma[9], double mr[9])
{
	double det = det3x3(ma);
	if ( fabs( det ) < 1e-5f ) {
		identity3x3(mr);
		return 0;
	}

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
    
	return 1;
}

int inverse3x3( float ma[9], float mr[9])
{
	float det = det3x3(ma);
	if ( fabs( det ) < 1e-5f ) {
		identity3x3(mr);
		return 0;
	}

	float t[9];
	
	t[0] =  ( ma[4]*ma[8] - ma[5]*ma[7] ) / det;
	t[1] = -( ma[1]*ma[8] - ma[7]*ma[2] ) / det;
	t[2] =  ( ma[1]*ma[5] - ma[4]*ma[2] ) / det;
	t[3] = -( ma[3]*ma[8] - ma[5]*ma[6] ) / det;
	t[4] =  ( ma[0]*ma[8] - ma[6]*ma[2] ) / det;
	t[5] = -( ma[0]*ma[5] - ma[3]*ma[2] ) / det;
	t[6] =  ( ma[3]*ma[7] - ma[6]*ma[4] ) / det;
	t[7] = -( ma[0]*ma[7] - ma[6]*ma[1] ) / det;
	t[8] =  ( ma[0]*ma[4] - ma[1]*ma[3] ) / det;
	
	memcpy(mr,t,sizeof(float)*9);
	return 1;
}


void double_precision4(float *m4, double *m4_dbl) {
    for (int i = 0; i < 16; i++) m4_dbl[i] = double(m4[i]);
}

void float_precision4(double *m4, float *m4_flt) {
    for (int i = 0; i < 16; i++) m4_flt[i] = float(m4[i]);
}

void transpose(double ma[9], double mr[9])
{
	double tmp[9];
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			tmp[c+r*3] = ma[r+c*3];
	memcpy(mr,tmp,sizeof(double)*9);
}



void submat3x3( double mr[16], double mb[9], int i, int j ) {
  int di, dj, si, sj;
  // loop through 3x3 submatrix
  for( di = 0; di < 3; di ++ ) {
    for( dj = 0; dj < 3; dj ++ ) {
      // map 3x3 element (destination) to 4x4 element (source)
      si = di + ( ( di >= i ) ? 1 : 0 );
      sj = dj + ( ( dj >= j ) ? 1 : 0 );
      // copy element
      mb[di * 3 + dj] = mr[si * 4 + sj];
    }
  }
}

 
double det4x4( double mr[16] )
{
	double det,result = 0, i = 1;
    double msub3[9];
    int     n;
    for ( n = 0; n < 4; n++, i *= -1 )
    {
	    submat3x3( mr, msub3, 0, n );
        det     = det3x3( msub3 );
        result += mr[n] * det * i;
    }
    return( result );
}

int inverse4x4( double ma[16], double mr[16] )
{
	double mr_tmp[16];
	double  mdet = det4x4( ma );
	double mtemp[9];
	int     i, j, sign;
    if ( fabs( mdet ) < 0.0005 )
	{
		identity4x4( mr );
		return( 0 );
	}
    for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
		{
			sign = 1 - ( (i + j) % 2 ) * 2;
			submat3x3( ma, mtemp, i, j );
            mr_tmp[i+j*4] = ( det3x3( mtemp ) * sign ) / mdet;
        }
	memcpy(mr,mr_tmp,sizeof(double)*16);
    return( 1 );
}


void getTrans(double m[3*4], double t[3])
{
	t[0] = m[3]; t[1] = m[7]; t[2] = m[11];
}

void transformVector3(double m[9], double v[3])
{
	double tmp[3];

	for (int r = 0; r < 3; r++)
		tmp[r] = v[0]*m[0+r*3]+v[1]*m[1+r*3]+v[2]*m[2+r*3];
	memcpy(v,tmp,sizeof(double)*3);
}

void transformVector4(double m[12], double v[4])
{
	double tmp[4];

	for (int r = 0; r < 4; r++)
		tmp[r] = v[0]*m[0+r*3]+v[1]*m[1+r*3]+v[2]*m[2+r*3];//+v[3]*m[3+r*3];
	//tmp[r] = v[0]*m[0+r*3]+v[1]*m[1+r*3]+v[2]*m[2+r*3]+v[3]*m[3+r*3];
	memcpy(v,tmp,sizeof(double)*4);
}


void buildTx(double t[3],double Tx[9])
{
	Tx[0] = 0; Tx[1] = -t[2]; Tx[2] = t[1];
	Tx[3] = t[2]; Tx[4] = 0; Tx[5] = -t[0];
	Tx[6] = -t[1]; Tx[7] = t[0]; Tx[8] = 0;
	return;
}

void buildTx(float *t, float *Tx)
{
	Tx[0] = 0; Tx[1] = -t[2]; Tx[2] = t[1];
	Tx[3] = t[2]; Tx[4] = 0; Tx[5] = -t[0];
	Tx[6] = -t[1]; Tx[7] = t[0]; Tx[8] = 0;
	return;
}


void matrix_mult3(double m1[9], double m2[9], double res[9])
{
	double tmp[9];
	
	for (int r=0; r < 3; r++)
	for (int c=0; c < 3; c++)
	tmp[c+r*3] = m1[0+r*3]*m2[c+0*3]+
				 m1[1+r*3]*m2[c+1*3]+
				 m1[2+r*3]*m2[c+2*3];
	memcpy(res,tmp,sizeof(double)*9);
}

void matrixMult3(float *m1, float *m2, float *res) {
	float tmp[9];
	for (int r=0; r < 3; r++)
	for (int c=0; c < 3; c++)
	tmp[c+r*3] = m1[0+r*3]*m2[c+0*3]+
				 m1[1+r*3]*m2[c+1*3]+
				 m1[2+r*3]*m2[c+2*3];
	memcpy(res,tmp,sizeof(float)*9);
}

void matrixMult3(double *m1, double *m2, double *res) {
        double tmp[9];
        for (int r=0; r < 3; r++)
        for (int c=0; c < 3; c++)
        tmp[c+r*3] = m1[0+r*3]*m2[c+0*3]+
                                 m1[1+r*3]*m2[c+1*3]+
                                 m1[2+r*3]*m2[c+2*3];
        memcpy(res,tmp,sizeof(double)*9);
}

void printMatrix3(double m[9])
{
	for (int r=0; r < 3; r++)
		printf("%f %f %f\n",m[0+r*3],m[1+r*3],m[2+r*3]);
	
}

void rodrigues(double x, double y, double z, double m[9])
{
	double a = sqrt(x*x+y*y+z*z);
	x/=a; y/=a; z/=a;
	double ca = cos(a);
	double sa = sin(a);
	
	m[0] = ca+x*x*(1-ca);
	m[1] = x*y*(1-ca)-z*sa;
	m[2] = y*sa+x*z*ca;

	m[3] = z*sa+x*y*(1-ca);
	m[4] = ca+y*y*(1-ca);
	m[5] = -x*sa+y*z*(1-ca);
	
	m[6] = -y*sa+x*z*(1-ca);
	m[7] = x*sa+y*z*(1-ca);
	m[8] = ca+z*z*(1-ca);
}

void rodrigues(double x, double y, double z, double tx, double ty, double tz, double m[12])
{
	double a = sqrt(x*x+y*y+z*z);
	x/=a; y/=a; z/=a;
	double ca = cos(a);
	double sa = sin(a);
	
	m[0] = ca+x*x*(1-ca);
	m[1] = x*y*(1-ca)-z*sa;
	m[2] = y*sa+x*z*(1-ca);

	m[4] = z*sa+x*y*(1-ca);
	m[5] = ca+y*y*(1-ca);
	m[6] = -x*sa+y*z*(1-ca);
	
	m[8] = -y*sa+x*z*(1-ca);
	m[9] = x*sa+y*z*(1-ca);
	m[10] = ca+z*z*(1-ca);

	/*
	m[3]  = m[0]*tx+m[1]*ty+m[2]*tz;
	m[7]  = m[4]*tx+m[5]*ty+m[6]*tz;
	m[11] = m[8]*tx+m[9]*ty+m[10]*tz;
	*/
	m[3]  = tx;
	m[7]  = ty;
	m[11] = tz;

}


void matrix_from_euler(double xang, double yang, double zang, double m[9])
{
	double A = cos(xang);
	double B = sin(xang);
	double C = cos(yang);
	double D = sin(yang);
	double E = cos(zang);
	double F = sin(zang);
	double AD = A*D;
	double BD = B*D;

	m[0] =  C * E;
	m[1] = -C * F;
	m[2] =  D;
	m[3] =  BD*E + A*F;
	m[4] = -BD*F + A*E;
	m[5] = -B * C;
	m[6] = -AD*E + B*F;
	m[7] =  AD*F + B*E;
	m[8] = A * C;
}

void matrix_from_euler4(float xang, float yang, float zang, float *m)
{
    float A = cos(xang);
    float B = sin(xang);
    float C = cos(yang);
    float D = sin(yang);
    float E = cos(zang);
    float F = sin(zang);
    float AD = A*D;
    float BD = B*D;

    m[0] =  C * E;       m[1] = -C * F;       m[2] =  D;      m[3] = 0;
    m[4] =  BD*E + A*F;  m[5] = -BD*F + A*E;  m[6] = -B * C;  m[7] = 0;
    m[8] = -AD*E + B*F;  m[9] =  AD*F + B*E;  m[10] = A * C;  m[11] = 0;
    m[12] = 0;           m[13] =  0;          m[14] = 0;      m[15] = 1;
}

void pseudoInverse3x4(double *m, double *mr)
{
	double tm[12],tmp[9],itmp[9];
	
	int r,c;

	for (r=0; r < 3; r++)
		for (c=0; c < 4; c++)
			tm[r+c*3] = m[c+r*4];

	for (r=0; r < 3; r++)
		for (c=0; c < 3; c++)
			tmp[c+r*3] = m[0+r*4]*tm[c+0*3]+
						 m[1+r*4]*tm[c+1*3]+
						 m[2+r*4]*tm[c+2*3]+
						 m[3+r*4]*tm[c+3*3];
						 
	inverse3x3(tmp,itmp);
	
	for (r=0; r < 4; r++)
		for (c=0; c < 3; c++)
			mr[c+r*3] =  tm[0+r*3]*itmp[c+0*3]+
						 tm[1+r*3]*itmp[c+1*3]+
						 tm[2+r*3]*itmp[c+2*3];
}

float det2(float *v1, float *v2) {
	//   i  j  k
	// v1x v1y 0
	// v2x v2y 0
	return v1[0]*v2[1]-v1[1]*v2[0];
}

bool sameSide(float *p1, float *p2, float *a, float *b) {
	float ba[2];
	float p1a[2];
	float p2a[2];
	ba[0] = b[0]-a[0];
	ba[1] = b[1]-a[1];
	p1a[0] = p1[0]-a[0];
	p1a[1] = p1[1]-a[1];
	p2a[0] = p2[0]-a[0];
	p2a[1] = p2[1]-a[1];
	float v1 = det2(ba, p1a);
	float v2 = det2(ba, p2a);
	if (v1*v2 >= 0) return true;
	else return false;
}

//bool pointInTriangle(float px, float py, float ax, float ay, float bx, float by, float cx, float cy);

bool pointInTriangle(float px, float py, float ax, float ay, float bx, float by, float cx, float cy) {
	float p[2]; p[0] = px; p[1] = py;
	float a[2]; a[0] = ax; a[1] = ay;
	float b[2]; b[0] = bx; b[1] = by;
	float c[2]; c[0] = cx; c[1] = cy;
	if (sameSide(p,a, b,c) && sameSide(p,b, a,c) && sameSide(p,c, a,b)) return true;
	return false;
}

void quaternion2Rot(float *q, float *m) {
    float w = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];

    m[0] = w*w+x*x-y*y-z*z;
    m[1] = 2*(x*y-w*z);
    m[2] = 2*(x*z+w*y);
    m[3] = 0;

    m[4] = 2*(x*y+w*z);
    m[5] = w*w-x*x+y*y-z*z;
    m[6] = 2*(y*z-w*x);
    m[7] = 0;

    m[8] = 2*(x*z-w*y);
    m[9] = 2*(y*z+w*x);
    m[10] = w*w-x*x-y*y+z*z;
    m[11] = 0;

    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;
}

void quaternion2Rot(double *q, double *m) {
    double w = q[0];
    double x = q[1];
    double y = q[2];
    double z = q[3];

    m[0] = w*w+x*x-y*y-z*z;
    m[1] = 2*(x*y-w*z);
    m[2] = 2*(x*z+w*y);
    m[3] = 0;

    m[4] = 2*(x*y+w*z);
    m[5] = w*w-x*x+y*y-z*z;
    m[6] = 2*(y*z-w*x);
    m[7] = 0;

    m[8] = 2*(x*z-w*y);
    m[9] = 2*(y*z+w*x);
    m[10] = w*w-x*x-y*y+z*z;
    m[11] = 0;

    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;
}

void rot2Quaternion(float *m, int n, float *q)
{
    float tr = m[0]+m[n+1]+m[2*n+2];
    if (tr > 0) {
        float s = 0.5f / sqrtf(tr+1.0f);
        q[0] = 0.25f / s;
        q[1] = (m[2*n+1] - m[n+2])*s;
        q[2] = (m[0*n+2] - m[2*n+0])*s; //(t(0,2) - t(2,0))*s;
        q[3] = (m[1*n+0] - m[0*n+1])*s; //(t(1,0) - t(0,1))*s;
    } else {
        if (m[0] > m[n+1] && m[0] > m[2*n+2]) {
            float s = 2.0f * sqrtf(1.0f+m[0]-m[n+1]-m[2*n+2]);
            q[0] = (m[2*n+1]-m[1*n+2]) / s;
            q[1] = 0.25f * s;
            q[2] = (m[1] + m[n]) / s;
            q[3] = (m[2]+m[2*n]) / s;
        } else if (m[n+1] > m[2*n+2]) {
            float s = 2.0f * sqrtf(1.0f + m[n+1] - m[0] - m[2*n+2]);
            q[0] = (m[2]-m[2*n]) / s;
            q[1] = (m[1] + m[n]) / s;
            q[2] = 0.25f * s;
            q[3] = (m[n+2]+m[2*n+1]) / s;
        } else {
            float s = 2.0f * sqrtf(1.0f + m[2*n+2] - m[0] - m[n+1]);
            q[0] = (m[n]-m[1]) / s;
            q[1] = (m[2]+m[2*n]) / s;
            q[2] = (m[n+2]+m[2*n+1]) / s;
            q[3] = 0.25f * s;
        }
    }
}

void rot2Quaternion(double *m, int n, double *q)
{
    double tr = m[0]+m[n+1]+m[2*n+2];
    if (tr > 0) {
        double s = 0.5 / sqrtf(tr+1.0f);
        q[0] = 0.25 / s;
        q[1] = (m[2*n+1] - m[n+2])*s;
        q[2] = (m[0*n+2] - m[2*n+0])*s; //(t(0,2) - t(2,0))*s;
        q[3] = (m[1*n+0] - m[0*n+1])*s; //(t(1,0) - t(0,1))*s;
    } else {
        if (m[0] > m[n+1] && m[0] > m[2*n+2]) {
            double s = 2.0 * sqrt(1.0f+m[0]-m[n+1]-m[2*n+2]);
            q[0] = (m[2*n+1]-m[1*n+2]) / s;
            q[1] = 0.25 * s;
            q[2] = (m[1] + m[n]) / s;
            q[3] = (m[2]+m[2*n]) / s;
        } else if (m[n+1] > m[2*n+2]) {
            double s = 2.0 * sqrt(1.0f + m[n+1] - m[0] - m[2*n+2]);
            q[0] = (m[2]-m[2*n]) / s;
            q[1] = (m[1] + m[n]) / s;
            q[2] = 0.25 * s;
            q[3] = (m[n+2]+m[2*n+1]) / s;
        } else {
            double s = 2.0 * sqrt(1.0f + m[2*n+2] - m[0] - m[n+1]);
            q[0] = (m[n]-m[1]) / s;
            q[1] = (m[2]+m[2*n]) / s;
            q[2] = (m[n+2]+m[2*n+1]) / s;
            q[3] = 0.25 * s;
        }
    }
}
void normalizeQuaternion(float *qr ) {
    float len = qr[0]*qr[0]+qr[1]*qr[1]+qr[2]*qr[2]+qr[3]*qr[3]+1e-7f;
    len = sqrt(len);
    qr[0] /= len;
    qr[1] /= len;
    qr[2] /= len;
    qr[3] /= len;
}

void normalizeQuaternion(double *qr ) {
    float len = qr[0]*qr[0]+qr[1]*qr[1]+qr[2]*qr[2]+qr[3]*qr[3];
    len = sqrt(len);
    qr[0] /= len;
    qr[1] /= len;
    qr[2] /= len;
    qr[3] /= len;
}


float qdot(float *q1, float *q2) {
    return q1[0]*q2[0]+q1[1]*q2[1]+q1[2]*q2[2]+q1[3]*q2[3];
}

void lerp(float *p1, float w1, float *p2, float w2, int n, float *pt) {
    for (int i = 0; i < n; i++) {
        pt[i] = p1[i]*w1 + p2[i]*w2;
    }
}


void slerp(float *q1, float *q2, float t, float *qt) {
    float dot = qdot(q1,q2);
    float q3[4];
    //	dot = cos(theta)
    //        if (dot < 0), q1 and q2 are more than 90 degrees apart,
    //        so we can invert one to reduce spinning
    if (dot < 0) {
        dot = -dot;
        q3[0] = -q2[0]; q3[1] = -q2[1]; q3[2] = -q2[2]; q3[3] = -q2[3];
    } else {
        q3[0] = q2[0]; q3[1] = q2[1]; q3[2] = q2[2]; q3[3] = q2[3];
    }

    if (dot < 0.95f) {
            float angle = acosf(dot);
            float w1 = sinf(angle*(1-t))/sinf(angle);
            float w2 = sinf(angle*t)/sinf(angle);
            lerp(q1,w1,q3,w2,4,qt);
    } else { // if the angle is small, use linear interpolation
            return lerp(q1,(1-t),q3,t,4,qt);
    }
}

void rot2euler(double *m4, double *theta) {
    double r[9];
    get3x3(&m4[0],&r[0]);
    theta[0]  = atan2(r[3+2],r[6+2]);
    double c2 = sqrt(r[0]*r[0]+r[1]*r[1]);
    theta[1]  = atan2(-r[2],c2);
    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    theta[2] = atan2(s1*r[6]-c1*r[3],c1*r[4]-s1*r[7]);
}

void poseDistance(float *dTf, double *dist, double *angle) {
    // get inverted relativeT : current -> ref, in this form translation is origin difference
    //float idT[16];
    //invertRT4(dT,idT);
    double dT[16]; double_precision4(dTf,&dT[0]);

    double dx = dT[3];
    double dy = dT[7];
    double dz = dT[11];
    *dist = (float)sqrt(dx*dx+dy*dy+dz*dz+1e-12f);
/*
    // check identity matrix as a special case:
    if ((fabs(dT[0]-1.0f) < 1e-0) && (fabs(dT[5]-1.0f) < 1e-8) && (fabs(dT[10]-1.0f) < 1e-8)) {
        *angle = 0;
        return;
    }
*/
    double theta[3];
    rot2euler(dT,&theta[0]);
    theta[0] = fabs(theta[0]);
    theta[1] = fabs(theta[1]);
    theta[2] = fabs(theta[2]);
    double r = (theta[0] > theta[1]) ? theta[0] : theta[1];
    *angle = (theta[2] > r) ? theta[2] : r;
    *angle = rad2deg(*angle);
/*    if (theta[1])
    double trace = dT[0]+dT[5]+dT[10];
    double c = (trace-1)/2;
    if (c < -1.0) c = -1.0f;
    if (c >  1.0) c =  1.0f;
    *angle = rad2deg(acos(c));*/
}

float poseDistanceProjection(float *relT, float *K, float *testObj, int nVertices) {
    float error = 0.0f;
    for ( int i = 0; i < nVertices; i++) {
        float p1[3],p2[3],r1[3],r2[3];
        p1[0] = testObj[i*3+0];
        p1[1] = testObj[i*3+1];
        p1[2] = testObj[i*3+2];

        transformRT3(relT, p1, p2);

        p1[0] /= p1[2]; p1[1] /= p1[2]; p1[2] = 1;
        p2[0] /= p2[2]; p2[1] /= p2[2]; p2[2] = 1;

        matrixMultVec3(K,p1,r1);
        matrixMultVec3(K,p2,r2);

        float dX = r1[0]-r2[0];
        float dY = r1[1]-r2[1];
        error += dX*dX+dY*dY;
    }
    return error / float(nVertices);
}

int generateGridObject(float **gridObj,int nStepsX, int nStepsY, int nLayers, float size, float viewDistanceMin, float zRange, float fovAngleX) {
    float stepX = (2*size)/nStepsX;
    float stepY = (2*size)/nStepsY;
    float stepZ = 0.0f;//zRange/nLayers;
    assert(nLayers > 0);
    if (nLayers == 1) viewDistanceMin += zRange/2.0f;
    else stepZ = zRange/float(nLayers-1);

    float *obj = new float[nStepsX*nStepsY*nLayers*3];
    *gridObj = obj;

    int nPoints = 0;
    for (int k = 0; k < nLayers; k++) {

        float xyScale = 1.0f;
        if (k > 0) {
            float zFirst = -viewDistanceMin;
            float zCur = -viewDistanceMin - stepZ*k;
            xyScale = (zCur*tan(deg2rad(fovAngleX/2)))/(zFirst*tan(deg2rad(fovAngleX/2)));
        }
        for (int j = 0; j < nStepsY; j++) {
            for (int i = 0; i < nStepsX; i++) {
                obj[nPoints*3+0] = (-size + i*stepX)*xyScale;
                obj[nPoints*3+1] = (-size + j*stepY)*xyScale;
                obj[nPoints*3+2] = -viewDistanceMin - stepZ*k;
                nPoints++;
            }
        }
    }
    return nPoints;
}


void normalizeRT4(float *T) {
    float len;

    len = sqrtf(T[0]*T[0]+T[1]*T[1]+T[2]*T[2]+1e-6f);
    T[0] /= len;
    T[1] /= len;
    T[2] /= len;

    len = sqrtf(T[4]*T[4]+T[5]*T[5]+T[6]*T[6]+1e-6f);
    T[4] /= len;
    T[5] /= len;
    T[6] /= len;

    len = sqrtf(T[8]*T[8]+T[9]*T[9]+T[10]*T[10]+1e-6f);
    T[8] /= len;
    T[9] /= len;
    T[10] /= len;
}


double dotProduct6CPU(double *a, double *b) {
    double dot = 0;
    for (int i = 0; i < 6; i++) dot += a[i]*b[i];
    return dot;
}

void matrixMultVec6CPU(double *A, double *x, double *r)
{
    for (int i = 0; i < 6; i++) r[i] = (double)0.0;
    for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
            r[j] += A[j*6+k]*x[k];
        }
    }
}


#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }
float quickMedian(float *arr, int n)
{
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
        if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
        if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low+1]) ;

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]) ;
            do hh--; while (arr[hh]  > arr[low]) ;

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]) ;
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]) ;

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}


void get3DPoint(float x, float y, float z, float *iK, float *xc, float *yc, float *zc) {
    float pd[4],cd[4];
    pd[0] = x; pd[1] = y; pd[2] = 1;
    matrixMultVec3(iK,pd,cd);
    *xc = -cd[0]*z;
    *yc = -cd[1]*z;
    *zc = -z;
}

void get3DRay(float x, float y, float *iK, float *xr, float *yr, float *zr) {
    float pd[4],cd[4];
    pd[0] = x; pd[1] = y; pd[2] = 1;
    matrixMultVec3(iK,pd,cd);
    float len = sqrtf(cd[0]*cd[0]+cd[1]*cd[1]+cd[2]*cd[2]);
    cd[0] /= len; cd[1] /= len; cd[2] /= len;
    *xr = -cd[0];
    *yr = -cd[1];
    *zr = -cd[2];
}

// this initializes projection from current view k into reference view
void projectInitZ(float *srcT, float *dstK, float *dstT, float *P, float *Tz) {
        float Mi[16],K4x4[16],T[16];
        invertRT4(dstT,Mi);
        matrixMult4x4RT(Mi,srcT,T);
    //	[ k11 k12 k13 0]   [ t11 t12 t13 t14 ]   [X]
    //	[ k21 k22 k23 0] * [ t21 t22 t23 t24 ] * [Y] =
    //	[ k31 k32 k33 0]   [ t31 t32 t33 t34 ]   [Z]
    //	[   0   0   0 0]   [   0   0   0   1 ]   [1]
    memset(K4x4,0,sizeof(float)*16);
    memcpy(&K4x4[0],&dstK[0],sizeof(float)*3);
    memcpy(&K4x4[4],&dstK[3],sizeof(float)*3);
    memcpy(&K4x4[8],&dstK[6],sizeof(float)*3);
    matrixMult4x4(K4x4, T, P);
    Tz[0] = T[8]; Tz[1] = T[9]; Tz[2] = T[10]; Tz[3] = T[11];
}

// this initializes projection from current view k into reference view
void projectInit(float *dstK, float *T, float *P) {
    //	[ k11 k12 k13 0]   [ t11 t12 t13 t14 ]   [X]
    //	[ k21 k22 k23 0] * [ t21 t22 t23 t24 ] * [Y] =
    //	[ k31 k32 k33 0]   [ t31 t32 t33 t34 ]   [Z]
    //	[   0   0   0 0]   [   0   0   0   1 ]   [1]
    float K4x4[16];
    memset(K4x4,0,sizeof(float)*16);
    memcpy(&K4x4[0],&dstK[0],sizeof(float)*3);
    memcpy(&K4x4[4],&dstK[3],sizeof(float)*3);
    memcpy(&K4x4[8],&dstK[6],sizeof(float)*3);
    matrixMult4x4(K4x4, T, P);
}


void relativeTransform(float *srcT, float *dstT, float *T) {
    float Mi[16];
    invertRT4(dstT,Mi);
    matrixMult4x4RT(Mi,srcT,T);
}

// this initializes projection from current view k into reference view
void projectInitXYZ(float *srcT, float *dstK, float *dstT, float *P, float *Tx, float *Ty, float *Tz) {
    float Mi[16],K4x4[16],T[16];
    invertRT4(dstT,Mi);
    matrixMult4x4RT(Mi,srcT,T);
    //	[ k11 k12 k13 0]   [ t11 t12 t13 t14 ]   [X]
    //	[ k21 k22 k23 0] * [ t21 t22 t23 t24 ] * [Y] =
    //	[ k31 k32 k33 0]   [ t31 t32 t33 t34 ]   [Z]
    //	[   0   0   0 0]   [   0   0   0   1 ]   [1]
    memset(K4x4,0,sizeof(float)*16);
    memcpy(&K4x4[0],&dstK[0],sizeof(float)*3);
    memcpy(&K4x4[4],&dstK[3],sizeof(float)*3);
    memcpy(&K4x4[8],&dstK[6],sizeof(float)*3);
    matrixMult4x4(K4x4, T, P);
    Tx[0] = T[0]; Tx[1] = T[1]; Tx[2] = T[2];  Tx[3] = T[3];
    Ty[0] = T[4]; Ty[1] = T[5]; Ty[2] = T[6];  Ty[3] = T[7];
    Tz[0] = T[8]; Tz[1] = T[9]; Tz[2] = T[10]; Tz[3] = T[11];
}

void projectFast(float *x3d, float *x2d, float *P) {
    transformRT3(P,x3d,x2d);
    x2d[0] /= x2d[2];
    x2d[1] /= x2d[2];
}

void projectFastZ(float *x3d, float *x2d, float *z, float *P, float *Tz) {
    transformRT3(P,x3d,x2d);
    x2d[0] /= x2d[2];
    x2d[1] /= x2d[2];
    *z = x3d[0]*Tz[0]+x3d[1]*Tz[1]+x3d[2]*Tz[2]+Tz[3];
}

void projectFastXYZ(float *x3d, float *x2d, float *p3, float *P, float *Tx, float *Ty, float *Tz) {
    transformRT3(P,x3d,x2d);
    x2d[0] /= x2d[2];
    x2d[1] /= x2d[2];
    p3[0] = x3d[0]*Tx[0]+x3d[1]*Tx[1]+x3d[2]*Tx[2]+Tx[3];
    p3[1] = x3d[0]*Ty[0]+x3d[1]*Ty[1]+x3d[2]*Ty[2]+Ty[3];
    p3[2] = x3d[0]*Tz[0]+x3d[1]*Tz[1]+x3d[2]*Tz[2]+Tz[3];
}


void distortPointCPU(float *pu, float *kc, float *K, float *pd) {
    // distort point
    float r2,r4,r6;
    float radialDist;
    float dx;
    float dy;

    // generate r2 components
    dx  = pu[0]*pu[0]; dy  = pu[1]*pu[1];
    // generate distorted coordinates
    r2 = dx+dy; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pd[0] = K[0]*pu[0]*radialDist+K[2];
    pd[1] = K[4]*pu[1]*radialDist+K[5];
}


/*
 *  Utility functions to find cubic and quartic roots,
 *  coefficients are passed like this:
 *
 *      c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4 = 0
 *
 *  The functions return the number of non-complex roots and
 *  put the values into the s array.
 *
 *  Author:         Jochen Schwarze (schwarze@isa.de)
 */
/*
extern double   sqrt(), cbrt(), cos(), acos();

// epsilon surrounding for near zero values 

#define     EQN_EPS     1e-9
#define	    IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)

#ifdef NOCBRT
#define     cbrt(x)     ((x) > 0.0 ? pow((double)(x), 1.0/3.0) : \
                          ((x) < 0.0 ? -pow((double)-(x), 1.0/3.0) : 0.0))
#endif

int solveQuadric(double *c, double *s)
{
    double p, q, D;

    // normal form: x^2 + px + q = 0 

    p = c[ 1 ] / (2 * c[ 2 ]);
    q = c[ 0 ] / c[ 2 ];

    D = p * p - q;

    if (IsZero(D))
    {
    s[ 0 ] = - p;
    return 1;
    }
    else if (D < 0)
    {
    return 0;
    }
    else if (D > 0)
    {
    double sqrt_D = sqrt(D);

    s[ 0 ] =   sqrt_D - p;
    s[ 1 ] = - sqrt_D - p;
    return 2;
    }
}
*/
/*
int solveCubic(double *c, double *s)
{
    int     i, num;
    double  sub;
    double  A, B, C;
    double  sq_A, p, q;
    double  cb_p, D;

    // normal form: x^3 + Ax^2 + Bx + C = 0 

    A = c[ 2 ] / c[ 3 ];
    B = c[ 1 ] / c[ 3 ];
    C = c[ 0 ] / c[ 3 ];

    //  substitute x = y - A/3 to eliminate quadric term: x^3 +px + q = 0 

    sq_A = A * A;
    p = 1.0/3 * (- 1.0/3 * sq_A + B);
    q = 1.0/2 * (2.0/27 * A * sq_A - 1.0/3 * A * B + C);

    // use Cardano's formula 

    cb_p = p * p * p;
    D = q * q + cb_p;

    if (IsZero(D))
    {
    if (IsZero(q)) // one triple solution 
    {
        s[ 0 ] = 0;
        num = 1;
    }
    else // one single and one double solution 
    {
        double u = cbrt(-q);
        s[ 0 ] = 2 * u;
        s[ 1 ] = - u;
        num = 2;
    }
    }
    else if (D < 0) // Casus irreducibilis: three real solutions 
    {
    double phi = 1.0/3 * acos(-q / sqrt(-cb_p));
    double t = 2 * sqrt(-p);

    s[ 0 ] =   t * cos(phi);
    s[ 1 ] = - t * cos(phi + M_PI / 3);
    s[ 2 ] = - t * cos(phi - M_PI / 3);
    num = 3;
    }
    else // one real solution 
    {
    double sqrt_D = sqrt(D);
    double u = cbrt(sqrt_D - q);
    double v = - cbrt(sqrt_D + q);

    s[ 0 ] = u + v;
    num = 1;
    }

    // resubstitute 

    sub = 1.0/3 * A;

    for (i = 0; i < num; ++i)
    s[ i ] -= sub;

    return num;
}

int solveQuartic(double *c, double *s)
{
    double  coeffs[ 4 ];
    double  z, u, v, sub;
    double  A, B, C, D;
    double  sq_A, p, q, r;
    int     i, num;

    // normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 

    A = c[ 3 ] / c[ 4 ];
    B = c[ 2 ] / c[ 4 ];
    C = c[ 1 ] / c[ 4 ];
    D = c[ 0 ] / c[ 4 ];

    //  substitute x = y - A/4 to eliminate cubic term x^4 + px^2 + qx + r = 0 

    sq_A = A * A;
    p = - 3.0/8 * sq_A + B;
    q = 1.0/8 * sq_A * A - 1.0/2 * A * B + C;
    r = - 3.0/256*sq_A*sq_A + 1.0/16*sq_A*B - 1.0/4*A*C + D;

    if (IsZero(r))
    {
    // no absolute term: y(y^3 + py + q) = 0 

    coeffs[ 0 ] = q;
    coeffs[ 1 ] = p;
    coeffs[ 2 ] = 0;
    coeffs[ 3 ] = 1;

    num = solveCubic(coeffs, s);

    s[ num++ ] = 0;
    }
    else
    {
    // solve the resolvent cubic ... 

    coeffs[ 0 ] = 1.0/2 * r * p - 1.0/8 * q * q;
    coeffs[ 1 ] = - r;
    coeffs[ 2 ] = - 1.0/2 * p;
    coeffs[ 3 ] = 1;

    solveCubic(coeffs, s);

    // ... and take the one real solution ... 
    z = s[ 0 ];

    // ... to build two quadric equations 
    u = z * z - r;
    v = 2 * z - p;

    if (IsZero(u))
        u = 0;
    else if (u > 0)
        u = sqrt(u);
    else
        return 0;

    if (IsZero(v))
        v = 0;
    else if (v > 0)
        v = sqrt(v);
    else
        return 0;

    coeffs[ 0 ] = z - u;
    coeffs[ 1 ] = q < 0 ? -v : v;
    coeffs[ 2 ] = 1;

    num = solveQuadric(coeffs, s);

    coeffs[ 0 ]= z + u;
    coeffs[ 1 ] = q < 0 ? v : -v;
    coeffs[ 2 ] = 1;

    num += solveQuadric(coeffs, s + num);
    }

    // resubstitute 

    sub = 1.0/4 * A;

    for (i = 0; i < num; ++i)
    s[ i ] -= sub;

    return num;
}
*/
int analyzeConvergence(float *JtJDev, float *bb, double tol, int maxSteps)
{
    double A[6*6];
    double x[6],xp[6];
    double r[6];
    double b[6];
    double dir[6];
  //  double diffMag=DBL_MAX,pDiffMag=DBL_MAX;
    // copy matrix to local memory for speeding up access
    for (int i = 0; i < 36; i++) A[i] = (double)JtJDev[i];
    //for (int i = 0; i < 6; i++) A[i+i*6] += (doublereal)1e-8;
    for (int i = 0; i < 6; i++) { b[i] = (double)bb[i]; xp[i] = x[i] = 0.0; r[i] = b[i]; dir[i] = b[i]; }
    int nSteps = 0;
//    double firstError = dotProduct6CPU(r,r);
//    double lastError = 0.0;
    int invalidIter = 0;
    while (nSteps < maxSteps) {
        double Adir[6];
        matrixMultVec6CPU(A,dir,Adir);
        //step length
        double rr = dotProduct6CPU(r,r);
        double Adirdir = dotProduct6CPU(Adir,dir);
        // compute abs(Adirdir), its numerically more stable than |Adirdir|:
        double div = Adirdir; if (div < 0) div = -div;
        double stepLength = 0.0;
        // prevent division by zero:
        if (div > tol) stepLength = rr/Adirdir;
      //  else invalidIter++;
        // update error:
        for (int i = 0; i < 6; i++) { r[i] -= stepLength*Adir[i]; }
        double rr2 = dotProduct6CPU(r,r);
  //      lastError = rr2;
        // early exit with previous x, (minimization step failed!)
/*        if (rr2 > rr) {
            return maxSteps;
        }*/
        // update params:
        for (int i = 0; i < 6; i++) { xp[i] = x[i]; x[i] += stepLength*dir[i];}
        // early exit, residual is below a threshold:
        if (rr2 < tol*tol) {
            return invalidIter;
        }
/*
        // measure step size:
        double diff[6];
        for (int i = 0; i < 6; i++) { diff[i] = xp[i] - x[i]; }
        pDiffMag = diffMag;
        diffMag = dotProduct6CPU(diff,diff);
        if (diffMag > pDiffMag*1.1) {
            return maxSteps;
        }
*/
        double beta = rr2/rr;
        if (beta > 1.0) invalidIter++;
        for (int i = 0; i < 6; i++) { dir[i] = r[i] + beta*dir[i]; }
        nSteps++;
    }
    return invalidIter;
}

