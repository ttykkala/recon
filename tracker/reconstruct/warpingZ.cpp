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
#include "warpingZ.h"
#include <string.h>
#include <expm.h>
#include <stdio.h>
#include "basic_math.h"
// CUDA
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <timer/performanceCounter.h>

WarpingZ::WarpingZ( int resoX, double *Kd)
{
	// setup mappings
	identity4x4(this->Mrel);
	identity4x4(this->baseM);
	identity4x4(this->M);
	for (int i = 0; i < 9; i++) Ko[i] = float(Kd[i]);
	setImageResolution(resoX);
}

void WarpingZ::setRelativeTransform( double *T )
{
	memcpy(this->Mrel,T,sizeof(double)*16);
	matrixMult4x4(this->baseM,this->Mrel,this->M);
}

void WarpingZ::setRelativeTransform( float *T )
{
	double m[16]; for (int i = 0; i < 16; i++) m[i] = T[i];
	setRelativeTransform(m);
}


void WarpingZ::mulRelativeTransform( double *T )
{
	matrixMult4x4(this->Mrel,T,this->Mrel);
	matrixMult4x4(this->baseM,this->Mrel,this->M);
}

void WarpingZ::setTransformSE3( float *x) {
	double xd[6];
	for (int i = 0; i < 6; i++) xd[i] = x[i];
	setTransformSE3(xd);
}

void WarpingZ::setTransformSE3( double *x )
{
	//printf("x: %f %f %f %f %f %f\n",x[0],x[1],x[2],x[3],x[4],x[5]);
	double A[16];
	A[0] = 0;		  A[1] = -x[2];   A[2] = x[1];	A[3]  = x[3];
	A[4] = x[2];	  A[5] = 0;		  A[6] =-x[0];	A[7]  = x[4];
	A[8] =-x[1];	  A[9] = x[0];	  A[10] = 0;	A[11] = x[5];
	A[12] = 0;		  A[13] = 0;	  A[14] = 0;	A[15] = 0;
	double Md[16];
	expm(4,A,1.0,Md,6);
	setRelativeTransform(Md);
}

void WarpingZ::mulTransformSE3( float *x ) {
	double xd[6];
	for (int i = 0; i < 6; i++) xd[i] = x[i];
	mulTransformSE3(xd);
}

void WarpingZ::mulTransformSE3( double *x )
{
	double A[16];
	A[0]  =    0;	A[1] = -x[2];   A[2]  = x[1]; A[3]  = x[3];
	A[4]  = x[2];	A[5] =     0;   A[6]  =-x[0]; A[7]  = x[4];
	A[8]  =-x[1];	A[9] =  x[0];	A[10] =    0; A[11] = x[5];
	A[12] =    0;	A[13] =    0;	A[14] =    0; A[15] =    0;

	double Md[16];
	expm(4,A,1.0,Md,6);
	mulRelativeTransform(Md);
}

void WarpingZ::warp( float *inPoints3d, int nPoints, float *outPoints2d, float *outPoints3d)
{
	float R[9],t[3];
	double Mi[16]; 
	invertRT4( this->M, Mi); copy3x3f(Mi,R); t[0] = Mi[3]; t[1] = Mi[7]; t[2] = Mi[11];

	float p3d[3],p2d[3];
	for (int i = 0; i < nPoints; i++) {
		matrixMultVec3(R, &inPoints3d[i*3], p3d); p3d[0] += t[0]; p3d[1] += t[1]; p3d[2] += t[2];
		outPoints3d[i*3+0] = p3d[0];
		outPoints3d[i*3+1] = p3d[1];
		outPoints3d[i*3+2] = p3d[2];
		matrixMultVec3(K,p3d,p2d); p2d[0] /= p2d[2]; p2d[1] /= p2d[2]; p2d[2] = 1;
		outPoints2d[i*2+0] = p2d[0];
		outPoints2d[i*2+1] = p2d[1];
	} 
}


void WarpingZ::warpInverse3( float *inPoints3d, int nPoints, float *outPoints3d) {
	float R[9],t[3];
	copy3x3f(M,R); t[0] = M[3]; t[1] = M[7]; t[2] = M[11];

	float p3d[3],p2d[3];
	for (int i = 0; i < nPoints; i++) {
		matrixMultVec3(R, &inPoints3d[i*3], p3d); p3d[0] += t[0]; p3d[1] += t[1]; p3d[2] += t[2];
		outPoints3d[i*3+0] = p3d[0];
		outPoints3d[i*3+1] = p3d[1];
		outPoints3d[i*3+2] = p3d[2];
	} 	
}

void WarpingZ::warp( float *inPoints3d, int nPoints, float *outPoints2d, float *outPoints3d, float *dZ1, float *dZ2, float *dZ3,float *dZ4, float *dZ5, float *dZ6)
{
	float R[9],t[3];
	double Mi[16]; 
	invertRT4( this->M, Mi); copy3x3f(Mi,R); t[0] = Mi[3]; t[1] = Mi[7]; t[2] = Mi[11];

	float p3d[3],p2d[3];
	for (int i = 0; i < nPoints; i++) {
		matrixMultVec3(R, &inPoints3d[i*3], p3d); p3d[0] += t[0]; p3d[1] += t[1]; p3d[2] += t[2];
		outPoints3d[i*3+0] = p3d[0];
		outPoints3d[i*3+1] = p3d[1];
		outPoints3d[i*3+2] = p3d[2];
		matrixMultVec3(K,p3d,p2d); p2d[0] /= p2d[2]; p2d[1] /= p2d[2]; p2d[2] = 1;
		outPoints2d[i*2+0] = p2d[0];
		outPoints2d[i*2+1] = p2d[1];
		dZ1[i] = -p3d[1];
		dZ2[i] =  p3d[0];
		dZ3[i] =       0;
		dZ4[i] =       0;
		dZ5[i] =       0;
		dZ6[i] =      -1;
	} 
	/*
	A[8]  A[9]  A[10]  A[11]

	memset(dA1,0,sizeof(float)*16); dA1[6]  =  1; dA1[9] = -1;
	memset(dA2,0,sizeof(float)*16); dA2[2]  = -1; dA2[8] =  1;
	memset(dA3,0,sizeof(float)*16); dA3[1]  =  1; dA3[4] = -1;
	memset(dA4,0,sizeof(float)*16); dA4[3]  = -1;
	memset(dA5,0,sizeof(float)*16); dA5[7]  = -1;
	memset(dA6,0,sizeof(float)*16); dA6[11] = -1;*/
}

float *WarpingZ::getK()
{
	return &K[0];
}

WarpingZ::~WarpingZ()
{
}


float *WarpingZ::getNormalizedK() {
	return &Ko[0];
}

void WarpingZ::setImageResolution( int resoX)
{
	for (int i = 0; i < 3; i++) {
		K[i] = Ko[i]*float(resoX);
		K[i+3] = Ko[i+3]*float(resoX);
	}
	K[6] = 0; K[7] = 0;  K[8] = 1;
	inverse3x3(this->K,this->iK);
}

void WarpingZ::resetBase() {
	identity4x4(this->Mrel);
	setMarker();
}

void WarpingZ::setMarker()
{
	invertRT4(Mrel,invMarker);
}

double *WarpingZ::getRelativeTransformToMarker()
{
	matrixMult4x4(invMarker,Mrel,MrelMarker);
	return (double*)MrelMarker;
}

double *WarpingZ::getTransform()
{
	return &M[0];
}

void WarpingZ::getInverseTransform(double *Mi)
{
	invertRT4( this->M, Mi);
}

void eulerAngles4(double *m, float *pitch, float *yaw, float *roll) {
	if (fabs(m[8]) != 1.0f) {
		float yaw1 = -asin(m[8]);
		float yaw2 = 3.141592653f - yaw1;
		float ca1 = cos(yaw1);
		float ca2 = cos(yaw2);
		float roll1 = atan2(m[4]/ca1,m[0]/ca1);
		float roll2 = atan2(m[4]/ca2,m[0]/ca2);
		if (fabs(roll1) < fabs(roll2)) { 
			*yaw = yaw1;
			*pitch = atan2(m[9]/ca1,m[10]/ca1); 
			*roll = roll1;
		} else {
			*roll = roll2;
			*pitch = atan2(m[9]/ca2,m[10]/ca2);
			*yaw = yaw2;
		}
	} else {
		*roll = 0;
		if (m[8] < 0) {
			*yaw = 3.141592653f/2.0f;
			*pitch = *roll + atan2(m[1],m[2]);
		} else {
			*yaw = -3.141592653f/2.0f;
			*pitch = -(*roll) + atan2(-m[1],-m[2]);
		}
	}
}

float WarpingZ::getZAngle()
{
	//double Mi[16];
	//invertRT4( this->M, Mi);
	float pitch,yaw,roll;
	eulerAngles4(this->M,&pitch,&yaw,&roll);
	return roll;
}


void WarpingZ::warpDiff(float *refPoints3D, int nPoints, float *points2d_dx1,float *points2d_dx2,float *points2d_dx3,float *points2d_dx4,float *points2d_dx5,float *points2d_dx6,float *points2d_dxz)
{
	/*
	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
	*/
//	double Mi[16]; 
	//invertRT4( this->M, Mi);
	//for (int i=0; i < 16; i++) printf("m[%d]:%f\n",i,Mi[i]);

	float dA1[16],dA2[16],dA3[16],dA4[16],dA5[16],dA6[16];
	// x is inverted here! 
	// dexp(A(x))/dx = A(x) around zero
	// exp(A(x))*exp(-A(x)) = I, thus exp(-A(x)) is the inverse
	// A(x) is linear function -> -A(x) = A(-x)
	memset(dA1,0,sizeof(float)*16); dA1[6]  =  1; dA1[9] = -1;
	memset(dA2,0,sizeof(float)*16); dA2[2]  = -1; dA2[8] =  1;
	memset(dA3,0,sizeof(float)*16); dA3[1]  =  1; dA3[4] = -1;
	memset(dA4,0,sizeof(float)*16); dA4[3]  = -1;
	memset(dA5,0,sizeof(float)*16); dA5[7]  = -1;
	memset(dA6,0,sizeof(float)*16); dA6[11] = -1;
	// generate inv([dR3(0),dt3(0)]) for each param x
	float dR1[9],dR2[9],dR3[9],dR4[9],dR5[9],dR6[9];
	float dC1[3],dC2[9],dC3[3],dC4[3],dC5[3],dC6[3];
	copy3x3(dA1,dR1); dC1[0] = dA1[3]; dC1[1] = dA1[7]; dC1[2] = dA1[11];
	copy3x3(dA2,dR2); dC2[0] = dA2[3]; dC2[1] = dA2[7]; dC2[2] = dA2[11];
	copy3x3(dA3,dR3); dC3[0] = dA3[3]; dC3[1] = dA3[7]; dC3[2] = dA3[11];
	copy3x3(dA4,dR4); dC4[0] = dA4[3]; dC4[1] = dA4[7]; dC4[2] = dA4[11];
	copy3x3(dA5,dR5); dC5[0] = dA5[3]; dC5[1] = dA5[7]; dC5[2] = dA5[11];
	copy3x3(dA6,dR6); dC6[0] = dA6[3]; dC6[1] = dA6[7]; dC6[2] = dA6[11];
	
	float p3d[3],n3d[3],p2d[3];
	float N[9];
	for (int i = 0; i < nPoints; i++) {
		matrixMultVec3(K, &refPoints3D[i*3], p3d); 
		N[0] = 1.0f/p3d[2]; N[1] =       0; N[2] = -p3d[0]/(p3d[2]*p3d[2]);
		N[3] =       0; N[4] = 1.0f/p3d[2]; N[5] = -p3d[1]/(p3d[2]*p3d[2]);
		N[6] =       0; N[7] =       0; N[8] =           1;

		matrixMultVec3(dR1, &refPoints3D[i*3], p3d); 
		p3d[0] += dC1[0]; 
		p3d[1] += dC1[1]; 
		p3d[2] += dC1[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx1[i*2+0] = p2d[0]; 
		points2d_dx1[i*2+1] = p2d[1];

		matrixMultVec3(dR2, &refPoints3D[i*3], p3d); 
		p3d[0] += dC2[0]; 
		p3d[1] += dC2[1]; 
		p3d[2] += dC2[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx2[i*2+0] = p2d[0]; 
		points2d_dx2[i*2+1] = p2d[1];

		matrixMultVec3(dR3, &refPoints3D[i*3], p3d); 
		p3d[0] += dC3[0]; 
		p3d[1] += dC3[1]; 
		p3d[2] += dC3[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx3[i*2+0] = p2d[0]; 
		points2d_dx3[i*2+1] = p2d[1];

		matrixMultVec3(dR4, &refPoints3D[i*3], p3d); 
		p3d[0] += dC4[0]; 
		p3d[1] += dC4[1]; 
		p3d[2] += dC4[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx4[i*2+0] = p2d[0]; 
		points2d_dx4[i*2+1] = p2d[1];

		matrixMultVec3(dR5, &refPoints3D[i*3], p3d); 
		p3d[0] += dC5[0]; 
		p3d[1] += dC5[1]; 
		p3d[2] += dC5[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx5[i*2+0] = p2d[0]; 
		points2d_dx5[i*2+1] = p2d[1];

		matrixMultVec3(dR6, &refPoints3D[i*3], p3d); 
		p3d[0] += dC6[0]; 
		p3d[1] += dC6[1]; 
		p3d[2] += dC6[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx6[i*2+0] = p2d[0]; 
		points2d_dx6[i*2+1] = p2d[1];
	}

}

void WarpingZ::warpDiffCurrent(float *refPoints3D, int nPoints, float *points2d_dx1,float *points2d_dx2,float *points2d_dx3,float *points2d_dx4,float *points2d_dx5,float *points2d_dx6)
{
	float R[9],t[3];
	double Mi[16]; float Mif[16];
	invertRT4( this->M, Mi); for (int i = 0; i < 16; i++) Mif[i] = Mi[i];
	copy3x3f(Mi,R); t[0] = Mif[3]; t[1] = Mif[7]; t[2] = Mif[11];

	/*
	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
	*/
//	double Mi[16]; 
	//invertRT4( this->M, Mi);
	//for (int i=0; i < 16; i++) printf("m[%d]:%f\n",i,Mi[i]);

	float dA1[16],dA2[16],dA3[16],dA4[16],dA5[16],dA6[16];
	// x is inverted here! 
	// dexp(A(x))/dx = A(x) 
	// exp(A(x))*exp(-A(x)) = I, thus exp(-A(x)) is the inverse
	// A(x) is linear function -> -A(x) = A(-x)
	memset(dA1,0,sizeof(float)*16); dA1[6]  =  1; dA1[9] = -1;
	memset(dA2,0,sizeof(float)*16); dA2[2]  = -1; dA2[8] =  1;
	memset(dA3,0,sizeof(float)*16); dA3[1]  =  1; dA3[4] = -1;
	memset(dA4,0,sizeof(float)*16); dA4[3]  = -1;
	memset(dA5,0,sizeof(float)*16); dA5[7]  = -1;
	memset(dA6,0,sizeof(float)*16); dA6[11] = -1;
	// concatenate base transform inverse
	matrixMult4x4(dA1,Mif,dA1);
	matrixMult4x4(dA2,Mif,dA2);
	matrixMult4x4(dA3,Mif,dA3);
	matrixMult4x4(dA4,Mif,dA4);
	matrixMult4x4(dA5,Mif,dA5);
	matrixMult4x4(dA6,Mif,dA6);

	// generate inv([dR3(0),dt3(0)]) for each param x
	float dR1[9],dR2[9],dR3[9],dR4[9],dR5[9],dR6[9];
	float dt1[3],dt2[9],dt3[3],dt4[3],dt5[3],dt6[3];
	copy3x3(dA1,dR1); dt1[0] = dA1[3]; dt1[1] = dA1[7]; dt1[2] = dA1[11];
	copy3x3(dA2,dR2); dt2[0] = dA2[3]; dt2[1] = dA2[7]; dt2[2] = dA2[11];
	copy3x3(dA3,dR3); dt3[0] = dA3[3]; dt3[1] = dA3[7]; dt3[2] = dA3[11];
	copy3x3(dA4,dR4); dt4[0] = dA4[3]; dt4[1] = dA4[7]; dt4[2] = dA4[11];
	copy3x3(dA5,dR5); dt5[0] = dA5[3]; dt5[1] = dA5[7]; dt5[2] = dA5[11];
	copy3x3(dA6,dR6); dt6[0] = dA6[3]; dt6[1] = dA6[7]; dt6[2] = dA6[11];
	
	float p3d[3],n3d[3],p2d[3];
	float N[9];
	for (int i = 0; i < nPoints; i++) {
		matrixMultVec3(R, &refPoints3D[i*3], p3d); p3d[0] += t[0]; p3d[1] += t[1]; p3d[2] += t[2];
		matrixMultVec3(K, p3d, n3d); 
		N[0] = 1.0f/n3d[2]; N[1] =       0;     N[2] = -n3d[0]/(n3d[2]*n3d[2]);
		N[3] =       0;     N[4] = 1.0f/n3d[2]; N[5] = -n3d[1]/(n3d[2]*n3d[2]);
		N[6] =       0;     N[7] =       0;     N[8] =           1;

		matrixMultVec3(dR1, &refPoints3D[i*3], p3d); 
		p3d[0] += dt1[0]; 
		p3d[1] += dt1[1]; 
		p3d[2] += dt1[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx1[i*2+0] = p2d[0]; 
		points2d_dx1[i*2+1] = p2d[1];

		matrixMultVec3(dR2, &refPoints3D[i*3], p3d); 
		p3d[0] += dt2[0]; 
		p3d[1] += dt2[1]; 
		p3d[2] += dt2[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx2[i*2+0] = p2d[0]; 
		points2d_dx2[i*2+1] = p2d[1];

		matrixMultVec3(dR3, &refPoints3D[i*3], p3d); 
		p3d[0] += dt3[0]; 
		p3d[1] += dt3[1]; 
		p3d[2] += dt3[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx3[i*2+0] = p2d[0]; 
		points2d_dx3[i*2+1] = p2d[1];

		matrixMultVec3(dR4, &refPoints3D[i*3], p3d); 
		p3d[0] += dt4[0]; 
		p3d[1] += dt4[1]; 
		p3d[2] += dt4[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx4[i*2+0] = p2d[0]; 
		points2d_dx4[i*2+1] = p2d[1];

		matrixMultVec3(dR5, &refPoints3D[i*3], p3d); 
		p3d[0] += dt5[0]; 
		p3d[1] += dt5[1]; 
		p3d[2] += dt5[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx5[i*2+0] = p2d[0]; 
		points2d_dx5[i*2+1] = p2d[1];

		matrixMultVec3(dR6, &refPoints3D[i*3], p3d); 
		p3d[0] += dt6[0]; 
		p3d[1] += dt6[1]; 
		p3d[2] += dt6[2];
		matrixMultVec3(K, p3d, n3d);
		matrixMultVec3(N, n3d, p2d); 
		points2d_dx6[i*2+0] = p2d[0]; 
		points2d_dx6[i*2+1] = p2d[1];
	}
}

