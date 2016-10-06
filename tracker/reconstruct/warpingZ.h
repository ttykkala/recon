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

#define __WARPINGZ_H__

class WarpingZ
{
private:
	double Mrel[16],M[16],baseM[16];
	float  Ko[9],K[9],iK[9];
	double invMarker[16],invTmp[16];
	double MrelMarker[16];
	void   mulRelativeTransform( double *T );
public:
	WarpingZ(int resoX, double *K);
	~WarpingZ();
	void warp(float *inPoints3d, int nPoints, float *outPoints2d, float *outPoints3d, float *dZ1, float *dZ2, float *dZ3,float *dZ4, float *dZ5, float *dZ6);
	void warp(float *inPoints3d, int nPoints, float *outPoints2d, float *outPoints3d);	
	void warpInverse3( float *inPoints3d, int nPoints, float *outPoints3d);
	void warpDiff(float *refPoints, int nPoints, float *points2d_dx1,float *points2d_dx2,float *points2d_dx3,float *points2d_dx4,float *points2d_dx5,float *points2d_dx6,float *points2d_dxz);
	void warpDiffCurrent(float *refPoints, int nPoints, float *points2d_dx1,float *points2d_dx2,float *points2d_dx3,float *points2d_dx4,float *points2d_dx5,float *points2d_dx6);	
	float *getNormalizedK();
	float *getK();
	void  resetBase();
	double *getTransform();
	void getInverseTransform(double *Mi);
	float getZAngle();
	double *getRelativeTransformToMarker();
	void   setRelativeTransform( double *M );
	void   setRelativeTransform( float *M );
	void  setMarker();
	void  setTransformSE3(double *x);
	void  setTransformSE3(float *x);
	void  mulTransformSE3(double *x);
	void mulTransformSE3( float *x );
	void  setImageResolution(int resoX);
};
