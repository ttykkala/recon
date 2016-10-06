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
#include "PointCloud3D.h"
// CUDA
#include <cuda_runtime.h>
#include <cutil_inline.h>


PointCloud3D::PointCloud3D( int numMaxPoints )
{
	nSamplePoints = 0;
	nSupportPoints = 0;
	nMaxPoints = numMaxPoints;
	refPoints3d = new float[numMaxPoints*3]; refSupport3d = new float[numMaxPoints*3];
	refPoints2d = new float[numMaxPoints*2]; refSupport2d = new float[numMaxPoints*2];
	tmpPoints3d = new float[numMaxPoints*3];
	curPoints3d = new float[numMaxPoints*3]; 
	curPoints2d = new float[numMaxPoints*2]; 
	offsets = new unsigned int[numMaxPoints];
	colors = new unsigned char[numMaxPoints];
	supportOffsets = new unsigned int[numMaxPoints];
	supportColors = new unsigned char[numMaxPoints];

	curPointsdZ1 = new float[numMaxPoints];
	curPointsdZ2 = new float[numMaxPoints];
	curPointsdZ3 = new float[numMaxPoints];
	curPointsdZ4 = new float[numMaxPoints];
	curPointsdZ5 = new float[numMaxPoints];
	curPointsdZ6 = new float[numMaxPoints];

	points2d_dx1 = new float[numMaxPoints*2]; 
	points2d_dx2 = new float[numMaxPoints*2]; 
	points2d_dx3 = new float[numMaxPoints*2]; 
	points2d_dx4 = new float[numMaxPoints*2]; 
	points2d_dx5 = new float[numMaxPoints*2]; 
	points2d_dx6 = new float[numMaxPoints*2]; 
	points2d_dxz = new float[numMaxPoints*2]; 
	differentialsComputed = false;
}

PointCloud3D::~PointCloud3D()
{
	delete[] refPoints2d;
	delete[] refSupport2d;
	delete[] refPoints3d;
	delete[] tmpPoints3d;

	delete[] refSupport3d;
	delete[] curPoints2d;
	delete[] curPoints3d;

	delete[] curPointsdZ1;
	delete[] curPointsdZ2;
	delete[] curPointsdZ3;
	delete[] curPointsdZ4;
	delete[] curPointsdZ5;
	delete[] curPointsdZ6;

	delete[] offsets;
	delete[] colors;
	delete[] supportOffsets;
	delete[] supportColors;

	delete[] points2d_dx1;
	delete[] points2d_dx2;
	delete[] points2d_dx3;
	delete[] points2d_dx4;
	delete[] points2d_dx5;
	delete[] points2d_dx6;
	delete[] points2d_dxz;
}


void PointCloud3D::addRefPoint( float x, float y, float z, unsigned int offset, unsigned char color, float i, float j)
{
	refPoints3d[nSamplePoints*3+0] = x;
	refPoints3d[nSamplePoints*3+1] = y;
	refPoints3d[nSamplePoints*3+2] = z;
	refPoints2d[nSamplePoints*2+0] = i;
	refPoints2d[nSamplePoints*2+1] = j;
	offsets[nSamplePoints] = offset;
	colors[nSamplePoints] = color;
	nSamplePoints++;
}

void PointCloud3D::addRefPointSupport( float x, float y, float z, unsigned int offset, unsigned char color, float i, float j)
{
	refSupport3d[nSupportPoints*3+0] = x;
	refSupport3d[nSupportPoints*3+1] = y;
	refSupport3d[nSupportPoints*3+2] = z;
	refSupport2d[nSupportPoints*2+0] = i;
	refSupport2d[nSupportPoints*2+1] = j;

	supportOffsets[nSupportPoints] = offset;
	supportColors[nSupportPoints] = color;
	nSupportPoints++;
}

void PointCloud3D::reset() {
	nSamplePoints = 0;
	nSupportPoints = 0;
	differentialsComputed = false;
}

void PointCloud3D::updateRefDevice() {
	if (nSupportPoints > 0) {
		// copy support points on device to same memory buffer with sample points 
		// this allows single warp call for all points
		memcpy(refPoints3d+nSamplePoints*3,refSupport3d,nSupportPoints*sizeof(float)*3);
		memcpy(colors+nSamplePoints,supportColors,nSupportPoints*sizeof(char));
		memcpy(offsets+nSamplePoints,supportOffsets,nSupportPoints*sizeof(int));
	}
}
