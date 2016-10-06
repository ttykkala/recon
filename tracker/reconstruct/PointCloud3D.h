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

class PointCloud3D {
public:
	bool differentialsComputed;
	int nSamplePoints;
	int nSupportPoints;
	int nMaxPoints;
	// main memory pointers
	float *refPoints2d,*refSupport2d; 
	float *refPoints3d; float *refSupport3d;
	float *tmpPoints3d;
	float *curPoints3d, *curPoints2d;
	float *curPointsdZ1;
	float *curPointsdZ2;
	float *curPointsdZ3;
	float *curPointsdZ4;
	float *curPointsdZ5;
	float *curPointsdZ6;
	unsigned int *offsets;
	unsigned char *colors;

	unsigned int *supportOffsets;
	unsigned char *supportColors;

	// main memory differentials
	float *points2d_dx1; 
	float *points2d_dx2;
	float *points2d_dx3;
	float *points2d_dx4;
	float *points2d_dx5;
	float *points2d_dx6;
	float *points2d_dxz;

	PointCloud3D(int numMaxPoints);
	~PointCloud3D();
	void addRefPoint(float x, float y, float z, unsigned int offset, unsigned char color, float i, float j);
	void addRefPointSupport(float x, float y, float z, unsigned int offset, unsigned char color, float i, float j);
	void reset();
	void updateRefDevice();
};
