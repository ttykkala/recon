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

#define __GRID_BUFFER_H__

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(__APPLE__) && defined(__MACH__)
#include <OpenGL/gl.h>	// Header File For The OpenGL32 Library
#include <OpenGL/glu.h>	// Header File For The GLu32 Library
#else
#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library
#endif
#include "RenderBuffer.h"
#include <map>
static const int gbufferSize = 2;

class GridBuffer : public RenderBufferObject {
private:
	//void setPivotPoint();
	float gridMin[3];
	float gridMax[3];
	float boxDim;
	unsigned int gridReso;
	enum {
		VERTEX_BUFFER = 0,
		COLOR_BUFFER1 = 1
	};
	std::map<unsigned int,int> sparseGrid;
public:
	GridBuffer(); 
	GridBuffer(float x0, float y0, float z0, float boxDim, unsigned int gridResolution);
	~GridBuffer();
//	void addQuad(float *v0, float *v1, float *v2, float *v3, unsigned char r, unsigned char g, unsigned char b);	
	void addVoxel(float *v, unsigned char *c);	
	void render();
	void upload();
	void reset();
	void getPoint(unsigned int index, float *x, float *y, float *z);
	float *getPoint(int index);
	unsigned char *getColor1(int index);
	unsigned char *getColor2(int index);
	int getPointCount();
	void getMeanPoints(float *x, float *y, float *z);
	void computeBBox();
	void updateBBox();
	void renderGridBox();
	bool voxelAllocated(float x, float y, float z);
	int nPoints;
	int maxPoints;
	float *xyz;
	unsigned char *voxelBit;
	unsigned char *rgb1;
	unsigned int buffers[gbufferSize];
	unsigned int newVerticeCount;
	float bboxMin[3];
	float bboxMax[3];
	float pivotPoint[3];
};	
