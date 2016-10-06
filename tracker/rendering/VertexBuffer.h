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

#define __VERTEX_BUFFER_H__

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/glew.h> // GLEW Library
#include "RenderBuffer.h"
static const int vbufferSize = 3;

class VertexBuffer : public RenderBufferObject {
private:
	void setPivotPoint();
	enum {
		VERTEX_BUFFER = 0,
		COLOR_BUFFER1 = 1,
		COLOR_BUFFER2 = 2
	};
public:
	VertexBuffer(); 
	VertexBuffer(int size);
	~VertexBuffer();
	void addVertex(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b,unsigned char r2=0, unsigned char g2=0, unsigned char b2=0);
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
	int nPoints;
	int maxPoints;
	float *xyz;
	unsigned char *rgb1;
	unsigned char *rgb2;
	unsigned int buffers[vbufferSize];
	unsigned int newVerticeCount;
	float bboxMin[3];
	float bboxMax[3];
	float pivotPoint[3];
	bool firstBlood;
};
