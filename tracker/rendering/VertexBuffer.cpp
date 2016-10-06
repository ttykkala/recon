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
#include "VertexBuffer.h"
#include <float.h>
#include <string.h>

VertexBuffer::VertexBuffer() { 
	xyz = NULL; 
	rgb1 = NULL; 
	rgb2 = NULL; 
	nPoints = 0; 
	maxPoints = 0;
	newVerticeCount = 0;
	buffers[2]=buffers[1]=buffers[0]=0;
	bboxMin[0]=bboxMin[1]=bboxMin[2]=FLT_MAX;
	bboxMax[0]=bboxMax[1]=bboxMax[2]=FLT_MIN;
	firstBlood = true;
	pivotPoint[0]=pivotPoint[1]=pivotPoint[2]=0;
}

VertexBuffer::VertexBuffer(int size) {
	xyz = new float[size*3];
	rgb1 = new unsigned char[size*3];
	rgb2 = new unsigned char[size*3];
	maxPoints = size;
	newVerticeCount = 0;
	glGenBuffers(3, buffers);
	// allocate buffers
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glBufferData(GL_ARRAY_BUFFER, size*3, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER2]);
	glBufferData(GL_ARRAY_BUFFER, size*3, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, size*3*sizeof(float), NULL, GL_STREAM_DRAW);
	reset();
}
void VertexBuffer::getPoint(unsigned int index, float *x, float *y, float *z) 
{ 
	*x = xyz[index*3+0]; 
	*y = xyz[index*3+1]; 
	*z = xyz[index*3+2];
}

float *VertexBuffer::getPoint(int index) { return &xyz[index*3]; }
unsigned char *VertexBuffer::getColor1(int index) { return &rgb1[index*3]; }
unsigned char *VertexBuffer::getColor2(int index) { return &rgb2[index*3]; }
int VertexBuffer::getPointCount() { return nPoints; }
void VertexBuffer::getMeanPoints(float *x, float *y, float *z) {*x = pivotPoint[0]; *y = pivotPoint[1]; *z = pivotPoint[2]; }

VertexBuffer::~VertexBuffer() {
	if (xyz != NULL) delete[] xyz;
	if (rgb1 != NULL) delete[] rgb1;
	if (rgb2 != NULL) delete[] rgb2;
	glDeleteBuffers(3, buffers);
}

void VertexBuffer::reset() { 
	nPoints = 0; 
	newVerticeCount = 0;
	bboxMin[0]=bboxMin[1]=bboxMin[2]=FLT_MAX;
	bboxMax[0]=bboxMax[1]=bboxMax[2]=FLT_MIN;
	pivotPoint[0]=pivotPoint[1]=pivotPoint[2]=0;
	firstBlood = true;
}


void VertexBuffer::addVertex(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b,unsigned char r2, unsigned char g2, unsigned char b2) {
	if (nPoints >= maxPoints-1) return;
	xyz[nPoints*3+0] = x;
	xyz[nPoints*3+1] = y;
	xyz[nPoints*3+2] = z;
	rgb1[nPoints*3+0] = r;
	rgb1[nPoints*3+1] = g;
	rgb1[nPoints*3+2] = b;
	rgb2[nPoints*3+0] = r2;
	rgb2[nPoints*3+1] = g2;
	rgb2[nPoints*3+2] = b2;
	nPoints++;
	newVerticeCount++;
}

void VertexBuffer::upload() {
	if (newVerticeCount == 0) return;
	int srcOffset = (nPoints-newVerticeCount)*3;
	//int newVerticeCount = nPoints;

	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	//glBufferData(GL_ARRAY_BUFFER, nPoints*3, NULL, GL_STREAM_DRAW);
	unsigned char* colorBuffer1 = (unsigned char*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	colorBuffer1 += srcOffset;
	memcpy(colorBuffer1, &rgb1[srcOffset], newVerticeCount*3);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER2]);
	//glBufferData(GL_ARRAY_BUFFER, nPoints*3, NULL, GL_STREAM_DRAW);
	unsigned char* colorBuffer2 = (unsigned char*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	colorBuffer2 += srcOffset;
	memcpy(colorBuffer2, &rgb2[srcOffset], newVerticeCount*3);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	//glBufferData(GL_ARRAY_BUFFER, nPoints*3*sizeof(float), NULL, GL_STREAM_DRAW);
	float* vertexBuffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	vertexBuffer += srcOffset;//*sizeof(float);
	memcpy(vertexBuffer, &xyz[srcOffset], newVerticeCount*3*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	newVerticeCount = 0;
	if (firstBlood) { setPivotPoint(); firstBlood = false;}
}

void VertexBuffer::render() {
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, nPoints);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void VertexBuffer::computeBBox()
{
	int offset = 0;
	for (int i = 0; i < nPoints; i++,offset+=3) {
		if (xyz[offset+0] < bboxMin[0]) bboxMin[0] = xyz[offset+0];
		if (xyz[offset+1] < bboxMin[1]) bboxMin[1] = xyz[offset+1];
		if (xyz[offset+2] < bboxMin[2]) bboxMin[2] = xyz[offset+2];
		if (xyz[offset+0] > bboxMax[0]) bboxMax[0] = xyz[offset+0];
		if (xyz[offset+1] > bboxMax[1]) bboxMax[1] = xyz[offset+1];
		if (xyz[offset+2] > bboxMax[2]) bboxMax[2] = xyz[offset+2];
	}
}

void VertexBuffer::updateBBox()
{
	int offset = (nPoints-newVerticeCount)*3;
	for (int i = 0; i < newVerticeCount; i++,offset+=3) {
		if (xyz[offset+0] < bboxMin[0]) bboxMin[0] = xyz[offset+0];
		if (xyz[offset+1] < bboxMin[1]) bboxMin[1] = xyz[offset+1];
		if (xyz[offset+2] < bboxMin[2]) bboxMin[2] = xyz[offset+2];
		if (xyz[offset+0] > bboxMax[0]) bboxMax[0] = xyz[offset+0];
		if (xyz[offset+1] > bboxMax[1]) bboxMax[1] = xyz[offset+1];
		if (xyz[offset+2] > bboxMax[2]) bboxMax[2] = xyz[offset+2];
	}
}

void VertexBuffer::setPivotPoint()
{
	if (nPoints == 0) return;
	pivotPoint[0]=pivotPoint[1]=pivotPoint[2]=0;
	int offset = 0;
	for (int i = 0; i < nPoints; i++,offset+=3) {
		pivotPoint[0] += xyz[offset+0];
		pivotPoint[1] += xyz[offset+1];
		pivotPoint[2] += xyz[offset+2];
	}
	pivotPoint[0] /= nPoints;
	pivotPoint[1] /= nPoints;
	pivotPoint[2] /= nPoints;
}
