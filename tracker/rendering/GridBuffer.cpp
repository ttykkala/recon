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

#include "GridBuffer.h"
#include <float.h>
#include <stdio.h>

GridBuffer::GridBuffer() { 
	xyz = NULL; 
	rgb1 = NULL; 
	nPoints = 0; 
	maxPoints = 0;
	newVerticeCount = 0;
	buffers[2]=buffers[1]=buffers[0]=0;
	bboxMin[0]=bboxMin[1]=bboxMin[2]=FLT_MAX;
	bboxMax[0]=bboxMax[1]=bboxMax[2]=-FLT_MAX;
	//firstBlood = true;
	pivotPoint[0]=pivotPoint[1]=pivotPoint[2]=0;
}

GridBuffer::GridBuffer(float x0, float y0, float z0, float boxDim, unsigned int gridResolution) {
	gridMin[0]=gridMin[1]=gridMin[2]=-boxDim/2.0f;
	gridMax[0]=gridMax[1]=gridMax[2]= boxDim/2.0f;
	this->boxDim = boxDim;
	this->gridReso = gridResolution;
	int size = gridResolution*gridResolution*gridResolution;
	xyz = new float[size*3*24];
	rgb1 = new unsigned char[size*3*24];
	maxPoints = size*24;
	voxelBit = new unsigned char[gridResolution*gridResolution*gridResolution];
	pivotPoint[0] = x0; pivotPoint[1] = y0; pivotPoint[2] = z0;
	glGenBuffers(2, buffers);
	// allocate buffers
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glBufferData(GL_ARRAY_BUFFER, size*3*24, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, size*3*24*sizeof(float), NULL, GL_STREAM_DRAW);
	reset();
};

void GridBuffer::getPoint(unsigned int index, float *x, float *y, float *z) 
{ 
	*x = xyz[index*3+0]; 
	*y = xyz[index*3+1]; 
	*z = xyz[index*3+2];
}

float *GridBuffer::getPoint(int index) { return &xyz[index*3]; }
unsigned char *GridBuffer::getColor1(int index) { return &rgb1[index*3]; }
int GridBuffer::getPointCount() { return nPoints; }
void GridBuffer::getMeanPoints(float *x, float *y, float *z) {*x = pivotPoint[0]; *y = pivotPoint[1]; *z = pivotPoint[2]; }

GridBuffer::~GridBuffer() {
	if (xyz != NULL) delete[] xyz;
	if (rgb1 != NULL) delete[] rgb1;
	glDeleteBuffers(2, buffers);
//	printf("bbox: %f %f %f %f %f %f\n",bboxMin[0],bboxMin[1],bboxMin[2],bboxMax[0],bboxMax[1],bboxMax[2]);
}

void GridBuffer::reset() { 
	nPoints = 0; 
	newVerticeCount = 0;
	bboxMin[0]=bboxMin[1]=bboxMin[2]=FLT_MAX;
	bboxMax[0]=bboxMax[1]=bboxMax[2]=-FLT_MAX;
	memset(voxelBit,0,gridReso*gridReso*gridReso);
}

bool GridBuffer::voxelAllocated(float x, float y, float z) {
	float gx = x - (gridMin[0]+pivotPoint[0]);
	float gy = y - (gridMin[1]+pivotPoint[1]);
	float gz = z - (gridMin[2]+pivotPoint[2]);
	
	// are we in grid domain?
	if (gx > 0 && gx < boxDim && gy > 0 && gy < boxDim && gz > 0 && gz < boxDim) {
		int i = gridReso * gx / boxDim;
		int j = gridReso * gy / boxDim;
		int k = gridReso * gz / boxDim;
		int vo = i+j*gridReso+k*gridReso*gridReso;
		if (voxelBit[vo]) return true;
		voxelBit[vo] = true;
		return false;
	} else return true;
}

void GridBuffer::addVoxel(float *v, unsigned char *c)
{
	if (nPoints >= maxPoints-24) return;
	if (voxelAllocated(v[0],v[1],v[2])) return;

    float cubeDim = boxDim / (gridReso*2.0f);
	// front face
	xyz[(nPoints+0)*3+0] = v[0]-cubeDim; 
	xyz[(nPoints+0)*3+1] = v[1]-cubeDim; 
	xyz[(nPoints+0)*3+2] = v[2]+cubeDim; 
	xyz[(nPoints+1)*3+0] = v[0]+cubeDim; 
	xyz[(nPoints+1)*3+1] = v[1]-cubeDim; 
	xyz[(nPoints+1)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+2)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+2)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+2)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+3)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+3)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+3)*3+2] = v[2]+cubeDim;

	// back face
	xyz[(nPoints+4)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+4)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+4)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+5)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+5)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+5)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+6)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+6)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+6)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+7)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+7)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+7)*3+2] = v[2]-cubeDim;

	// top face
	xyz[(nPoints+8)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+8)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+8)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+9)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+9)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+9)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+10)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+10)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+10)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+11)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+11)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+11)*3+2] = v[2]-cubeDim;

	// bottom face
	xyz[(nPoints+12)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+12)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+12)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+13)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+13)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+13)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+14)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+14)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+14)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+15)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+15)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+15)*3+2] = v[2]+cubeDim;

	// side face left
	xyz[(nPoints+16)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+16)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+16)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+17)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+17)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+17)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+18)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+18)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+18)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+19)*3+0] = v[0]-cubeDim;
	xyz[(nPoints+19)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+19)*3+2] = v[2]+cubeDim;

	// side face right
	xyz[(nPoints+20)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+20)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+20)*3+2] = v[2]-cubeDim;
	xyz[(nPoints+21)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+21)*3+1] = v[1]+cubeDim;
	xyz[(nPoints+21)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+22)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+22)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+22)*3+2] = v[2]+cubeDim;
	xyz[(nPoints+23)*3+0] = v[0]+cubeDim;
	xyz[(nPoints+23)*3+1] = v[1]-cubeDim;
	xyz[(nPoints+23)*3+2] = v[2]-cubeDim;

	for (int i = 0; i < 24; i++) {
		rgb1[(nPoints+i)*3+0] = c[0];
		rgb1[(nPoints+i)*3+1] = c[1];
		rgb1[(nPoints+i)*3+2] = c[2];
	}
	nPoints+=24;
	newVerticeCount+=24;
}

void GridBuffer::upload() {
	if (newVerticeCount == 0) return;
	int srcOffset = (nPoints-newVerticeCount)*3;

	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	unsigned char* colorBuffer1 = (unsigned char*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	colorBuffer1 += srcOffset;
	memcpy(colorBuffer1, &rgb1[srcOffset], newVerticeCount*3);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	float* vertexBuffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	vertexBuffer += srcOffset;//*sizeof(float);
	memcpy(vertexBuffer, &xyz[srcOffset], newVerticeCount*3*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	updateBBox();
	newVerticeCount = 0;
}

void GridBuffer::render() {
	glPushAttrib(GL_ENABLE_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glPolygonMode(GL_FRONT,GL_FILL);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glPointSize(1.0f);
	glDrawArrays(GL_QUADS, 0, nPoints);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopAttrib();
}

void GridBuffer::computeBBox()
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

void GridBuffer::updateBBox()
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

void GridBuffer::renderGridBox()
{
	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_COLOR_MATERIAL);
	glDisable(GL_BLEND);
	glDisable(GL_POINT_SMOOTH);	
	glDisable(GL_LINE_SMOOTH);	
	glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glLineWidth(3.0f);
	glColor3f(1,0,0);

	glPushMatrix();

	glTranslatef(pivotPoint[0],pivotPoint[1],pivotPoint[2]);
	glBegin(GL_LINES);
	glVertex3f(gridMin[0],gridMin[1],gridMin[2]); glVertex3f(gridMax[0],gridMin[1],gridMin[2]);
	glVertex3f(gridMax[0],gridMin[1],gridMin[2]); glVertex3f(gridMax[0],gridMax[1],gridMin[2]);
	glVertex3f(gridMax[0],gridMax[1],gridMin[2]); glVertex3f(gridMin[0],gridMax[1],gridMin[2]);
	glVertex3f(gridMin[0],gridMax[1],gridMin[2]); glVertex3f(gridMin[0],gridMin[1],gridMin[2]);

	glVertex3f(gridMin[0],gridMin[1],gridMax[2]); glVertex3f(gridMax[0],gridMin[1],gridMax[2]);
	glVertex3f(gridMax[0],gridMin[1],gridMax[2]); glVertex3f(gridMax[0],gridMax[1],gridMax[2]);
	glVertex3f(gridMax[0],gridMax[1],gridMax[2]); glVertex3f(gridMin[0],gridMax[1],gridMax[2]);
	glVertex3f(gridMin[0],gridMax[1],gridMax[2]); glVertex3f(gridMin[0],gridMin[1],gridMax[2]);

	glVertex3f(gridMin[0],gridMin[1],gridMin[2]); glVertex3f(gridMin[0],gridMin[1],gridMax[2]);
	glVertex3f(gridMax[0],gridMin[1],gridMin[2]); glVertex3f(gridMax[0],gridMin[1],gridMax[2]);
	glVertex3f(gridMax[0],gridMax[1],gridMin[2]); glVertex3f(gridMax[0],gridMax[1],gridMax[2]);
	glVertex3f(gridMin[0],gridMax[1],gridMin[2]); glVertex3f(gridMin[0],gridMax[1],gridMax[2]);

	glEnd();
	glPopMatrix();
	glPopAttrib();
}
