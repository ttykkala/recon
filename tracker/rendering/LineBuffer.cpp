#include "LineBuffer.h"
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

//static float length = 0;

LineBuffer::LineBuffer(int size) {
	xyz = new float[size*3*2];
	rgb1 = new unsigned char[size*3*2];
	maxPoints = size*2;
	glGenBuffers(2, buffers);
	// allocate buffers
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glBufferData(GL_ARRAY_BUFFER, size*3*2, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, size*3*2*sizeof(float), NULL, GL_STREAM_DRAW);
	reset();
};

void LineBuffer::getPoint(unsigned int index, float *x, float *y, float *z) 
{ 
	*x = xyz[index*3+0]; 
	*y = xyz[index*3+1]; 
	*z = xyz[index*3+2];
}

float *LineBuffer::getPoint(int index) { return &xyz[index*3]; }
unsigned char *LineBuffer::getColor(int index) { return &rgb1[index*3]; }
int LineBuffer::getPointCount() { return nPoints; }
int LineBuffer::getMaxPointCount() { return maxPoints; }

LineBuffer::~LineBuffer() {
	if (xyz != NULL) delete[] xyz;
	if (rgb1 != NULL) delete[] rgb1;
	glDeleteBuffers(2, buffers);
}

void LineBuffer::reset() { 
	nPoints = 0; 
	newVerticeCount = 0;
}

void LineBuffer::addLine(float x, float y, float z, float x2, float y2, float z2, unsigned char r, unsigned char g, unsigned char b) {
	if (nPoints >= maxPoints-2) return;
	
	//length += sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2));
	//printf("total length: %f\n",length);

	xyz[nPoints*3+0] = x;
	xyz[nPoints*3+1] = y;
	xyz[nPoints*3+2] = z;
	
	xyz[(nPoints+1)*3+0] = x2;
	xyz[(nPoints+1)*3+1] = y2;
	xyz[(nPoints+1)*3+2] = z2;
	
	rgb1[nPoints*3+0] = r;
	rgb1[nPoints*3+1] = g;
	rgb1[nPoints*3+2] = b;
	
	rgb1[(nPoints+1)*3+0] = r;
	rgb1[(nPoints+1)*3+1] = g;
	rgb1[(nPoints+1)*3+2] = b;

	nPoints+=2;
	newVerticeCount+=2;
}

void LineBuffer::upload() {
	if (newVerticeCount == 0) return;
	int srcOffset = (nPoints-newVerticeCount)*3;
	//int newVerticeCount = nPoints;

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
	newVerticeCount = 0;
}

void LineBuffer::render() {
	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_TEXTURE_2D);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
    glPolygonMode(GL_FRONT,GL_FILL);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glDrawArrays(GL_LINES, 0, nPoints);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopAttrib();
}

void LineBuffer::render(int nSegments) {
    if (nSegments*2 >= nPoints) nSegments = nPoints/2;
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER1]);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
//    glPolygonMode(GL_FRONT,GL_FILL);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glDrawArrays(GL_POINTS, 0, nSegments*2);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glPopAttrib();
}
