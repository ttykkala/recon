/********************************************************************
	CNRS Copyright (C) 2011.
	This software is owned by CNRS. No unauthorized distribution of the code is allowed.
	
	Address:
	CNRS/I3S
	2000 Route des Lucioles
	06903 Sophia-Antipolis
	
	created:	14:3:2011   21:07
	filename: 	d:\projects\phd-project\kyklooppi\realtime\utils\rendering\GridBuffer.h
	author:		Tommi Tykkala
	purpose:	for storing 3D reconstructions
*********************************************************************/
#pragma once

#define __LINE_BUFFER_H__

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

static const int lbufferSize = 2;

class LineBuffer : public RenderBufferObject {
private:
	enum {
		VERTEX_BUFFER = 0,
		COLOR_BUFFER1 = 1
	};
public:
	LineBuffer(int size);
	~LineBuffer();
	void addLine(float x, float y, float z, float x2, float y2, float z2, unsigned char r, unsigned char g, unsigned char b);
    void render();
    void render(int nSegments);
	void upload();
	void reset();
	void getPoint(unsigned int index, float *x, float *y, float *z);
	float *getPoint(int index);
	unsigned char *getColor(int index);
	int getPointCount();
    int getMaxPointCount();
	int nPoints;
	int maxPoints;
	float *xyz;
	unsigned char *rgb1;
	unsigned int buffers[lbufferSize];
	unsigned int newVerticeCount;
};	
