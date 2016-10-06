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

#define __VERTEX_BUFFER2D_H__

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/glew.h> // GLEW Library
#include "RenderBuffer.h"

class VertexBuffer2d : public RenderBufferObject {
private:
	enum {
		VERTEX_BUFFER = 0,
		COLOR_BUFFER = 1
	};
	int maxPoints;
	int nPoints;
	unsigned int buffers[2];
	void *cudaBuffers[2];
public:
	VertexBuffer2d(int size);
	~VertexBuffer2d();
	void copyLineData(float *xy, unsigned char r, unsigned char g, unsigned char b, int nLines);
	void render();
	void reset();
};
