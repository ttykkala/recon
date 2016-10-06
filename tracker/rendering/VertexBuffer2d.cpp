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
#include "VertexBuffer2d.h"
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


VertexBuffer2d::VertexBuffer2d(int size) {
	maxPoints = 2*size;
	glGenBuffers(2, buffers);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, buffers[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER_ARB, 2*maxPoints*sizeof(float), NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer((struct cudaGraphicsResource**)&cudaBuffers[VERTEX_BUFFER], buffers[VERTEX_BUFFER], cudaGraphicsMapFlagsWriteDiscard);

	glBindBuffer(GL_ARRAY_BUFFER_ARB, buffers[COLOR_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER_ARB, 3*maxPoints, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer((struct cudaGraphicsResource**)&cudaBuffers[COLOR_BUFFER], buffers[COLOR_BUFFER], cudaGraphicsMapFlagsWriteDiscard);
	nPoints = 0;
}

VertexBuffer2d::~VertexBuffer2d() {
	cutilSafeCall(cudaGraphicsUnregisterResource((struct cudaGraphicsResource*)cudaBuffers[VERTEX_BUFFER]));
	cutilSafeCall(cudaGraphicsUnregisterResource((struct cudaGraphicsResource*)cudaBuffers[COLOR_BUFFER]));
	glDeleteBuffers(2, buffers);
}

void VertexBuffer2d::reset() {
	this->nPoints = 0;
}
void VertexBuffer2d::copyLineData(float *xy, unsigned char r, unsigned char g, unsigned char b, int nLines) {
	if (nLines >= maxPoints) return;
	this->nPoints = nLines*4; 

	float *devPtr = NULL;
	size_t numBytes;
	struct cudaGraphicsResource *cudaResource = (struct cudaGraphicsResource*)cudaBuffers[VERTEX_BUFFER];
	cutilSafeCall(cudaGraphicsMapResources(1, &cudaResource, 0));
	cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &numBytes, (struct cudaGraphicsResource*)cudaBuffers[VERTEX_BUFFER]);
	cudaMemcpy(devPtr, xy, nPoints*sizeof(float)*2, cudaMemcpyDeviceToDevice );
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cudaResource, 0));	
}


void VertexBuffer2d::render() {
	//glBindBuffer(GL_ARRAY_BUFFER, buffers[COLOR_BUFFER]);
	//glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[VERTEX_BUFFER]);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_LINES, 0, nPoints/2);
	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}
