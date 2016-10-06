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


#include <string>
#include <GL/glew.h> // GLEW Library
// CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <cublas.h>
#include <assert.h>
#include "shader.h"
#include "TriangleBuffer2.h"
#include "hostUtils.h"

TriangleBuffer2::TriangleBuffer2(float *points, int numTriangles, unsigned int *vertexIndices3, const char *vbufferName)
{
    devPtr      = NULL;
    indexDevPtr = NULL;
    vbo    = -1;
    ibo    = -1;
    this->maxNumVertices = 0;
    this->numVertices    = 0;
    this->numElements    = 0;
    this->maxNumElements = 0;
    this->cudaStream     = 0;
    this->vertexDataHost = NULL;
    this->indexDataHost = NULL;
    this->faceNormals = NULL;
    this->vertexNormals = NULL;
    sprintf(&name[0],"default vbuffer name");
    this->stride = TRIBUFFER_STRIDE;

    init(points,numTriangles,vertexIndices3,vbufferName);
}

void TriangleBuffer2::generateNormals(float *points, int numTriangles, unsigned int *vertexIndex3) {
    // clear buffers
    memset(faceNormals,0,sizeof(float)*numTriangles*3);
    memset(vertexNormals,0,sizeof(float)*numTriangles*9);
    // compute face normals:
    for (int i = 0; i < numTriangles; i++) {
        float *p0 = &points[vertexIndex3[i*3+0]*3];
        float *p1 = &points[vertexIndex3[i*3+1]*3];
        float *p2 = &points[vertexIndex3[i*3+2]*3];

        float u[3],v[3],n[3];
        u[0] = p1[0]-p0[0]; u[1] = p1[1]-p0[1]; u[2] = p1[2]-p0[2];
        v[0] = p2[0]-p0[0]; v[1] = p2[1]-p0[1]; v[2] = p2[2]-p0[2];

        n[0] =  u[1]*v[2] - u[2]*v[1];
        n[1] = -(u[0]*v[2] - u[2]*v[0]);
        n[2] =  u[0]*v[1] - u[1]*v[0];

        float len = sqrtf(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        faceNormals[i*3+0] = -n[0]/len;
        faceNormals[i*3+1] = -n[1]/len;
        faceNormals[i*3+2] = -n[2]/len;

        vertexNormals[vertexIndex3[i*3+0]*3+0] += faceNormals[i*3+0];
        vertexNormals[vertexIndex3[i*3+0]*3+1] += faceNormals[i*3+1];
        vertexNormals[vertexIndex3[i*3+0]*3+2] += faceNormals[i*3+2];

        vertexNormals[vertexIndex3[i*3+1]*3+0] += faceNormals[i*3+0];
        vertexNormals[vertexIndex3[i*3+1]*3+1] += faceNormals[i*3+1];
        vertexNormals[vertexIndex3[i*3+1]*3+2] += faceNormals[i*3+2];

        vertexNormals[vertexIndex3[i*3+2]*3+0] += faceNormals[i*3+0];
        vertexNormals[vertexIndex3[i*3+2]*3+1] += faceNormals[i*3+1];
        vertexNormals[vertexIndex3[i*3+2]*3+2] += faceNormals[i*3+2];
    }

    // post-normalize
    for (int i = 0; i < numTriangles*3; i++) {
        float vn[3];
        vn[0] = vertexNormals[vertexIndex3[i]*3+0];
        vn[1] = vertexNormals[vertexIndex3[i]*3+1];
        vn[2] = vertexNormals[vertexIndex3[i]*3+2];
        float len = sqrt(vn[0]*vn[0]+vn[1]*vn[1]+vn[2]*vn[2]);
        vertexNormals[vertexIndex3[i]*3+0] = vn[0]/len;
        vertexNormals[vertexIndex3[i]*3+1] = vn[1]/len;
        vertexNormals[vertexIndex3[i]*3+2] = vn[2]/len;
    }
}

void TriangleBuffer2::update(cv::Mat *xyzImage, int xyzStride) {
    int npoints = xyzImage->cols*xyzImage->rows;
    if (npoints == 0) return;
    if (npoints > maxNumElements) {
        printf("trimesh: input grid contains too many points! (%d vs %d)\n",npoints,maxNumElements); fflush(stdin); fflush(stdout); return;
    }

    float *points = (float*)xyzImage->ptr();
    int w = xyzImage->cols; int wp = w*xyzStride;
    int h = xyzImage->rows;
    int np = 0;
    for (int y = 0; y < h-1; y++) {
        for (int x = 0; x < w-1; x++) {
            int off = x+y*w;
            int offp = off*xyzStride;
            if (points[offp+6]>0 && points[offp+xyzStride+6]>0 && points[offp+wp+6]>0 && points[offp+wp+xyzStride+6]>0)
            {
                this->vertexDataHost[np*stride+0] = points[offp+0];
                this->vertexDataHost[np*stride+1] = points[offp+1];
                this->vertexDataHost[np*stride+2] = points[offp+2];
                this->vertexDataHost[np*stride+3] = -points[offp+3];
                this->vertexDataHost[np*stride+4] = -points[offp+4];
                this->vertexDataHost[np*stride+5] = -points[offp+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
                this->vertexDataHost[np*stride+0] = points[offp+xyzStride+0];
                this->vertexDataHost[np*stride+1] = points[offp+xyzStride+1];
                this->vertexDataHost[np*stride+2] = points[offp+xyzStride+2];
                this->vertexDataHost[np*stride+3] = -points[offp+xyzStride+3];
                this->vertexDataHost[np*stride+4] = -points[offp+xyzStride+4];
                this->vertexDataHost[np*stride+5] = -points[offp+xyzStride+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
                this->vertexDataHost[np*stride+0] = points[offp+xyzStride+wp+0];
                this->vertexDataHost[np*stride+1] = points[offp+xyzStride+wp+1];
                this->vertexDataHost[np*stride+2] = points[offp+xyzStride+wp+2];
                this->vertexDataHost[np*stride+3] = -points[offp+xyzStride+wp+3];
                this->vertexDataHost[np*stride+4] = -points[offp+xyzStride+wp+4];
                this->vertexDataHost[np*stride+5] = -points[offp+xyzStride+wp+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
                this->vertexDataHost[np*stride+0] = points[offp+0];
                this->vertexDataHost[np*stride+1] = points[offp+1];
                this->vertexDataHost[np*stride+2] = points[offp+2];
                this->vertexDataHost[np*stride+3] = -points[offp+3];
                this->vertexDataHost[np*stride+4] = -points[offp+4];
                this->vertexDataHost[np*stride+5] = -points[offp+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
                this->vertexDataHost[np*stride+0] = points[offp+xyzStride+wp+0];
                this->vertexDataHost[np*stride+1] = points[offp+xyzStride+wp+1];
                this->vertexDataHost[np*stride+2] = points[offp+xyzStride+wp+2];
                this->vertexDataHost[np*stride+3] = -points[offp+xyzStride+wp+3];
                this->vertexDataHost[np*stride+4] = -points[offp+xyzStride+wp+4];
                this->vertexDataHost[np*stride+5] = -points[offp+xyzStride+wp+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
                this->vertexDataHost[np*stride+0] = points[offp+wp+0];
                this->vertexDataHost[np*stride+1] = points[offp+wp+1];
                this->vertexDataHost[np*stride+2] = points[offp+wp+2];
                this->vertexDataHost[np*stride+3] = -points[offp+wp+3];
                this->vertexDataHost[np*stride+4] = -points[offp+wp+4];
                this->vertexDataHost[np*stride+5] = -points[offp+wp+5];
                this->vertexDataHost[np*stride+6] = 0.0f;
                this->vertexDataHost[np*stride+7] = 0.0f;
                this->vertexDataHost[np*stride+8] = 0.0f;
                np++;
            }
        }
    }
    numElements = np;
    numVertices = np;
    if (np <= 0) return;
    upload();
}

void TriangleBuffer2::init(float *points, int numTriangles, unsigned int *vertexIndex3, const char *vbufferName) {
    devPtr = NULL;
    if (vbufferName != NULL) strcpy(this->name,vbufferName);

    printf("allocating trimesh with %d tris\n",numTriangles);
    this->maxNumVertices = numTriangles*3;
    this->numVertices    = numTriangles*3;
    this->numElements    = numTriangles*3;
    this->maxNumElements = numTriangles*3;

    /*printFreeDeviceMemory();
    printf("numVerts: %d\n",numVertices);
    printf("vbuf requested: %3.1fmb\n",float(numVertices * sizeof(float) * stride) / (1024*1024));
    printf("elems requested: %3.1fmb\n",float(numTriangles * 3 * sizeof(int)) / (1024*1024));
*/
    // allocate buffers for cuda interop
    glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numElements * sizeof(float) * stride, NULL, GL_STREAM_COPY);
    if (vbo == 0) {
        printf("vbo allocation failed!\n");
        fflush(stdin); fflush(stdout); fflush(stderr);
    }
    assert(vbo != 0);
    checkCudaErrors(cudaGLRegisterBufferObject(vbo));

    indexDevPtr = NULL;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numElements * sizeof(int), NULL, GL_STREAM_COPY);
    assert(ibo != 0);
    checkCudaErrors(cudaGLRegisterBufferObject(ibo));
    this->cudaStream     = 0;
    this->vertexDataHost = new float[numElements*stride]; memset(vertexDataHost,0,sizeof(float)*numElements*stride);
    this->indexDataHost  = new unsigned int[numElements];
    this->faceNormals   = new float[numTriangles*3]; memset(faceNormals,0,sizeof(float)*numTriangles*3);
    this->vertexNormals = new float[numVertices*3]; memset(vertexNormals,0,sizeof(float)*numVertices*3);

    if (points != NULL) {
        generateNormals(points,numTriangles,vertexIndex3);
        // initialize default vertices
        for (int i = 0; i < numElements; i++) {
            this->vertexDataHost[i*stride+0] = points[vertexIndex3[i]*3+0];
            this->vertexDataHost[i*stride+1] = points[vertexIndex3[i]*3+1];
            this->vertexDataHost[i*stride+2] = points[vertexIndex3[i]*3+2];
            this->vertexDataHost[i*stride+3] = vertexNormals[vertexIndex3[i]*3+0];
            this->vertexDataHost[i*stride+4] = vertexNormals[vertexIndex3[i]*3+1];
            this->vertexDataHost[i*stride+5] = vertexNormals[vertexIndex3[i]*3+2];
            this->vertexDataHost[i*stride+6] = 1.0f;
            this->vertexDataHost[i*stride+7] = 0.0f;
            this->vertexDataHost[i*stride+8] = 0.0f;
        }
    }
    // straight-forward indexing:
    for (int i = 0; i < numElements; i++) {
        this->indexDataHost[i] = i;
    }
    upload();
}

TriangleBuffer2::TriangleBuffer2() {
    devPtr      = NULL;
    indexDevPtr = NULL;
    vbo    = -1;
    ibo    = -1;
    this->maxNumVertices = 0;
    this->numVertices    = 0;
    this->numElements    = 0;
    this->maxNumElements = 0;
    this->cudaStream     = 0;
    this->vertexDataHost = NULL;
    this->indexDataHost = NULL;
    this->faceNormals   = NULL;
    this->vertexNormals = NULL;
    sprintf(&name[0],"default vbuffer name");
    this->stride = TRIBUFFER_STRIDE;
}

void TriangleBuffer2::release() {
 //   printf("releasing %s\n",name);
    if (vbo != -1) {
//        printf("releasing vbo (%s)\n",name);
        checkCudaErrors(cudaGLUnregisterBufferObject(vbo));
        glDeleteBuffers(1, &vbo);
        vbo = -1;

    }
    if (ibo != -1) {
 //       printf("releasing ibo (%s)\n",name);
        checkCudaErrors(cudaGLUnregisterBufferObject(ibo));
        glDeleteBuffers(1, &ibo);
        ibo = -1;
    }
    if (vertexDataHost != NULL) delete[] vertexDataHost; vertexDataHost = NULL;
    if (indexDataHost != NULL) delete[] indexDataHost; indexDataHost = NULL;
    if (faceNormals != NULL) delete[] faceNormals; faceNormals = NULL;
    if (vertexNormals != NULL) delete[] vertexNormals; vertexNormals = NULL;
}

TriangleBuffer2::~TriangleBuffer2()
{

}

void *TriangleBuffer2::lock() {
	// already locked?
	if (devPtr != NULL) return devPtr; 
	// map the buffer to CUDA
    checkCudaErrors(cudaGLMapBufferObjectAsync((void**)&devPtr,vbo,cudaStream));
	return devPtr;
}

void TriangleBuffer2::unlock() {
   // printf("unlocking %s\n",name);
    if (devPtr != NULL) { checkCudaErrors(cudaGLUnmapBufferObjectAsync(vbo,cudaStream)); devPtr = NULL; }
}

void *TriangleBuffer2::lockIndex() {
        // already locked?
        if (indexDevPtr != NULL) return indexDevPtr;
        // map the buffer to CUDA
            checkCudaErrors(cudaGLMapBufferObjectAsync((void**)&indexDevPtr,ibo,cudaStream));
        return indexDevPtr;
}

void TriangleBuffer2::unlockIndex() {
        if (indexDevPtr != NULL) { 	checkCudaErrors(cudaGLUnmapBufferObjectAsync(ibo,cudaStream)); indexDevPtr = NULL; }
}


void TriangleBuffer2::setStream( cudaStream_t stream )
{
	this->cudaStream = stream;
}

void TriangleBuffer2::addVertex(float x, float y, float z, float r, float g, float b) 
{
    if (numVertices >= maxNumVertices-1) return;
    vertexDataHost[numVertices*stride+0] = x;
    vertexDataHost[numVertices*stride+1] = y;
    vertexDataHost[numVertices*stride+2] = z;

    int rgbOffset = 6;
    vertexDataHost[numVertices*stride+rgbOffset+0] = r;
    vertexDataHost[numVertices*stride+rgbOffset+1] = g;
    vertexDataHost[numVertices*stride+rgbOffset+2] = b;
    indexDataHost[numElements] = numElements;
    numVertices++; numElements++;
}

void TriangleBuffer2::upload() {
    float *devicePointer = (float*)lock();
    cudaMemcpyAsync(devicePointer,vertexDataHost,sizeof(float)*stride*numVertices,cudaMemcpyHostToDevice,cudaStream);
    unlock();

    unsigned int *indexPointer = (unsigned int*)lockIndex();
    cudaMemcpyAsync(indexPointer,indexDataHost,sizeof(int)*numElements,cudaMemcpyHostToDevice,cudaStream);
    unlockIndex();    
}

void TriangleBuffer2::render() {
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 6;

    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLES, numElements,GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void TriangleBuffer2::render(Shader *shader, float *lightPos) {
    glEnable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);

    if (stride != TRIBUFFER_STRIDE) {
        printf("glsl rendering requires standard stride %d\n",TRIBUFFER_STRIDE);
        fflush(stdin); fflush(stdout);
        return;
    }

    glBindBuffer( GL_ARRAY_BUFFER, vbo );

    glDisableClientState( GL_VERTEX_ARRAY );
    glDisableClientState( GL_COLOR_ARRAY );

    int vertexIndex = shader->getAttrib("inputVertex");    
    int colorIndex  = shader->getAttrib("inputColor");
    int normalIndex = shader->getAttrib("inputNormal");

    // enable the attribute at that location
    glEnableVertexAttribArray(vertexIndex);
    glEnableVertexAttribArray(colorIndex);
    glEnableVertexAttribArray(normalIndex);

    unsigned char *ptrBase = 0;
    glVertexAttribPointer(vertexIndex, 3, GL_FLOAT, GL_FALSE, stride*sizeof(float), ptrBase);  ptrBase += 3*sizeof(float);
    glVertexAttribPointer(normalIndex, 3, GL_FLOAT, GL_FALSE, stride*sizeof(float), ptrBase);
    ptrBase += 3*sizeof(float);
    glVertexAttribPointer(colorIndex,  3, GL_FLOAT, GL_FALSE, stride*sizeof(float), ptrBase);
    shader->bind();
    // uniforms are set after bind:
    shader->setUniformVec4("globalLight",lightPos);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLES, numElements,GL_UNSIGNED_INT, (GLvoid*) 0);

    glDisableVertexAttribArray(vertexIndex);
    glDisableVertexAttribArray(colorIndex);
    glDisableVertexAttribArray(normalIndex);

    shader->unbind();
    glDisable(GL_LIGHTING);
}


void TriangleBuffer2::renderAll() { 
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 6;
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);
    glDrawArrays(GL_POINTS,0,numVertices);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void TriangleBuffer2::setVertexAmount( int num )
{
	if (num > maxNumVertices) printf("vertex buffer exceeded!\n"); else numVertices = num;
}
