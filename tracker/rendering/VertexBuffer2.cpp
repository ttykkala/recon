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


#include <GL/glew.h> // GLEW Library
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <cublas.h>
#include <assert.h>
#include <string.h>
#include "shader.h"
#include "VertexBuffer2.h"

static int refCount = 0;
const int numNormalIbos = 16;
static unsigned int normalIbo[numNormalIbos];
static unsigned int normalElements[numNormalIbos];

VertexBuffer2::VertexBuffer2( int numVertices, bool renderableBuffer,int stride,const char *vbufferName)
{
    init(numVertices,renderableBuffer,stride,vbufferName);
}

int getNormalElementCount(int numVertices, unsigned int iboIndex) {
    unsigned int stride = 1<<iboIndex;
    return 2*int(numVertices/stride);
}


int generateNormalIndices(int maxVertices, unsigned int iboIndex, unsigned int *normalIndex, int vertexBufferStride) {
    unsigned int stride = 1<<iboIndex;
    int offset = 0;
    for (int i = 0; i < maxVertices; i+= stride) {
        normalIndex[offset+0] = i*(vertexBufferStride/3);
        normalIndex[offset+1] = normalIndex[offset+0]+1;
        offset+=2;
    }
    return offset;
}


void VertexBuffer2::init(int numVertices, bool renderableBuffer,int stride,const char *vbufferName) {
    m_renderableBufferFlag = renderableBuffer;
    devPtr = NULL;
    indexDevPtr = NULL;
    this->stride = stride;
    if (vbufferName != NULL) strcpy(this->name,vbufferName);

    if (m_renderableBufferFlag) {
        // allocate buffers for cuda interop
        glGenBuffers( 1, &vbo);
        glBindBuffer( GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(float) * stride, NULL, GL_STREAM_COPY);
        if (vbo == 0) {
            printf("vbo allocation failed!\n");
            fflush(stdin); fflush(stdout); fflush(stderr);
        }
        assert(vbo != 0);
        checkCudaErrors(cudaGLRegisterBufferObject(vbo));

        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, numVertices * sizeof(int), NULL/*&indexDataHost[0]*/, GL_STREAM_COPY);
        if (ibo == 0) {
            printf("ibo allocation failed!\n");
            fflush(stdin); fflush(stdout); fflush(stderr);
        }

        assert(ibo != 0);
        checkCudaErrors(cudaGLRegisterBufferObject(ibo));

        if (refCount == 0) {
            // NOTE! normal indices are only computed for compressed vbuffers
            int maxVertices = 320*240;
            unsigned int *normalIndex = new unsigned int[maxVertices*2];
            for (int i = 0; i < numNormalIbos; i++) {
                normalElements[i] = generateNormalIndices(maxVertices,i,normalIndex,COMPRESSED_STRIDE);
                glGenBuffers(1, &normalIbo[i]);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, normalIbo[i]);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, normalElements[i] * sizeof(int), &normalIndex[0], GL_STREAM_COPY);
            }
            delete[] normalIndex;
        }
    } else {
        vbo    = -1;
        ibo    = -1;
        cudaMalloc((void **)&devPtr, numVertices * sizeof(float) * stride);
        cudaMalloc((void **)&indexDevPtr, numVertices * sizeof(int));
    }
    this->maxNumVertices = numVertices;
    this->numVertices    = 0;
    this->numElements    = 0;
    this->maxNumElements = numVertices;
    this->cudaStream     = 0;
    this->vertexDataHost = new float[numVertices*stride];
    this->indexDataHost  = new unsigned int[numVertices];

    // initialize default indexing
    for (int i = 0; i < maxNumElements; i++) {
        this->indexDataHost[i] = i;
    }
    unsigned int *indexPointer = (unsigned int*)lockIndex();
    cudaMemcpyAsync(indexPointer,indexDataHost,sizeof(int)*maxNumElements,cudaMemcpyHostToDevice,cudaStream);
    unlockIndex();
    refCount++;
}

VertexBuffer2::VertexBuffer2() {
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
    m_renderableBufferFlag = true;
    sprintf(&name[0],"default vbuffer name");
    this->stride = VERTEXBUFFER_STRIDE;
}

void VertexBuffer2::release() {
  //  printf("releasing %s\n",name); fflush(stdin); fflush(stdout);
    unlock();
    unlockIndex();
    if (vbo != -1) {
       //printf("releasing vbo (%s)\n",name); fflush(stdin); fflush(stdout);
       checkCudaErrors(cudaGLUnregisterBufferObject(vbo));
        glDeleteBuffers(1, &vbo);
        vbo = -1;
    }
    if (ibo != -1) {
        //printf("releasing ibo (%s)\n",name); fflush(stdin); fflush(stdout);
        checkCudaErrors(cudaGLUnregisterBufferObject(ibo));
        glDeleteBuffers(1, &ibo);
        ibo = -1;
    }
    if (vertexDataHost != NULL) delete[] vertexDataHost; vertexDataHost = NULL;
    if (indexDataHost != NULL) delete[] indexDataHost; indexDataHost = NULL;
    refCount--;
    if (refCount == 0 && m_renderableBufferFlag) {
        for (int i = 0; i < numNormalIbos; i++) {
            if (normalIbo[i] != -1) {
                glDeleteBuffers(1, &normalIbo[i]);
                normalIbo[i] = -1;
            }
        }
    }
}

VertexBuffer2::~VertexBuffer2()
{

}

void *VertexBuffer2::lock() {
	// already locked?
	if (devPtr != NULL) return devPtr; 
	// map the buffer to CUDA
    if (m_renderableBufferFlag) {
        checkCudaErrors(cudaGLMapBufferObjectAsync((void**)&devPtr,vbo,cudaStream));
    } else {
        devPtr = NULL;
    }
	return devPtr;
}

void VertexBuffer2::unlock() {
   // printf("unlocking %s\n",name);
    if (devPtr != NULL && m_renderableBufferFlag) { checkCudaErrors(cudaGLUnmapBufferObjectAsync(vbo,cudaStream)); devPtr = NULL; }
}

void *VertexBuffer2::lockIndex() {
        // already locked?
        if (indexDevPtr != NULL) return indexDevPtr;
        // map the buffer to CUDA
        if (m_renderableBufferFlag) {
            checkCudaErrors(cudaGLMapBufferObjectAsync((void**)&indexDevPtr,ibo,cudaStream));
        } else {
            indexDevPtr = NULL;
        }
        return indexDevPtr;
}

void VertexBuffer2::unlockIndex() {
        if (indexDevPtr != NULL && m_renderableBufferFlag) { 	checkCudaErrors(cudaGLUnmapBufferObjectAsync(ibo,cudaStream)); indexDevPtr = NULL; }
}


void VertexBuffer2::setStream( cudaStream_t stream )
{
	this->cudaStream = stream;
}

void VertexBuffer2::addVertex(float x, float y, float z, float r, float g, float b) 
{
    if (numVertices >= maxNumVertices-1 || !(stride == VERTEXBUFFER_STRIDE || stride == BASEBUFFER_STRIDE)) return;
    vertexDataHost[numVertices*stride+0] = x;
    vertexDataHost[numVertices*stride+1] = y;
    vertexDataHost[numVertices*stride+2] = z;

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }
    vertexDataHost[numVertices*stride+rgbOffset+0] = r;
    vertexDataHost[numVertices*stride+rgbOffset+1] = g;
    vertexDataHost[numVertices*stride+rgbOffset+2] = b;
    indexDataHost[numElements] = numElements;
    numVertices++; numElements++;
}

void VertexBuffer2::addLine(float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b) {
    if (numVertices >= maxNumVertices-1 || !(stride == VERTEXBUFFER_STRIDE || stride == BASEBUFFER_STRIDE)) return;
    vertexDataHost[numVertices*stride+0] = x1;
    vertexDataHost[numVertices*stride+1] = y1;
    vertexDataHost[numVertices*stride+2] = z1;

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }

    vertexDataHost[numVertices*stride+rgbOffset+0] = r;
    vertexDataHost[numVertices*stride+rgbOffset+1] = g;
    vertexDataHost[numVertices*stride+rgbOffset+2] = b;
    numVertices++;

    vertexDataHost[numVertices*stride+0] = x2;
    vertexDataHost[numVertices*stride+1] = y2;
    vertexDataHost[numVertices*stride+2] = z2;

    vertexDataHost[numVertices*stride+rgbOffset+0] = r;
    vertexDataHost[numVertices*stride+rgbOffset+1] = g;
    vertexDataHost[numVertices*stride+rgbOffset+2] = b;
    numVertices++;
}

void VertexBuffer2::addVertexWithoutElement(float x, float y, float z, float r, float g, float b)
{
    if (numVertices >= maxNumVertices-1 || !(stride == VERTEXBUFFER_STRIDE || stride == BASEBUFFER_STRIDE)) return;
    vertexDataHost[numVertices*stride+0] = x;
    vertexDataHost[numVertices*stride+1] = y;
    vertexDataHost[numVertices*stride+2] = z;
    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }
    vertexDataHost[numVertices*stride+rgbOffset+0] = r;
    vertexDataHost[numVertices*stride+rgbOffset+1] = g;
    vertexDataHost[numVertices*stride+rgbOffset+2] = b;
    numVertices++;
}

void VertexBuffer2::upload() {
    float *devicePointer = (float*)lock();
    cudaMemcpyAsync(devicePointer,vertexDataHost,sizeof(float)*stride*numVertices,cudaMemcpyHostToDevice,cudaStream);
    unlock();

    unsigned int *indexPointer = (unsigned int*)lockIndex();
    cudaMemcpyAsync(indexPointer,indexDataHost,sizeof(int)*numElements,cudaMemcpyHostToDevice,cudaStream);
    unlockIndex();
}


void VertexBuffer2::copyTo(VertexBuffer2 &vbuf) {
    float *vdataSrc = (float*)this->devPtr;
    if (vdataSrc == NULL) return;
    int *idataSrc = (int*)this->indexDevPtr;
    if (idataSrc == NULL) return;

    float *vdataDst = (float*)vbuf.devPtr;
    if (vdataDst == NULL) return;
    int *idataDst = (int*)vbuf.indexDevPtr;
    if (idataDst == NULL) return;

    cudaMemcpyAsync(idataDst,idataSrc,sizeof(int)*numElements,cudaMemcpyDeviceToDevice,cudaStream);
    cudaMemcpyAsync(vdataDst,vdataSrc,sizeof(float)*stride*numVertices,cudaMemcpyDeviceToDevice,cudaStream);

    vbuf.setElementsCount(numElements);
    vbuf.setVertexAmount(numVertices);
}

void VertexBuffer2::render() {
    if (!m_renderableBufferFlag) return;

    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
//    glPushAttrib((GL));
//    glDisable(GL_LIGHTING);
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    } else if (stride == COMPRESSED_STRIDE) {
        rgbOffset = 6;
    }

    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_POINTS, numElements,GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void VertexBuffer2::renderColor(float r, float g, float b, float a) {
    if (!m_renderableBufferFlag) return;

    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
//    glPushAttrib((GL));
//    glDisable(GL_LIGHTING);
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    } else if (stride == COMPRESSED_STRIDE) {
        rgbOffset = 6;
    }

    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColor4f(r,g,b,a);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_POINTS, numElements,GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void VertexBuffer2::render(Shader *shader, float *lightPos) {
    if (!m_renderableBufferFlag || shader == NULL) return;
    glEnable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);

    if (stride != VERTEXBUFFER_STRIDE && stride != COMPRESSED_STRIDE) {
        printf("glsl rendering requires standard stride %d or %d\n",VERTEXBUFFER_STRIDE,COMPRESSED_STRIDE);
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
    if (stride == VERTEXBUFFER_STRIDE) ptrBase += 5*sizeof(float);
    else ptrBase += 3*sizeof(float);
    glVertexAttribPointer(colorIndex,  3, GL_FLOAT, GL_FALSE, stride*sizeof(float), ptrBase);
    shader->bind();
    // uniforms are set after bind:
    shader->setUniformVec4("globalLight",lightPos);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_POINTS, numElements,GL_UNSIGNED_INT, (GLvoid*) 0);

    glDisableVertexAttribArray(vertexIndex);
    glDisableVertexAttribArray(colorIndex);
    glDisableVertexAttribArray(normalIndex);

    shader->unbind();
    glDisable(GL_LIGHTING);
}


void VertexBuffer2::renderPackedLines(int iboNum) {
    if (!m_renderableBufferFlag) return;
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    float *ptrBase = 0;
    glColor4f(1,1,0,1);

    if (stride != COMPRESSED_STRIDE) {
        printf("packed line rendering requires standard stride %d\n",COMPRESSED_STRIDE);
        fflush(stdin); fflush(stdout);
        return;
    }

    // sanity check:
    if (iboNum >= numNormalIbos) iboNum = numNormalIbos-1;

    // compute max element index for sparsified normal clouds
    // for example:
    // how many elements can be selected between every 16th source elements, when numVertices==numElements can not be exceeded?
    int nElems = getNormalElementCount(numElements, iboNum);

    glVertexPointer(3,GL_FLOAT,3*sizeof(float),ptrBase);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, normalIbo[iboNum]);
    glDrawElements(GL_LINES, nElems, GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void VertexBuffer2::renderBaseBufferLines() {
    if (!m_renderableBufferFlag) return;
    if (stride != BASEBUFFER_STRIDE) return;

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 3;

    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);
    glDrawArrays(GL_LINES, 0, numVertices);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void VertexBuffer2::renderAll() {
    if (!m_renderableBufferFlag) return;
    if (stride != VERTEXBUFFER_STRIDE && stride != BASEBUFFER_STRIDE) return;

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }

    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);
    glDrawArrays(GL_POINTS,0,numVertices);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void VertexBuffer2::renderPointRange2d(int start, int end) {
    if (!m_renderableBufferFlag) return;
    if (stride != VERTEXBUFFER_STRIDE && stride != BASEBUFFER_STRIDE) return;

    if (numVertices < 1) return;
    if (end < start) end = start;

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );

    int rgbOffset = 0;
    if (stride == VERTEXBUFFER_STRIDE) {
        rgbOffset = 8;
    } else if (stride == BASEBUFFER_STRIDE) {
        rgbOffset = 3;
    }

    float *ptrBase = 0; ptrBase += start*stride;
    glVertexPointer(2,GL_FLOAT,stride*sizeof(float),ptrBase); ptrBase += rgbOffset;
    glColorPointer(3,GL_FLOAT,stride*sizeof(float),(GLvoid*)ptrBase);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawArrays(GL_POINTS,0,(end-start)+1);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}



void VertexBuffer2::setVertexAmount( int num )
{
	if (num > maxNumVertices) printf("vertex buffer exceeded!\n"); else numVertices = num;
}
