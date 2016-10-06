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

#include <opencv2/opencv.hpp>

// forward declare cuda stream type ( no need for headers)
struct CUstream_st;
typedef CUstream_st *cudaStream_t;
class Shader;

#define TRIBUFFER_STRIDE 9

class TriangleBuffer2 {
private:
    unsigned int vbo;
    unsigned int ibo;
    unsigned int maxNumVertices;
    unsigned int numVertices;
    unsigned int maxNumElements;
    unsigned int numElements;
    float        *vertexDataHost;
    unsigned int *indexDataHost;
    float *faceNormals;
    float *vertexNormals;
    int stride;
    void generateNormals(float *points, int numTriangles, unsigned int *vertexIndex3);
public:
    char name[512];
    void *devPtr;
    void *indexDevPtr;
    cudaStream_t cudaStream;
    TriangleBuffer2();
    TriangleBuffer2(float *points, int numTriangles, unsigned int *vertexIndices3, const char *vbufferName=NULL);
    void init(float *points, int numTriangles, unsigned int *vertexIndices3, const char *vbufferName=NULL); // for late init
    void update(cv::Mat *xyzImage, int xyzStride);
    void release();
    ~TriangleBuffer2();
    void *lock();
    void unlock();
    void *lockIndex();
    void unlockIndex();
    void setStream(cudaStream_t stream);
    void addVertex(float x, float y, float z, float r, float g, float b);
    void upload();
    void render();
    void render(Shader *shader, float *lightPos);
    void renderAll();
    void setVertexAmount(int num);
    unsigned int getVertexCount() { return numVertices; }
    unsigned int getElementsCount() { return numElements; }
    unsigned int getMaxVertexCount() { return maxNumVertices; }
    unsigned int getMaxElementsCount() { return maxNumElements; }
    // this is for indicating how many valid indices exist in ibo
    void setElementsCount(int newCount) { numElements = newCount; }
    unsigned int *getIndexBufferCPU() { return indexDataHost; }
    int getStride() { return stride; }
};
