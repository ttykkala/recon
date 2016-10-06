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

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <calib/calib.h>
#include <vector>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <PageLockedMemory.h>
#include <inttypes.h>

class BBOX {
public:
    BBOX(float *nodeCorner, float *nodeDim, const char *path);
    ~BBOX();
    float dim[3];
    float corner[3];
    float massCenter[3];
    float priorityValue;
    float colorR,colorG,colorB;
    unsigned int vbo;
    unsigned int pointCount;
    float *vertexBufferHost;
	char boxpath[512];
    //bool loadedToGPU;
    //bool prevLoadedToGPU;
    int  boxIndexInCache;
    bool boxAppears;
    bool boxDisappears;
    bool onGPU;
    bool inRAM;
    bool boxUpdatePossible;
    float *devPtr;
    int loadVertexData();
    int savePackedVertexData();
	unsigned int enumVertices();
	int allocateAndUpdateVbo();
    void releaseVbo();
    void release();
};

class LidarNode {
  public:
    LidarNode();
    ~LidarNode();
    LidarNode *findNode(unsigned int searchID);
    int behindCamera(Eigen::Vector3f &cpos, Eigen::Vector3f &cdir);
	//int inFrustum(Eigen::Matrix4f &cameraMtx, Eigen::Matrix3f &K, int texWidth, int texHeight, float *p2d);
    int inFrustum(float *planes, int numPlanes);
    //void minimizeAABB();
    void updateAABB();
    void release();
    unsigned int id;
    LidarNode *parent;
    LidarNode *children[8];
    float dim[3];
    float corner[3];
    BBOX *data;
    float boxCorners[9*3];
    float projectedCorners[9*2];
     void updateBoxCorners();
     void center(float *centerVec);
     float clamp(float val, float minval, float maxval);
private:
};

class LidarOctree {
private:
    void addToQueue(BBOX *bbox);
    void testAddBox(BBOX *bbox);
    void resetVariables();
    void uploadBoxesToGPU();
    int  loadSparseModel(const char *indexFilename);
    int  allocateSparseGPUPoints();
    int  allocateGPUPoints(unsigned int gpuPoints);
    int  tryToStoreGPU(BBOX *bbox);
    bool testOldbox(BBOX *b);
    int  allocateGPUCache(int maxPointsPerCube, int maxCubes);
    void transferPointsToGPU(float *devPtr,int cacheBoxIndex, float *hostData, int numPoints);
    void copyPointsToCache(float *devPtr, int cacheBoxIndex, float *hostData, float *boxDevPtr, int numPoints);
    void releaseGPUCache();
    void resetCache();
    int  allocateCacheSlot(BBOX *bbox, int &startingSlot);
    void updateCache();
    void storePriorityList();
    //int numPriorityQueueBoxes;
    int boxCacheMegaBytes;
    Eigen::Vector3f cameraDir;
    Eigen::Vector3f cameraPos;
    Eigen::Matrix4f pose;
    Eigen::Matrix4f cameraTransform;
    Eigen::Matrix3f K;
    float planes[4*4];
    int imageWidth,imageHeight;
    unsigned int vbo;
    unsigned int sparseVbo;
//	float *hostVertexMemory;
	PageLockedMemory hostVertexMemory;
    float *sparseVertexMemory;
    unsigned int sparsePoints;
	struct cudaGraphicsResource *cudaVbo;
    float *gpuPointsDev;
    uint64_t m_allocatedMemory;
    uint64_t m_reservedMemory;
    float *cachedVertexData;
    unsigned int maxPointsPerCube;
    unsigned int pointCount;
    unsigned int maxPointsOnGPU;
    unsigned int maxVisiblePointsOnGPU;
    unsigned int maxCubesInCache;
    uint64_t totalNumberOfPoints;
    BBOX **cacheAllocated;
    bool glInited;
	unsigned int loadBoxes(const char *filename);
    std::vector<BBOX*> prevPriorityQueueBoxes;
    unsigned int loadIndex(const char *filename);
    unsigned int loadDat(const char *filename);
public:	
    LidarNode *root;
    bool visible;
    int id;
    LidarOctree(const char *path, int id);
    ~LidarOctree();
    int storeSparseModel(unsigned int sizeMB, const char *fn);
    unsigned int getVbo();
    unsigned int getSparseVbo();
    unsigned int getPointCount();
    unsigned int getSparsePointCount();
	void release();	
    bool loadPose(char *posefile);
    void resetPose();
    void updateInternalCamera(float *planes4, Eigen::Matrix4f &cameraPose, float *Kext, int texWidth, int texHeight);
    void updatePointQueue(float *planes4,Eigen::Matrix4f &cameraPose, float *K, int texWidth, int texHeight);
    bool recurseVisibleNodes(LidarNode *node);
    bool isInited();
    void initGL(unsigned int boxCacheMB, unsigned int boxMB);
    std::vector<BBOX*> boxes;
	unsigned int vertexUploads;
    std::vector<BBOX*> priorityQueueBoxes;
};

