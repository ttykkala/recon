#include <fstream>  // std::ifstream
#include <iostream> // std::cout
#include <stdlib.h>
#include <LidarOctree.h>
#include <helper_cuda.h>
#include <basic_math.h>
#include <opencv2/opencv.hpp>
#include "eigenMath.h"
using namespace Eigen;
using namespace cv;
using namespace std;

void BBOX::release() {
	releaseVbo();
}

void BBOX::releaseVbo() {
    if (vbo > 0) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
}

BBOX::BBOX(float *nodeCorner, float *nodeDim, const char *path) {
    vertexBufferHost    = NULL;
    pointCount          = 0;
    vbo                 = 0;
    //loadedToGPU         = false;
    //prevLoadedToGPU     = false;
    boxIndexInCache     = 0;
    boxAppears          = false;
    boxDisappears       = false;
    onGPU               = false;
    inRAM               = false;
    boxUpdatePossible   = false;
    devPtr              = NULL;
    priorityValue  = 0;
    massCenter[0]  = 0;
    massCenter[1]  = 0;
    massCenter[2]  = 0;
    colorR = 1.0f;
    colorG = 0.0f;
    colorB = 0.0f;
	strncpy(this->boxpath, path, 512);
    this->dim[0] = nodeDim[0];
    this->dim[1] = nodeDim[1];
    this->dim[2] = nodeDim[2];
    this->corner[0] = nodeCorner[0];
    this->corner[1] = nodeCorner[1];
    this->corner[2] = nodeCorner[2];
}

BBOX::~BBOX() {
    release();
}

static int allocatedBytes = 0;

int BBOX::allocateAndUpdateVbo() {
	if (vertexBufferHost == NULL) return 0;
	int freeMemoryMB=0;
	//glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI,&freeMemoryMB); freeMemoryMB /= 1024;
	int requestedBytes = pointCount * sizeof(float) * 6;
	glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, requestedBytes, vertexBufferHost, GL_STATIC_DRAW);//GL_STREAM_COPY);
	GLenum errorCode = glGetError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	if (vbo == 0 || (errorCode != GL_NO_ERROR)) {
        printf("vbo allocation failed! (requested for %dMB, vbos take: %dMB)\n", (requestedBytes)/(1024*1024),allocatedBytes/(1024*1024));
		if (errorCode == GL_OUT_OF_MEMORY) printf("out of memory!\n");
		fflush(stdout);
//		assert(vbo != 0);
		return 0;
	} else {
		allocatedBytes += requestedBytes;
		//printf("vbos allocate: %d\n",allocatedBytes/(1024*1024)); fflush(stdout);
	}
	return 1;
}

unsigned int BBOX::enumVertices() {
	FILE *f = fopen(boxpath, "rb");
	if (f == NULL) { printf("%s not found!\n", boxpath); fflush(stdout); return 0; }
	fread(&this->pointCount, sizeof(unsigned int), 1, f);
	fclose(f);
	return this->pointCount;
}

int BBOX::savePackedVertexData() {
    FILE *f = fopen(boxpath,"wb");
    if (f == NULL) { printf("%s not found!\n",boxpath); fflush(stdout); return 0;}

    float *newData = new float[pointCount*6];
    unsigned int newPointCount = 0;
    for (unsigned int vindex = 0; vindex < pointCount; vindex++) {
        float sum = 0.0f;
        for (int i = 0; i < 6; i++) {
            sum += fabs(vertexBufferHost[vindex*6+i]);
        }
        if (sum != 0.0f) {
            newData[newPointCount*6+0] = vertexBufferHost[vindex*6+0];
            newData[newPointCount*6+1] = vertexBufferHost[vindex*6+1];
            newData[newPointCount*6+2] = vertexBufferHost[vindex*6+2];
            newData[newPointCount*6+3] = vertexBufferHost[vindex*6+3];
            newData[newPointCount*6+4] = vertexBufferHost[vindex*6+4];
            newData[newPointCount*6+5] = vertexBufferHost[vindex*6+5];
            newPointCount++;
        }
    }
    fwrite(&newPointCount,sizeof(unsigned int),1,f);
    if (newPointCount > 0) {
        fwrite(newData,1,sizeof(float)*newPointCount*6,f);
    }
    fclose(f);
    delete[] newData;
}


int BBOX::loadVertexData() {
	//printf("loading box data: %s\n",filename); fflush(stdout);
    FILE *f = fopen(boxpath,"rb");
	if (f == NULL) { printf("%s not found!\n",boxpath); fflush(stdout); return 0;}
    unsigned int prevPtCount = this->pointCount;
    fread(&this->pointCount,sizeof(unsigned int),1,f);
    if (this->pointCount == 0) { fclose(f); return 0; }
    fread(this->vertexBufferHost,1,sizeof(float)*pointCount*6,f);
    fclose(f);

    if (prevPtCount != this->pointCount) {
        printf("BBOX::loadVertexData(): ERROR! point count has changed!\n"); fflush(stdout);
        this->pointCount = MIN(this->pointCount,prevPtCount);
    }

	//printf("loaded  %u points\n",this->pointCount); fflush(stdout);
    //extents[0] = 0; extents[1] = 0; extents[2] = 0;
    massCenter[0]  = 0; massCenter[1]  = 0; massCenter[2]  = 0;
    corner[0] = corner[1] = corner[2] = FLT_MAX;
    float maxpoint[3]; maxpoint[0] = maxpoint[1] = maxpoint[2] = -FLT_MAX;
    for (unsigned int off = 0; off < pointCount; off++) {
        float x = vertexBufferHost[off*6+0];
        float y = vertexBufferHost[off*6+1];
        float z = vertexBufferHost[off*6+2];
        massCenter[0]+=x;
        massCenter[1]+=y;
        massCenter[2]+=z;
        if (x < corner[0]) corner[0] = x;
        if (y < corner[1]) corner[1] = y;
        if (z < corner[2]) corner[2] = z;
        if (x > maxpoint[0]) maxpoint[0] = x;
        if (y > maxpoint[1]) maxpoint[1] = y;
        if (z > maxpoint[2]) maxpoint[2] = z;
    }
    massCenter[0]/=pointCount;
    massCenter[1]/=pointCount;
    massCenter[2]/=pointCount;

    dim[0] = maxpoint[0]-corner[0];
    dim[1] = maxpoint[1]-corner[1];
    dim[2] = maxpoint[2]-corner[2];

    inRAM = true;
    boxUpdatePossible=true;    
    /*if (!allocateAndUpdateVbo()) {
		return 0;
    }*/
    //printf("point count: %d\n",pointCount);
    return 1;
}

LidarNode::LidarNode() {
    id = 0;
    parent = NULL;
    for (int i = 0; i < 8; i++) children[i] = NULL;
    data = NULL;
}

LidarNode *LidarNode::findNode(unsigned int searchID) {
    if (this->id == searchID) return this;

    for (int i = 0; i < 8; i++) {
        if (this->children[i] != NULL) {
            LidarNode *foundNode = this->children[i]->findNode(searchID);
            if (foundNode) return foundNode;
        }
    }
    return NULL;
}
/*
void LidarNode::minimizeAABB() {
    if (data != NULL) {
        for (int j = 0; j < 3; j++) {
            corner[j] = data->corner[j];
            dim[j]    = data->dim[j];
        }
        return;
    }

    float childMin[3] = {FLT_MAX,FLT_MAX,FLT_MAX};
    float childMax[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX};

    for (int i = 0; i < 8; i++) {
        if (this->children[i] != NULL) {
            float minp[3],maxp[3];
            this->children[i]->minimizeAABB();
            for (int j = 0; j < 3; j++) {
                minp[j] = this->children[i]->corner[j];
                maxp[j] = minp[j]+this->children[i]->dim[j];
                if (minp[j] < childMin[j]) childMin[j] = minp[j];
                if (maxp[j] > childMax[j]) childMax[j] = maxp[j];
            }
        }
    }
    for (int j = 0; j < 3; j++) {
        corner[j] = childMin[j];
        dim[j]   = childMax[j]-childMin[j];
    }
}
*/
void LidarNode::updateAABB() {
    if (data != NULL) {
        for (int j = 0; j < 3; j++) {
            corner[j] = data->corner[j];
            dim[j]    = data->dim[j];
        }
        updateBoxCorners();
        return;
    }

    float childMin[3] = {FLT_MAX,FLT_MAX,FLT_MAX};
    float childMax[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int i = 0; i < 8; i++) {
        if (this->children[i] != NULL) {
            float minp[3],maxp[3];
            for (int j = 0; j < 3; j++) {
                minp[j] = this->children[i]->corner[j];
                maxp[j] = minp[j]+this->children[i]->dim[j];
                if (minp[j] < childMin[j]) childMin[j] = minp[j];
                if (maxp[j] > childMax[j]) childMax[j] = maxp[j];
            }
        }
    }
    for (int j = 0; j < 3; j++) {
        corner[j] = childMin[j];
        dim[j]   = childMax[j]-childMin[j];
    }
    updateBoxCorners();
}

LidarNode::~LidarNode() {

}

void LidarNode::release() {
    for (int i = 0; i < 8; i++) {
        if (this->children[i] != NULL) {
            this->children[i]->release();
            delete this->children[i];
            this->children[i] = NULL;
        }
    }
}

void LidarNode::updateBoxCorners() {
    int i = 0;
    boxCorners[i*3+0] = corner[0];
    boxCorners[i*3+1] = corner[1];
    boxCorners[i*3+2] = corner[2];
    i++;

    boxCorners[i*3+0] = corner[0]+dim[0];
    boxCorners[i*3+1] = corner[1];
    boxCorners[i*3+2] = corner[2];
    i++;

    boxCorners[i*3+0] = corner[0]+dim[0];
    boxCorners[i*3+1] = corner[1];
    boxCorners[i*3+2] = corner[2]+dim[2];
    i++;

    boxCorners[i*3+0] = corner[0];
    boxCorners[i*3+1] = corner[1];
    boxCorners[i*3+2] = corner[2]+dim[2];
    i++;

    boxCorners[i*3+0] = corner[0];
    boxCorners[i*3+1] = corner[1]+dim[1];
    boxCorners[i*3+2] = corner[2];
    i++;

    boxCorners[i*3+0] = corner[0]+dim[0];
    boxCorners[i*3+1] = corner[1]+dim[1];
    boxCorners[i*3+2] = corner[2];
    i++;

    boxCorners[i*3+0] = corner[0]+dim[0];
    boxCorners[i*3+1] = corner[1]+dim[1];
    boxCorners[i*3+2] = corner[2]+dim[2];
    i++;

    boxCorners[i*3+0] = corner[0];
    boxCorners[i*3+1] = corner[1]+dim[1];
    boxCorners[i*3+2] = corner[2]+dim[2];
    i++;

    center(&boxCorners[8*3+0]);
}

unsigned int LidarOctree::loadIndex(const char *filename) {
    FILE *f = fopen(filename,"rb");
    if (f == NULL) return 0;
    char buf0[512]; memset(buf0,0,512);
    char buf1[512]; memset(buf1,0,512);
    char buf2[512]; memset(buf2,0,512);
    char buf3[512]; memset(buf3,0,512);
    char buf4[512]; memset(buf4,0,512);
    char buf5[512]; memset(buf5,0,512);
    int maxPointsPerCube2 = 0;
    fscanf(f,"%s %s %s %d %s %s %s\n",&buf0[0],&buf1[0],&buf2[0],&maxPointsPerCube2,&buf3[0],&buf4[0],&buf5[0]);
    printf("buf: %s %s %s %d %s %s %s\n",buf0,buf1,buf2,maxPointsPerCube2,buf3,buf4,buf5);

    int nboxlimit = INT_MAX;//100;//INT_MAX;
    unsigned int maxPointsPerCube = 0;
    while (1) {
        LidarNode *node = new LidarNode();
        unsigned int parentID = 0;
        memset(&buf0[0],0,512); // clear file name buffer
        int ret = fscanf(f,"%u %u %f %f %f %f %f %f %s\n",&node->id, &parentID, &node->corner[0],&node->corner[1],&node->corner[2],&node->dim[0],&node->dim[1],&node->dim[2],&buf0[0]);
        //printf("%u %u %f %f %f %f %f %f %s\n",node->id, parentID, node->corner[0],node->corner[1],node->corner[2],node->dim[0],node->dim[1],node->dim[2],buf0); fflush(stdout);
        if (ret != 9) { delete node; break; }
        else {
            node->updateBoxCorners();
            if (root == NULL) { printf("setting root node!\n");  fflush(stdout); root = node; }
            else {
                //printf("finding parent for %u, parentid: %u\n",node->id,parentID); fflush(stdout);
                LidarNode *parent = root->findNode(parentID);
                if (parent != NULL) {
                    unsigned int childIndex = node->id - 8*parentID-1;
                    parent->children[childIndex] = node;
                    if (strcmp(buf0,"node")!=0) {
                        BBOX *box = new BBOX(node->corner, node->dim, buf0);
                        //printf("loading box for %u (child:%u)\n",node->id,childIndex); fflush(stdout);
                        if (box != NULL) {
                            if (box->enumVertices()) {
                                node->data = box;
                                this->totalNumberOfPoints += box->pointCount;
                                if (box->pointCount > maxPointsPerCube) maxPointsPerCube = box->pointCount;
                                boxes.push_back(box);
                                if (boxes.size() >= nboxlimit)
                                    goto quickexit;
                            }
                            else delete box;
                        }
                        else {
                            printf("box == NULL!\n"); fflush(stdout);
                        }

                    }
                } else {
                    printf("parent could not be found!\n"); fflush(stdout);
                    delete node;
                }
            }
        }
    }
quickexit:
    fclose(f);
    printf("octree created, max point count :%d (header limit: %d)\n", maxPointsPerCube,maxPointsPerCube2); fflush(stdout);
    return maxPointsPerCube;
}

unsigned int LidarOctree::loadDat(const char *filename) {
    FILE *f = fopen(filename,"rb");
    if (f == NULL) return 0;
    char buf0[512];
    unsigned int numBoxes = 0;
    fread(&numBoxes,sizeof(int),1,f);
    printf("numBoxes: %u\n",numBoxes);

    unsigned int maxPointsPerCube = 0;
    for (unsigned int i = 0; i < numBoxes; i++) {
        LidarNode *node = new LidarNode();
        unsigned int parentID = 0;
        unsigned int pointCount = 0;
        memset(&buf0[0],0,512); // clear file name buffer
        fread(&node->id,sizeof(int),1,f);
        fread(&parentID,sizeof(int),1,f);
        fread(&node->corner[0],sizeof(float),1,f);
        fread(&node->corner[1],sizeof(float),1,f);
        fread(&node->corner[2],sizeof(float),1,f);
        fread(&node->dim[0],sizeof(float),1,f);
        fread(&node->dim[1],sizeof(float),1,f);
        fread(&node->dim[2],sizeof(float),1,f);
        fread(&buf0[0],sizeof(char),512,f);
        fread(&pointCount,sizeof(int),1,f);
        //printf("%u %u %f %f %f %f %f %f %s\n",node->id, parentID, node->corner[0],node->corner[1],node->corner[2],node->dim[0],node->dim[1],node->dim[2],buf0); fflush(stdout);
        node->updateBoxCorners();
        if (root == NULL) { printf("setting root node!\n");  fflush(stdout); root = node; }
        else {
            LidarNode *parent = root->findNode(parentID);
            if (parent != NULL) {
                unsigned int childIndex = node->id - 8*parentID-1;
                parent->children[childIndex] = node;
                if (strcmp(buf0,"node")!=0) {
                    BBOX *box = new BBOX(node->corner, node->dim, buf0);
                    if (box != NULL) {
                        if (pointCount > 0) {
                            node->data = box;
                            box->pointCount = pointCount;
                            this->totalNumberOfPoints += pointCount;
                            if (pointCount > maxPointsPerCube) maxPointsPerCube = pointCount;
                            boxes.push_back(box);
                        }
                        else delete box;
                    }
                    else {
                        printf("box == NULL!\n"); fflush(stdout);
                    }
                }
            } else {
                printf("parent could not be found!\n"); fflush(stdout);
                delete node;
            }
        }
    }
    fclose(f);
    printf("octree created, max point count :%d (header limit: %d)\n", maxPointsPerCube,maxPointsPerCube); fflush(stdout);
    return maxPointsPerCube;
}


unsigned int LidarOctree::loadBoxes(const char *filename) {
    this->totalNumberOfPoints = 0;
    const char *dot = strrchr(filename,'.');
    if (dot == NULL) {
        printf("invalid filename %s\n",filename); return 0;
    }
    unsigned int maxPointsPerCube = 0;
    if (dot[1] == 'i') {
        maxPointsPerCube=loadIndex(filename);
    } else if (dot[1] == 'd') {
        maxPointsPerCube=loadDat(filename);
    } else {
        printf("invalid filename %s\n",filename);
        return 0;
    }

	//hostVertexMemory = new float[this->totalNumberOfPoints*6];
	hostVertexMemory.alloc(this->totalNumberOfPoints * 6 * sizeof(float));
	if (hostVertexMemory.ptr() == NULL) { printf("unable to allocate %lluMB for host vertex memory\n", (this->totalNumberOfPoints * 6 * sizeof(float)) / (1024 * 1024)); fflush(stdout); return 0; }

    uint64_t pointOffset = 0;
	for (size_t bi = 0; bi < boxes.size(); bi++) {
		BBOX *b = boxes[bi];
        b->vertexBufferHost = (float*)hostVertexMemory.ptr(pointOffset*6*sizeof(float));
		pointOffset += b->pointCount;
	}
    return maxPointsPerCube;
}

void LidarOctree::resetVariables() {
    visible             = true;
    resetPose();
}

int LidarOctree::allocateGPUCache(int maxPointsPerCube, int maxCubes) {
    unsigned int maxPoints = maxPointsPerCube*maxCubes;
    unsigned int requestedBytes = maxPoints * sizeof(float) * 6;
	printf("requesting %u bytes\n", requestedBytes); fflush(stdout);
#if defined(ENABLE_CUDA)
    printf("allocateGPUCache: CUDA is present, cache is pinned host memory!\n"); fflush(stdout);
	// allocate host cache as pinned memory (pagelocking enabled)
	cudaError_t status = cudaHostAlloc((void**)&cachedVertexData, requestedBytes, cudaHostAllocDefault);// cudaHostAllocMapped);
	if (status != cudaSuccess) { printf("Error allocating pinned host memory\n"); return 0; }
/*	float *devPtr = NULL;
	checkCudaErrors(cudaHostGetDevicePointer(&devPtr, cachedVertexData, 0));
	//kernel << <1, ns >> >(dev_ptr_p);
	checkCudaErrors(cudaDeviceSynchronize());*/
#else
	cachedVertexData = new float[maxPoints * 6]; 
#endif
	memset(cachedVertexData, 0, requestedBytes);

    glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, requestedBytes, cachedVertexData, GL_STREAM_DRAW);//GL_STREAM_COPY
	GLenum errorCode = glGetError();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (errorCode != GL_NO_ERROR) {
        printf("vbo allocation failed! (requested for %dMB, vbos take: %dMB)\n", (requestedBytes)/(1024*1024),allocatedBytes/(1024*1024));
        if (errorCode == GL_OUT_OF_MEMORY) printf("out of memory!\n");
        fflush(stdout);
        return 0;
    }
#if defined(ENABLE_CUDA)
    // NOTE: "cudaGraphicsRegisterFlagsNone" will double GPU memory consumption ;)
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaVbo, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
#endif
    return 1;
}

void LidarOctree::copyPointsToCache(float *devPtr, int cacheBoxIndex, float *hostData, float *boxDevPtr, int numPoints) {
    if (cacheBoxIndex < 0 || cacheBoxIndex >= int(maxCubesInCache)) return;
    int offsetA     = cacheBoxIndex*maxPointsPerCube*6;
    int offsetB     = cacheBoxIndex*maxPointsPerCube*6+numPoints*6;
    int zeroBytes   = (maxPointsPerCube-numPoints)*6*sizeof(float); if (zeroBytes < 0) zeroBytes = 0;
    unsigned int dataBytes = numPoints*6*sizeof(float);
    memcpy(&cachedVertexData[offsetA],hostData,dataBytes);
    if (zeroBytes) { memset(&cachedVertexData[offsetB],0,zeroBytes); }
#if defined(ENABLE_CUDA)
     cudaMemcpyAsync(devPtr+offsetA,boxDevPtr,dataBytes,cudaMemcpyDeviceToDevice,0);
     if (zeroBytes) {
        cudaMemsetAsync(devPtr+offsetB,0,zeroBytes,0);
     }
#endif
}

//todo: make sure hostData is also pagelocked memory!
void LidarOctree::transferPointsToGPU(float *devPtr, int cacheBoxIndex, float *hostData, int numPoints) {
    if (cacheBoxIndex < 0 || cacheBoxIndex >= int(maxCubesInCache)) return;
    int offsetA     = cacheBoxIndex*maxPointsPerCube*6;
    int offsetB     = cacheBoxIndex*maxPointsPerCube*6+numPoints*6;
    int zeroBytes   = (maxPointsPerCube-numPoints)*6*sizeof(float); if (zeroBytes < 0) zeroBytes = 0;
    unsigned int dataBytes = numPoints*6*sizeof(float);
    memcpy(&cachedVertexData[offsetA],hostData,dataBytes);
	if (zeroBytes) { memset(&cachedVertexData[offsetB],0,zeroBytes); }
#if defined(ENABLE_CUDA)
     cudaMemcpyAsync(devPtr+offsetA,&cachedVertexData[offsetA],dataBytes+zeroBytes,cudaMemcpyHostToDevice,0);
#else
    glBufferSubData(GL_ARRAY_BUFFER, offsetA*sizeof(float), dataBytes+zeroBytes, &cachedVertexData[offsetA]);
#endif    
    //pointCount+=numPoints;
    /*float *mappedPtr = (float*)glMapBufferRange(GL_ARRAY_BUFFER, offsetA*sizeof(float), dataBytes+zeroBytes, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLIudaGraphicsIT_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
	memcpy(mappedPtr, &cachedVertexData[offsetA], dataBytes+zeroBytes);
	glFlushMappedBufferRange(GL_ARRAY_BUFFER, offsetA*sizeof(float), dataBytes+zeroBytes);
	GLboolean success = glUnmapBuffer(GL_ARRAY_BUFFER);
	*/
}


unsigned int LidarOctree::getVbo() {
    return vbo;
}

unsigned int LidarOctree::getSparseVbo() {
    return sparseVbo;
}


unsigned int LidarOctree::getPointCount() {
    return maxVisiblePointsOnGPU;
}

unsigned int LidarOctree::getSparsePointCount() {
    return sparsePoints;
}

bool LidarOctree::isInited() {
    return glInited;
}


int LidarOctree::loadSparseModel(const char *indexFilename) {
    // generate sparse model filename.ucp
    char buf[512]; memset(buf,0,512);
    int fnsize = strlen(indexFilename);
    int offset = 0;
    for (int i = 0; i < fnsize; i++) {
        buf[i] = indexFilename[i];
        if (indexFilename[i] == '.') {
            offset = i;
            break;
        }
    }
    if (offset == 0) return 0;
    offset++;
    buf[offset+0]='u';
    buf[offset+1]='c';
    buf[offset+2]='p';

    // load sparse vertex model
    FILE *f = fopen(buf,"rb");
    if (f == NULL) { printf("file %s not found!\n",buf); return 0;}
    unsigned int cnt=0;
    fread(&cnt,sizeof(unsigned int),1,f);
    if (cnt == 0) { fclose(f); return 0; }
    sparseVertexMemory = new float[cnt * 6];
    fread(sparseVertexMemory,1,sizeof(float)*cnt*6,f);
    fclose(f);
    return cnt;
}

LidarOctree::LidarOctree(const char *path, int id) {
    resetVariables();
    this->id   = id;
    this->root = NULL;
    this->cachedVertexData = NULL;
    this->pointCount = 0;
    this->maxPointsOnGPU = 0;
    this->gpuPointsDev = NULL;
    this->vbo = 0;
    this->sparseVbo = 0;
    this->sparseVertexMemory = NULL;
    this->m_allocatedMemory = 0;
    this->m_reservedMemory = 0;
	this->totalNumberOfPoints = 0;
	this->cudaVbo = NULL;
	this->vertexUploads = 0;
    this->glInited = false;
    this->cacheAllocated = NULL;
    this->sparsePoints = loadSparseModel(path);
	this->maxPointsPerCube = loadBoxes(path);
    if (maxPointsPerCube == 0) {
        fprintf(stderr,"octree loading failed due to loadBoxes(..) failure!\n"); fflush(stderr);
        return;
    }
    this->maxPointsOnGPU = 0;
    this->maxVisiblePointsOnGPU = 0;
    this->maxCubesInCache = 0;
    cacheAllocated = NULL;  
}
void LidarNode::center(float *centerVec) {
    centerVec[0] = corner[0]+dim[0]/2.0f;
    centerVec[1] = corner[1]+dim[1]/2.0f;
    centerVec[2] = corner[2]+dim[2]/2.0f;
}

float euclidianDistance2(float *v1, float *v2) {
    float diff[3];
    diff[0] = v1[0]-v2[0];
    diff[1] = v1[1]-v2[1];
    diff[2] = v1[2]-v2[2];
   return diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2];
}

float zDistance(float *massCenter, Eigen::Vector3f &cpos, Eigen::Vector3f &cdir) {
    float diff[3];
    float dist = FLT_MAX;
    for (int i = 0; i < 1; i++) {
        diff[0] = (massCenter[i*3+0]-cpos(0))*cdir(0);
        diff[1] = (massCenter[i*3+1]-cpos(1))*cdir(1);
        diff[2] = (massCenter[i*3+2]-cpos(2))*cdir(2);
        float dot = diff[0]+diff[1]+diff[2];
        if (dot < 0) {
            //float d = diff[0]*diff[0]+diff[1]*diff[1];
            float d = diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2];
            if (d < dist) dist = d;
        }
    }
    return dist;
}

int LidarOctree::allocateGPUPoints(unsigned int gpuPoints) {
    if (gpuPoints == 0) return 0;
#if defined(ENABLE_CUDA)
    m_allocatedMemory = gpuPoints*6;
    checkCudaErrors(cudaMalloc(&gpuPointsDev,m_allocatedMemory*sizeof(float)));
    return 1;
#endif
    return 0;
}


int LidarOctree::allocateSparseGPUPoints() {
    if (sparsePoints <= 0) return 0;
    // generate sparse vbo:
    int requestedBytes = sparsePoints * sizeof(float) * 6;
    glGenBuffers( 1, &sparseVbo);
    glBindBuffer( GL_ARRAY_BUFFER, sparseVbo);
    glBufferData(GL_ARRAY_BUFFER, requestedBytes, sparseVertexMemory, GL_STATIC_DRAW);
    GLenum errorCode = glGetError();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (vbo == 0 || (errorCode != GL_NO_ERROR)) {
        printf("sparse vbo allocation failed! (requested for %dMB, vbos take: %dMB)\n", (requestedBytes)/(1024*1024),allocatedBytes/(1024*1024));
        if (errorCode == GL_OUT_OF_MEMORY) printf("out of memory!\n");
        fflush(stdout);
        return 0;
    } else {
        allocatedBytes += requestedBytes;
    }
    return 1;
}

void LidarOctree::initGL(unsigned int boxCacheMB, unsigned int boxMB) {
    if (glInited) return;

#if defined(ENABLE_CUDA)
    this->maxPointsOnGPU = (boxMB*1024*1024)/(sizeof(float)*6);
#endif
    this->maxVisiblePointsOnGPU = (boxCacheMB*1024*1024)/(sizeof(float)*6);
    this->maxCubesInCache = int(maxVisiblePointsOnGPU / maxPointsPerCube);
    maxVisiblePointsOnGPU = this->maxCubesInCache*this->maxPointsPerCube;
    printf("max points in cache: %u, max points on gpu: %u, max cubes: %d, total points: %llu\n",this->maxVisiblePointsOnGPU,this->maxPointsOnGPU,this->maxCubesInCache,this->totalNumberOfPoints);
    cacheAllocated = new BBOX*[maxCubesInCache];
    priorityQueueBoxes.reserve(maxCubesInCache);
    prevPriorityQueueBoxes.reserve(maxCubesInCache);
    resetCache();

    allocateSparseGPUPoints();
    allocateGPUPoints(maxPointsOnGPU);
    allocateGPUCache(maxPointsPerCube,maxCubesInCache);
/*
#if defined(ENABLE_CUDA)
	printf("registering octree RAM as pinned memory!\n");
	// lock area pages (CUDA >= 4.0)
	__int64 requiredBytes = this->totalNumberOfPoints * 6 * sizeof(float);
	checkCudaErrors(cudaHostRegister(hostVertexMemory, requiredBytes, cudaHostRegisterPortable));
#endif
*/
    glInited = true;
}

void LidarOctree::resetCache() {
    for (unsigned int i = 0; i < maxCubesInCache; i++) {
        cacheAllocated[i] = NULL;
    }
}

void LidarOctree::addToQueue(BBOX *bbox) {
    // try to add box into the queue:
    for (size_t i = 0; i < priorityQueueBoxes.size(); i++) {
        BBOX *queuebox = priorityQueueBoxes[i];
        if (bbox->priorityValue > queuebox->priorityValue) {
            priorityQueueBoxes[i] = bbox;
            // rotate the remaining boxes and drop the last one:
            for (size_t j = i+1; j < priorityQueueBoxes.size(); j++) {
                BBOX *nextbox = priorityQueueBoxes[j];
                priorityQueueBoxes[j] = queuebox;
                queuebox = nextbox;
            }
            // if theres' room add new box to the end of the queue, otherwise drop it out:
            if (priorityQueueBoxes.size() < maxCubesInCache) {
                priorityQueueBoxes.push_back(queuebox);
            }
            return;
        }
    }
    // bbox priority too low, try to add box to the end of the priority queue:
    if (priorityQueueBoxes.size() < maxCubesInCache) {
        priorityQueueBoxes.push_back(bbox);
    }
}

float testInFront(float *point, Eigen::Vector3f &cpos, Eigen::Vector3f &cdir) {
    float dist = (point[0]-cpos(0))*cdir(0)+(point[1]-cpos(1))*cdir(1)+(point[2]-cpos(2))*cdir(2);
    return -dist;
}

float projectToZAxis(float *point, Eigen::Vector3f &cpos, Eigen::Vector3f &cdir) {	
	float z = (point[0] - cpos(0))*cdir(0) + (point[1] - cpos(1))*cdir(1) + (point[2] - cpos(2))*cdir(2);
	if (z > 0) z = 0.0f;
	return z;
}


int LidarNode::behindCamera(Eigen::Vector3f &cpos, Eigen::Vector3f &cdir) {
    //float minZ = FLT_MAX;
    bool visible = false;
//    float minDistance = FLT_MAX;
    for (int i = 0; i < 8; i++) {
        float dist = testInFront(&boxCorners[i * 3],cpos,cdir);
        if ( dist > 0) {
            visible=true;
            return 0;
        }
//        if (fabs(dist) < minDistance) minDistance = fabs(dist);
//        float z = -projectToZAxis(&boxCorners[i * 3], cpos, cdir);
//		if (z < minZ) minZ = z;
	}
//    if (visible && minDistance < 20.0f) return 0;
    return 1;
//	if (minZ > 0.0f && minZ > 15.0f) return 1;
    //if (minZ > 0.0f && minZ < FLT_MAX) return 1;
//    return 0;
}

float LidarNode::clamp(float val, float minval, float maxval) {
    if (val < minval) val = minval;
    if (val > maxval) val = maxval;
    return val;
}


/*
int LidarNode::inFrustum(Eigen::Matrix4f &cameraMtx, Eigen::Matrix3f &K, int texWidth, int texHeight, float *p2out) {
    int visible = 0;
    for (int i = 0; i < 9; i++) {
        Eigen::Vector4f point(boxCorners[i*3+0],boxCorners[i*3+1],boxCorners[i*3+2],1);
        Eigen::Vector4f cpt = cameraMtx * point;
        if (cpt(2) > 0.0f) continue;
        cpt(0) /= cpt(2); cpt(1) /= cpt(2);
        Eigen::Vector2f p2d;
        p2d(0) = cpt(0)*K(0,0)+K(0,2);
        p2d(1) = cpt(1)*K(1,1)+K(1,2);

        p2out[i*2+0] = clamp(p2d(0),0,texWidth-1);
        p2out[i*2+1] = clamp(p2d(1),0,texHeight-1);

        if (p2d(0) >= 0 && p2d(0) < texWidth && p2d(1) >= 0 && p2d(1) < texHeight ) {
            visible = 1;
        }
    }
    return visible;
}*/

int LidarNode::inFrustum(float *planes, int numPlanes) {
    float c[3],ext[3];
    center(&c[0]);
    ext[0] = dim[0]/2.0f;
    ext[1] = dim[1]/2.0f;
    ext[2] = dim[2]/2.0f;

    int visible = 1;
    for (int i = 0; i < numPlanes; i++) {
        float *p = &planes[i*4];
        float d = p[0]*c[0]+p[1]*c[1]+p[2]*c[2];
        float r = fabs(p[0])*ext[0]+fabs(p[1])*ext[1]+fabs(p[2])*ext[2];
//        float r = p[0]*ext[0]+p[1]*ext[1]+p[2]*ext[2];
        float d_p_r = d + r;
        float d_m_r = d - r;
        if(d_p_r < -p[3]) {
            visible=0;
        } else if (d_m_r < -p[3]) {
            //visible = INTERSECT;
        }
    }
    return visible;
}


int measureBoxSize(float *p2d, float minRadius, float *size) {
    double center[2] = {0,0};
    for (int i = 0; i < 9; i++) {
        center[0] += p2d[i*2+0];
        center[1] += p2d[i*2+1];
    }
    center[0] /= 9.0f;
    center[1] /= 9.0f;
    float maxRadius = 0.0f;
    for (int i = 0; i < 9; i++) {
        float radius = (p2d[i*2+0]-center[0])*(p2d[i*2+0]-center[0])+(p2d[i*2+1]-center[1])*(p2d[i*2+1]-center[1]);
        if (radius > maxRadius) maxRadius = radius;
    }
    if (maxRadius < minRadius) return 0;
    *size = maxRadius;
    return 1;
}

bool LidarOctree::recurseVisibleNodes(LidarNode *node) {
    if (node == NULL) return false;
    if (node->behindCamera(cameraPos,cameraDir)) {
        return false;
    }

    //if (!node->inFrustum(cameraTransform,K,imageWidth,imageHeight,&node->projectedCorners[0]))
    if (!node->inFrustum(planes,4)) {
        return false;
    }

    bool update = false;
    if (node->data != NULL) {
        BBOX *bbox = node->data;
        if (bbox->boxUpdatePossible) {
            node->updateAABB();
            update = true;
            bbox->boxUpdatePossible=false;
        }
//        float c[3];
//        node->center(&c[0]);
//        bbox->priorityValue = 1.0f/(zDistance(&c[0],cameraPos,cameraDir)+1.0f);        
//        bbox->priorityValue = 1.0f/(zDistance(bbox->massCenter,cameraPos,cameraDir)+1.0f);
        bbox->priorityValue = 1.0f/(zDistance(&node->boxCorners[8*3+0],cameraPos,cameraDir)+1.0f);
        addToQueue(bbox);
        return update;
    } else {
        bool update = false;
        for (int i = 0; i < 8; i++) {
            if (recurseVisibleNodes(node->children[i])) update = true;
        }
        if (update) node->updateAABB();
        return update;
    }
}


int LidarOctree::storeSparseModel(unsigned int sizeMB, const char *fn) {
    if (boxes.size() == 0) return 0;
    uint64_t modelPoints = 0;
    for (size_t i = 0; i < boxes.size(); i++) {
        BBOX *b = boxes[i];
        if (!b->inRAM) {
            b->loadVertexData();
        }
        modelPoints += uint64_t(b->pointCount);
    }
 //   printf("points: %llu\n",modelPoints);

    uint64_t outputPoints = (uint64_t(sizeMB)*1024*1024)/(sizeof(float)*6);
    float *data = new float[outputPoints*6];//pointsPerBox*6*boxes.size()];
    memset(data,0,sizeof(float)*outputPoints*6);
    unsigned int totalPoints = 0;
    for (size_t i = 0; i < boxes.size(); i++) {
        BBOX *b = boxes[i];
        uint64_t selectedPoints = uint64_t(double(outputPoints)*double(b->pointCount)/double(modelPoints));
        float skipper = float(b->pointCount)/float(selectedPoints);
        if (skipper < 1) skipper = 1;
        float boxPoints = float(b->pointCount);
        for (float p = 0; p < boxPoints; p+=skipper) {
            uint64_t vindex = (uint64_t)p;
            data[totalPoints*6+0] = b->vertexBufferHost[vindex*6+0];
            data[totalPoints*6+1] = b->vertexBufferHost[vindex*6+1];
            data[totalPoints*6+2] = b->vertexBufferHost[vindex*6+2];
            data[totalPoints*6+3] = b->vertexBufferHost[vindex*6+3];
            data[totalPoints*6+4] = b->vertexBufferHost[vindex*6+4];
            data[totalPoints*6+5] = b->vertexBufferHost[vindex*6+5];
            // point is moved into sparse model
            // reset this point from the original model
            // which marks it for cleanup
            b->vertexBufferHost[vindex*6+0] = 0;
            b->vertexBufferHost[vindex*6+1] = 0;
            b->vertexBufferHost[vindex*6+2] = 0;
            b->vertexBufferHost[vindex*6+3] = 0;
            b->vertexBufferHost[vindex*6+4] = 0;
            b->vertexBufferHost[vindex*6+5] = 0;
            totalPoints++;
            if (totalPoints >= outputPoints) {
                b->savePackedVertexData();
                goto write_file;
            }
        }
        b->savePackedVertexData();
    }
write_file:
    FILE *f = fopen(fn,"wb");
    if (f == NULL) goto exit;
    fwrite(&totalPoints,sizeof(int),1,f);
    fwrite(&data[0],sizeof(float),totalPoints*6,f);
    fclose(f);
exit:
    delete[] data;
    return 1;
}

int LidarOctree::tryToStoreGPU(BBOX *bbox) {
    if (bbox == NULL) return 0;
    if (bbox->onGPU) return 0;
#if defined(ENABLE_CUDA)
    uint64_t boxMemory = uint64_t(bbox->pointCount)*6;
    if (m_reservedMemory+boxMemory > m_allocatedMemory ) return 0;
    checkCudaErrors(cudaMemcpy(&gpuPointsDev[m_reservedMemory],bbox->vertexBufferHost,boxMemory*sizeof(float),cudaMemcpyHostToDevice));
    bbox->onGPU = true;
    bbox->devPtr = &gpuPointsDev[m_reservedMemory];
    m_reservedMemory += boxMemory;
#endif
    return 1;
}

void LidarOctree::uploadBoxesToGPU() {
    if (cachedVertexData == NULL) return;
    pointCount = 0;
	vertexUploads = 0;
    glBindBuffer( GL_ARRAY_BUFFER, vbo);

    float* devPtr=NULL;

#if defined(ENABLE_CUDA)
    cudaGraphicsMapResources(1,&cudaVbo,0);
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr,&numBytes,cudaVbo);
#endif
    unsigned int cntGPU = 0;
    for (size_t i = 0; i < maxCubesInCache; i++) {
        BBOX *bbox = cacheAllocated[i];
        if (bbox == NULL) continue;
        if (bbox->onGPU) cntGPU++;
        //bbox->colorR = 0.0f; bbox->colorG  = 1.0f;
        if (bbox->boxAppears) {
            //bbox->colorR = 1.0f; bbox->colorG  = 0.0f;
            if (!bbox->inRAM) {
                bbox->loadVertexData();
                bbox->boxUpdatePossible = true;
            }
            if (!bbox->onGPU) tryToStoreGPU(bbox);
			if (bbox->onGPU && bbox->devPtr != NULL) {
                copyPointsToCache(devPtr,bbox->boxIndexInCache, bbox->vertexBufferHost,bbox->devPtr, bbox->pointCount);
            } else {
                transferPointsToGPU(devPtr,bbox->boxIndexInCache,bbox->vertexBufferHost,bbox->pointCount);
                vertexUploads++;
            }
        }
        pointCount += maxPointsPerCube;
    }
//    printf("gpu count: %u/%u\n",cntGPU,maxCubesInCache);
#if defined(ENABLE_CUDA)        
    cudaStreamSynchronize(0);
    cudaGraphicsUnmapResources(1,&cudaVbo,0); //unmap buffer object
#endif
    glBindBuffer( GL_ARRAY_BUFFER, 0);
}

int LidarOctree::allocateCacheSlot(BBOX *bbox, int &startingSlot) {
    // find a free slot in cache:
    for (unsigned int i = startingSlot; i < maxCubesInCache; i++) {
        if (cacheAllocated[i] == NULL) {
            cacheAllocated[i] = bbox;
            bbox->boxIndexInCache = i;
            startingSlot = (i+1)%maxCubesInCache;
            return 1;
        }
    }
    // problem: no free slot found:
    return 0;
}

bool LidarOctree::testOldbox(BBOX *b) {
    for (size_t j = 0; j < prevPriorityQueueBoxes.size(); j++) {
        BBOX *bold = prevPriorityQueueBoxes[j];
        if (bold == b) {
            return true;
        }
    }
    return false;

}

void LidarOctree::updateCache() {
     // set previous visible box attributes to their initial values:
    for (size_t j = 0; j < prevPriorityQueueBoxes.size(); j++) {
        BBOX *b = prevPriorityQueueBoxes[j];
        b->boxAppears    = false;
        b->boxDisappears = true;
    }

    // see which of the current boxes are appearing:
    for (size_t i = 0; i < priorityQueueBoxes.size(); i++) {
        BBOX *b = priorityQueueBoxes[i];
        b->boxDisappears = false;        
        // roll bbox attributes in time in case it is a new one:
        if (!testOldbox(b)) {
            b->boxAppears = true;
        }
    }
    // release bboxes in cache which have disappeared:
    for (size_t i = 0; i < prevPriorityQueueBoxes.size(); i++) {
        BBOX *bbox = prevPriorityQueueBoxes[i];
        // if prev box not anymore on gpu
        if (bbox == NULL) {
            printf("wtf11\n"); fflush(stdout);
        } else if (bbox->boxDisappears) {
            // remove box from cache
            if (bbox->boxIndexInCache >= 0 && bbox->boxIndexInCache < int(maxCubesInCache)) {
                cacheAllocated[bbox->boxIndexInCache] = NULL;
            } else {
                printf("wtf!\n"); fflush(stdout);
            }
        }
    }
    // flag all new boxes in priority queue to be transferred to gpu:
    int startingSlot = 0;
    for (size_t i = 0; i < priorityQueueBoxes.size(); i++) {
        BBOX *bbox = priorityQueueBoxes[i];
        // a new box to cache?
        if (bbox->boxAppears) {
            // add box to cache
            if (!allocateCacheSlot(bbox,startingSlot)) {
                printf("impossible! cache slot not found!\n"); fflush(stdout);
            }
        }
    }
}

void LidarOctree::storePriorityList() {
    prevPriorityQueueBoxes.clear();
    for (size_t i = 0; i < priorityQueueBoxes.size(); i++) {
        prevPriorityQueueBoxes.push_back(priorityQueueBoxes[i]);
    }
}

void LidarOctree::updatePointQueue(float *planes4, Eigen::Matrix4f &cameraPose, float *K, int texWidth, int texHeight) {
    if (!glInited) {
        printf("lidaroctree: gl not inited!\n"); fflush(stdout);
        return;
    }
	updateInternalCamera(planes4,cameraPose,K,texWidth,texHeight);
    // store previous cache state:
    storePriorityList();
    // clear priority queue:
    priorityQueueBoxes.clear();
    // setup new priority queue:
    recurseVisibleNodes(root);
    // update cache based on priority queue:
    updateCache();
    // upload missing data to gpu:
    uploadBoxesToGPU();
}

LidarOctree::~LidarOctree() {
    release();
}

void LidarOctree::releaseGPUCache() {
#if defined(ENABLE_CUDA)
	printf("releaseGPUCache: CUDA is present, cache was pinned host memory!\n"); fflush(stdout);
	if (cachedVertexData != NULL) cudaFreeHost(cachedVertexData);
    if (gpuPointsDev != NULL) cudaFree(gpuPointsDev);
#else
	if (cachedVertexData != NULL) delete[] cachedVertexData;
#endif
    if (vbo > 0) {
#if defined(ENABLE_CUDA)
        checkCudaErrors(cudaGraphicsUnregisterResource(cudaVbo));
#endif
        glBindBuffer(GL_ARRAY_BUFFER,0);
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
}

void LidarOctree::release() {
    for (size_t i = 0; i < boxes.size(); i++) {
        BBOX *box = boxes[i];
        delete box;
    }
    boxes.clear();

/*	if (hostVertexMemory) {
#if defined(ENABLE_CUDA)
		if (isInited()) {
			// unlock area pages (CUDA >= 4.0)
			checkCudaErrors(cudaHostUnregister(hostVertexMemory));
			// free regular memory
//			checkCudaErrors(cudaFreeHost(hostVertexMemory));
		}
		delete[] hostVertexMemory;
#else
//#endif
	}*/
    if (sparseVertexMemory) delete[] sparseVertexMemory;
	if (hostVertexMemory.ptr()) { hostVertexMemory.release(); };
    if (cacheAllocated) { delete[] cacheAllocated; }
    if (root) { root->release(); delete root; root = NULL; }
    releaseGPUCache();
}

void LidarOctree::resetPose() {
    pose = Eigen::Matrix4f::Identity();
    cameraTransform = Eigen::Matrix4f::Identity();
    K = Eigen::Matrix3f::Identity();
    this->cameraPos = Eigen::Vector3f(0,0, 0);
    this->cameraDir = Eigen::Vector3f(0,0,-1);
    imageWidth  = 0;
    imageHeight = 0;
}

void LidarOctree::updateInternalCamera(float *planes4Ext, Eigen::Matrix4f &cameraPose, float *Kext, int texWidth, int texHeight) {
    cameraTransform = cameraPose;
    pose = cameraPose.inverse();
    cameraPos(0) = pose(0,3);
    cameraPos(1) = pose(1,3);
    cameraPos(2) = pose(2,3);
    cameraDir(0) = pose(0,2);
    cameraDir(1) = pose(1,2);
    cameraDir(2) = pose(2,2);
    K(0,0) = fabs(Kext[0]); K(0,1) = fabs(Kext[1]); K(0,2) = fabs(Kext[2]);
    K(1,0) = fabs(Kext[3]); K(1,1) = fabs(Kext[4]); K(1,2) = fabs(Kext[5]);
    K(2,0) = fabs(Kext[6]); K(2,1) = fabs(Kext[7]); K(2,2) = fabs(Kext[8]);
    imageWidth  = texWidth;
    imageHeight = texHeight;
    for (int i = 0; i < 16; i++) planes[i] = planes4Ext[i];
}
