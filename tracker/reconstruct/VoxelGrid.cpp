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


#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <reconstruct/VoxelGrid.h>
#include <kernels/VoxelGrid.cuh>
#include <kernels/particles_kernel.cuh>
#include <tracker/basic_math.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

VoxelGrid::VoxelGrid(uint gridResolution, uint winWidth, uint winHeight, bool openglContext, int deviceID, float minDist, float maxDist) :
    m_bInitialized(false),
    m_doDepthSort(true),
    m_timer(NULL),
    m_time(0.0f),
    cudaRayCastImage(0),
    cudaRayCastXYZImage(0),
    cudaRayCastSmoothXYZImage(0),
    mDeviceID(deviceID)
{
    m_params.maxDist = maxDist;
    m_params.minDist = minDist;
    m_params.winWidth = winWidth;
    m_params.winHeight = winHeight;
    m_params.cursorPos = make_float3(0.0f, 0.0f, 0.0f);
    m_params.cubeOrigin = make_float3(0,0.01,-2000.01f);
/*
    // freiburg2-2:
     m_params.cubeOrigin = make_float3(0, 500.01,-2300.01f);
     float gridSize = 1600.0f;
//     float gridSize = 5000.0f;
     m_params.cubeDim = make_float3(gridSize,gridSize,gridSize);
*/
/*
    // TTY-forum:
     m_params.cubeOrigin = make_float3(0, -100.01,-800.01f);
     m_params.cubeDim = make_float3(500.0f,500.0f,500.0f);
     */

     // kinect8 test:
    float dim = 1500.0f;
    m_params.cubeOrigin = make_float3(0, 0.01, -1600.01f);
    m_params.cubeDim = make_float3(dim,dim,dim);//700.0f,700.0f,700.0f);    
  /*  // kinect2 test:
    float dim = 1000.0f;
    m_params.cubeOrigin = make_float3(0, 0.01, -500.01f);
    m_params.cubeDim = make_float3(dim,dim,dim);//700.0f,700.0f,700.0f);
*/
  /*
    // kinect5 test:
    float dim = 900.0f;
    m_params.cubeOrigin = make_float3(400, 0.01,-1600.01f);
    m_params.cubeDim = make_float3(dim,dim,dim);//700.0f,700.0f,700.0f);
*/

/*
    // kinect2: 2
   float dim = 1000.0f;
    m_params.cubeOrigin = make_float3(0, 300.01,-1500.01f);
    m_params.cubeDim = make_float3(dim,dim,dim);//700.0f,700.0f,700.0f);
  */
    /*
    // versoteq:
     m_params.cubeOrigin = make_float3(0, -0.01,-1200.01f);
     m_params.cubeDim = make_float3(400.0f,400.0f,400.0f);
*/
    //    m_params.cubeDim = make_float3(2000.0f,2000.0f,2000.0f);
    m_params.gridResolution = gridResolution;
    m_params.voxelSize = make_float3(m_params.cubeDim.x/gridResolution,m_params.cubeDim.y/gridResolution,m_params.cubeDim.z/gridResolution);
    identity4x4(&m_params.T.m[0]); identity4x4(&m_params.iT.m[0]);
    identity3x3(&m_params.K.m[0]);
    initialize(openglContext);

}

VoxelGrid::~VoxelGrid()
{
    release();
}

void
VoxelGrid::initialize(bool openglContext)
{
    assert(!m_bInitialized);
    uint reso  = m_params.gridResolution;
    m_distMap.alloc(m_params.winWidth*m_params.winHeight, false, false);    // create as VBO
    m_tsdf.alloc(reso*reso*reso, false, false);
    int numTexels = m_params.winWidth*m_params.winHeight;
    int numValues = numTexels * 4;
    int sizeTexData = sizeof(GLfloat) * numValues;
    checkCudaErrors(cudaMalloc((void **)&cudaRayCastImage, sizeTexData));          checkCudaErrors(cudaMemset(cudaRayCastImage,         0, sizeTexData));
    checkCudaErrors(cudaMalloc((void **)&cudaRayCastXYZImage, sizeTexData));       checkCudaErrors(cudaMemset(cudaRayCastXYZImage,      0, sizeTexData));
    checkCudaErrors(cudaMalloc((void **)&cudaRayCastNormalImage, sizeTexData));    checkCudaErrors(cudaMemset(cudaRayCastNormalImage,   0, sizeTexData));
    checkCudaErrors(cudaMalloc((void **)&cudaRayCastSmoothXYZImage, sizeTexData)); checkCudaErrors(cudaMemset(cudaRayCastSmoothXYZImage,0, sizeTexData));

    sdkCreateTimer(&m_timer);
    setParameters(&m_params);
    //identityIndex();
    m_bInitialized = true;
    reset();
}



void
VoxelGrid::release()
{
    assert(m_bInitialized);
    cudaFree(cudaRayCastImage);
    cudaFree(cudaRayCastXYZImage);
    cudaFree(cudaRayCastNormalImage);
    cudaFree(cudaRayCastSmoothXYZImage);
    m_distMap.free();
    m_tsdf.free();
    printf("releasing voxel grid.\n");
}

// random float [0, 1]
inline float frand()
{
    return rand() / (float) RAND_MAX;
}

// signed random float [-1, 1]
inline float sfrand()
{
    return frand()*2.0f-1.0f;
}

// random signed vector
inline Eigen::Vector3f svrand()
{
    return Eigen::Vector3f(sfrand(), sfrand(), sfrand());
}


// random point in circle
inline Eigen::Vector2f randCircle()
{
    Eigen::Vector2f r;

    do
    {
        r = Eigen::Vector2f(sfrand(), sfrand());
    }
    while (r.norm() > 1.0f);

    return r;
}

// random point in sphere
inline Eigen::Vector3f randSphere()
{
    Eigen::Vector3f r;

    do
    {
        r = svrand();
    }
    while (r.norm() > 1.0f);

    return r;
}
// depth sort the particles
void
VoxelGrid::depthSort(const Eigen::Vector3f &cameraOrigin, const Eigen::Vector3f &cameraDir)
{
   /* if (!m_doDepthSort)
    {
        return;
    }
    // R' -R'c
    float3 sortVector;
    sortVector.x = cameraDir.x;
    sortVector.y = cameraDir.y;
    sortVector.z = cameraDir.z;
    float3 origin;
    origin.x = cameraOrigin.x;
    origin.y = cameraOrigin.y;
    origin.z = cameraOrigin.z;

    m_pos.map();
    m_indices.map();
    uint nElements = m_params.gridResolution*m_params.gridResolution*m_params.gridResolution;
    // calculate depth
    calcDepth(m_pos.getDevicePtr(), m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), sortVector, origin, nElements);

    // radix sort
    sortParticles(m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), nElements);

    m_pos.unmap();
    m_indices.unmap();*/
}

void VoxelGrid::reset()
{
    clear();
}


void writeDepth(const char *fn, int dimx, int dimy, float *src, float minZ, float maxZ) {
    float zcap = 50.0f;
    FILE *fp = fopen(fn, "wb"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    unsigned char *buf = new unsigned char[dimx*dimy*3];
    int off4 = 0;
    for (int j = 0; j < dimy; j++) {
        int off = (dimy-1-j)*dimx;
        int off3 = (dimy-1-j)*dimx*3;
        for (int i = 0; i < dimx; i++,off++,off3+=3,off4+=4) {
            float zval = 255*(fabs(src[off])-minZ)/(maxZ-minZ);
/*            if (zval < minZ) minZ = zval;
            if (zval > maxZ) maxZ = zval;
            if (zval < 0.0f) zval = 0;
            if (zval > zcap) zval = zcap;
            zval = 255*zval/zcap;*/
            buf[off3+0] = zval;//src[off4+0]*255.0f;
            buf[off3+1] = zval;//src[off4+1]*255.0f;
            buf[off3+2] = zval;//src[off4+2]*255.0f;
        }
    }
    printf("minZ: %f maxZ: %f\n",minZ,maxZ);
    fwrite(buf,dimx*dimy*3,1,fp);
    delete[] buf;
    fclose(fp);
}

void VoxelGrid::preprocessMeasurementRGBD(cudaArray *rgbdImage, int width, int height) {
    m_distMap.map();
    reconstructCuda(rgbdImage, m_distMap.getDevicePtr(), width, height);
/*
    static int cnt = 0;
    char buf[512]; sprintf(buf,"scratch/dist%04d.ppm",cnt++);
    float *dmem = new float[width*height];
    cudaMemcpy(dmem,m_distMap.getDevicePtr(),sizeof(float)*width*height,cudaMemcpyDeviceToHost);
    writeDepth(buf,width,height,dmem,m_params.minDist,m_params.maxDist);
    delete dmem;
*/
    m_distMap.unmap();
}

void VoxelGrid::setCameraParams(float *Tcur, float *K) {
    float iTcur[16]; invertRT4(Tcur,&iTcur[0]);
    // from world to camera:
    memcpy(&m_params.T.m[0],&iTcur[0],sizeof(float)*16);
    // from camera to world:
    memcpy(&m_params.iT.m[0],&Tcur[0],sizeof(float)*16);
    memcpy(&m_params.K.m[0],K,sizeof(float)*9);
    setParameters(&m_params);
}

void VoxelGrid::update(float *Tcur, float *K) {
    setCameraParams(Tcur,K);
    m_distMap.map(); m_tsdf.map();
    float2 *tsdf   = m_tsdf.getDevicePtr();
    float *distMap =  m_distMap.getDevicePtr();
    updateVoxelGrid(tsdf, m_params.gridResolution,m_params.gridResolution,m_params.gridResolution,distMap);
    m_tsdf.unmap(); m_distMap.unmap();
}

void VoxelGrid::rayCastImage(float *Tcur, float *K, bool useCubicFilter) {
    float iTcur[16]; invertRT4(Tcur,&iTcur[0]);
    // from world to camera:
    memcpy(&m_params.T.m[0],&iTcur[0],sizeof(float)*16);
    // from camera to world:
    memcpy(&m_params.iT.m[0],&Tcur[0],sizeof(float)*16);
    memcpy(&m_params.K.m[0],K,sizeof(float)*9);
    setParameters(&m_params);

    int nPoints = m_params.winWidth*m_params.winHeight;
    m_tsdf.map();
    float2 *tsdf = m_tsdf.getDevicePtr();
    rayCastVoxels(tsdf,m_params.winWidth,m_params.winHeight,cudaRayCastImage,useCubicFilter, cudaRayCastXYZImage,cudaRayCastNormalImage);
    m_tsdf.unmap();
}

void VoxelGrid::clear() {
/*    float2 *ptr = m_tsdf.getHostPtr();
    for (int i = 0; i < m_params.gridResolution*m_params.gridResolution*m_params.gridResolution; i++) { ptr[i].x = 1; ptr[i].y = 0; }
    m_tsdf.copy(GpuArray<float2>::HOST_TO_DEVICE);
*/
    int nPoints = m_params.gridResolution*m_params.gridResolution*m_params.gridResolution;
    m_tsdf.map();
    float2 *ptr = m_tsdf.getDevicePtr();
    resetVoxelGrid(ptr,nPoints,1,0);
    m_tsdf.unmap();
}

float4 *VoxelGrid::getRayCastImage() {
    return cudaRayCastImage;
}

float4 *VoxelGrid::getRayCastXYZImage() {
    return cudaRayCastXYZImage;
}

float4 *VoxelGrid::getRayCastNormalImage() {
    return cudaRayCastNormalImage;
}

void VoxelGrid::copyTo(VoxelGrid *target, int sliceIndex, int totalSlices) {
    GpuArray<float2> &targetArray = target->getTSDFArray();

    int nPoints = m_params.gridResolution*m_params.gridResolution*m_params.gridResolution;
    m_tsdf.map(); targetArray.map();
    int sliceSize = nPoints/totalSlices;
    int sliceSizeBytes = sliceSize*sizeof(float2);
    float2 *dst = targetArray.getDevicePtr();
    float2 *src = m_tsdf.getDevicePtr();
    cudaMemcpyPeerAsync(dst+sliceIndex*sliceSize,0,src+sliceIndex*sliceSize,1,sliceSizeBytes);
    m_tsdf.unmap(); targetArray.unmap();
}

void VoxelGrid::resetToRayCastResult(float4 *raycastXYZImage) {
    int nPoints = m_params.gridResolution*m_params.gridResolution*m_params.gridResolution;
    m_tsdf.map();
    float2 *ptr = m_tsdf.getDevicePtr();
    resetVoxelGrid(ptr,nPoints,0,0);
    updateVoxelGridXYZ(ptr, m_params.gridResolution,m_params.gridResolution, m_params.gridResolution, raycastXYZImage);
    m_tsdf.unmap();
}

void VoxelGrid::resample(VoxelGrid *srcGrid) {
    GpuArray<float2> &srcTsdf = srcGrid->getTSDFArray();
    m_tsdf.map(); srcTsdf.map();
    float2 *dst = m_tsdf.getDevicePtr();
    float2 *src = srcTsdf.getDevicePtr();
    resampleVoxelGrid(dst, src, m_params.gridResolution,m_params.gridResolution, m_params.gridResolution);
    srcTsdf.unmap(); m_tsdf.unmap();
}
