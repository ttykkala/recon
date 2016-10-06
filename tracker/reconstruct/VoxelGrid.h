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


#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>
#include "../kernels/particles_kernel.cuh"
#include "vector_functions.h"
#include "../rendering/GpuArray.h"
//#include "nvMath.h"
#include <Eigen/Geometry>

//using namespace nv;

class VoxelGrid
{
    public:
        VoxelGrid(uint gridResolution, uint winWidth, uint winHeight, bool openglContext, int deviceID, float minDist, float maxDist);
        ~VoxelGrid();

        void reset();
        void copyTo(VoxelGrid *target,int sliceIndex, int totalSlices);

        uint getGridResolution()
        {
            return m_params.gridResolution;
        }
        GpuArray<float2> &getTSDFArray() { return m_tsdf; }
        uint getCloudBuffer()
        {
            return m_distMap.getVbo();
        }
        Eigen::Vector3f getCubeOrigin() { Eigen::Vector3f origin; origin(0) =  m_params.cubeOrigin.x; origin(1) =  m_params.cubeOrigin.y; origin(2) =  m_params.cubeOrigin.z; return origin; }
        Eigen::Vector3f getCubeDim() {    Eigen::Vector3f    dim;    dim(0) =  m_params.cubeDim.x;    dim(1)    =  m_params.cubeDim.y;       dim(2) =  m_params.cubeDim.z; return dim; }
        SimParams &getParams()
        {
            return m_params;
        }
        Eigen::Vector3f getVoxelSize() { Eigen::Vector3f vsize; vsize(0) = m_params.voxelSize.x; vsize(1) = m_params.voxelSize.y; vsize(2) = m_params.voxelSize.z; return vsize; }
        void depthSort(const Eigen::Vector3f &cameraOrigin, const Eigen::Vector3f &cameraDir);
        void preprocessMeasurementRGBD(cudaArray *cudaArray, int width, int height);
        void update(float *Tcur, float *K);
        void rayCastImage(float *Tcur, float *K, bool useCubicFilter);
        float4 *getRayCastImage();
        float4 *getRayCastXYZImage();
        float4 *getRayCastNormalImage();
        int getDeviceID() { return mDeviceID; }
        void resetToRayCastResult(float4 *raycastXYZImage);
        void setCameraParams(float *Tcur, float *K);
        void resample(VoxelGrid *srcGrid);
        void clear();
protected: // methods
        VoxelGrid() {}
        void initialize(bool openglContext);
        void release();
        void updateProjection(int sliceIndex, int totalSlices);
        void rayCastVoxelImage();
    protected: // data
        bool m_bInitialized;
        GpuArray<float> m_distMap;
        GpuArray<float2> m_tsdf; // density and weight
        float4 *cudaRayCastImage;
        float4 *cudaRayCastXYZImage;
        float4 *cudaRayCastNormalImage;
        float4 *cudaRayCastSmoothXYZImage;
        // params
        SimParams m_params;

        bool m_doDepthSort;
        StopWatchInterface *m_timer;
        float m_time;
        int mDeviceID;
};

#endif // __PARTICLESYSTEM_H__
