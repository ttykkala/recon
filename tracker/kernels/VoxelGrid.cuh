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

#include "particles_kernel.cuh"

extern "C"
{
    void initCuda(bool bUseGL);
    void setParameters(SimParams *hostParams);
    void sortParticles(float *sortKeys, uint *indices, uint numParticles);
    void projectVoxelGrid(float4 *voxelVert3D, float2 *voxelVert2D, uint nPoints);
    void reconstructCuda(cudaArray *tex, float *outData, int width, int height);
    void updateVoxelGrid(float2 *tsdf, uint gridResoX, uint gridResoY, uint gridResoZ, float *distMap);
    void updateVoxelGridXYZ(float2 *tsdf, uint gridResoX, uint gridResoY, uint gridResoZ, float4 *xyzMap);
    void rayCastVoxels(float2 *tsdfData, uint width, uint height, float4 *cudaRayCastImage, bool useCubicFilter, float4 *cudaRayCastImageXYZ,float4 *cudaSmoothRayCastImageXYZ);
    void resetVoxelGrid(float2 *tsdfData, uint nCubes, float v1, float v2);
    void resampleVoxelGrid(float2 *tsdfDataDst, float2 *tsdfDataSrc, uint gridResoX, uint gridResoY, uint gridResoZ);
 }

