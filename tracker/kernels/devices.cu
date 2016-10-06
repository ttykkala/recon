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

#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "devices.cuh"
#include "hostUtils.h"

namespace devutils {
    #include "kernelUtils.h"
}
using namespace devutils;



void sleepMs(int ms) {
    timespec delay = { 0, ms*1000000 };
    timespec remaining;
    nanosleep(&delay,&remaining);
 //   usleep(ms*1000); //convert to microseconds
    return;
}

extern "C"
{
    int numCudaDevices = 0;
    cudaDeviceProp cudaDeviceProps[32];
    barrierSync staticBarrier;
    barrierSync *barrier = NULL;
    bool g_killThreads = false;
    TGpuTask *gpuTasks = NULL;

    void printDevProp( cudaDeviceProp devProp )
    {
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %u\n",  (unsigned int)devProp.totalGlobalMem);
        printf("Total shared memory per block: %u\n",  (unsigned int)devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %u\n",  (unsigned int)devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  (unsigned int)devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  (unsigned int)devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multi-processors:    %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        printf("Can map host memory:           %s\n",  (devProp.canMapHostMemory ? "Yes" : "No"));
        return;
    }


    bool setupPeerAccess(int srcDevice, int dstDevice) {
        cudaDeviceProp &srcProps = cudaDeviceProps[srcDevice];
        cudaDeviceProp &dstProps = cudaDeviceProps[dstDevice];

        // Check that we got UVA on both devices
        printf("Checking GPU%d and GPU%d for UVA capabilities...\n", srcDevice, dstDevice);
        const bool has_uva = (srcProps.unifiedAddressing && dstProps.unifiedAddressing);

        printf("> %s (GPU%d) supports UVA: %s\n", srcProps.name, srcDevice, (srcProps.unifiedAddressing ? "Yes" : "No"));
        printf("> %s (GPU%d) supports UVA: %s\n", dstProps.name, dstDevice, (dstProps.unifiedAddressing ? "Yes" : "No"));

        if (has_uva)
        {
            printf("Both GPUs can support UVA, enabling...\n");
        }
        else
        {
            printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
            return false;
        }
        // Enable peer access
        printf("Enabling peer access between GPU%d and GPU%d...\n", srcDevice, dstDevice);
        checkCudaErrors(cudaDeviceEnablePeerAccess(dstDevice, 0));
        return true;
    }

    bool disablePeerAccess(int srcDevice, int dstDevice) {
        printf("Disabling peer access between GPU%d and GPU%d...\n", srcDevice, dstDevice); fflush(stdout);
         checkCudaErrors(cudaDeviceDisablePeerAccess(dstDevice));
         return true;
    }


    bool setupCudaDevices(int minimumDevices) {
        cudaGetDeviceCount(&numCudaDevices);
        if (numCudaDevices == 0) { printf("no cuda devices found!\n"); exit(0); }
        printf("There are %d CUDA devices.\n", numCudaDevices);

        if (numCudaDevices < minimumDevices) {
            printf("problem: %d CUDA devices are required for execution!\naborting...\n",minimumDevices); return false;
        }
        // cut the amount of cuda devices into two, if there are more than 2 GPUs available:
        numCudaDevices = 2;
        for (int i = 0; i < numCudaDevices; i++) {
            printf("\nCUDA GL Device #%d\n",i);
            cudaGetDeviceProperties(&cudaDeviceProps[i], 0);
            printDevProp(cudaDeviceProps[i]);
        }

        barrier = &staticBarrier;
        barrier->initialize(numCudaDevices);
        gpuTasks = new TGpuTask[numCudaDevices];

        for (int i = 0; i < numCudaDevices; i++) {
            sdkCreateTimer(&gpuTasks[i].timer);
            sdkResetTimer(&gpuTasks[i].timer);
            gpuTasks[i].device = i;
            gpuTasks[i].threadActive = false;
        }
        // initialize first GPU device to be CUDA interop device:
        checkCudaErrors(cudaGLSetGLDevice(0));
        for (int i = 1; i < numCudaDevices; i++) setupPeerAccess(0,i);
        gpuTasks[0].threadActive = true;
        return true;
    }

      void cleanupCudaDevices() {
          printf("releasing  cuda devices!\n"); fflush(stdout);
          if (barrier != NULL) {
              barrier->disable();
          }
          for (int i = 1; i < numCudaDevices; i++) {
              disablePeerAccess(0,i);
          }
          for (int i = 0; i < numCudaDevices; i++) {
              sdkDeleteTimer(&gpuTasks[i].timer);
          }
          if (gpuTasks) {
              delete[] gpuTasks; gpuTasks = NULL;
          }
      }

}   // extern "C"
