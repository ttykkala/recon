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

#include "barrier_sync.h"
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)

typedef struct
{
    //Device ID for multi-GPU version
    int device;
    StopWatchInterface *timer;
    bool threadActive;

} TGpuTask;


extern "C"
{
    extern int numCudaDevices;
    extern cudaDeviceProp cudaDeviceProps[32];
    //OS thread ID
    extern TGpuTask *gpuTasks;
    extern bool g_killThreads;
    extern barrierSync *barrier;
    bool setupCudaDevices(int minimumDevices);
    void cleanupCudaDevices();
    bool setupPeerAccess(int srcDevice, int dstDevice);
    bool disablePeerAccess(int srcDevice, int dstDevice);
}

#define SAFE_RELEASE(x) { if (x!=NULL) delete x; x = NULL; }
#define SAFE_RELEASE_ARRAY(x) { if (x!=NULL) delete[] x; x = NULL; }


void sleepMs(int ms);
