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

class VertexBuffer2;

#define HISTOGRAM64_BIN_COUNT 64
typedef unsigned int uint;
typedef unsigned char uchar;

//typedef struct CUstream_st;
typedef CUstream_st *cudaStream_t;

////////////////////////////////////////////////////////////////////////////////
// GPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void initHistogram64(void);
extern "C" void closeHistogram64(void);

extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev,float *weightedResidual, cudaStream_t stream);

