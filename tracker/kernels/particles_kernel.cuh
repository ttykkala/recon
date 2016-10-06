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
#ifdef USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;


typedef struct
{
    float m[16];
}  float4x4;

typedef struct
{
    float m[9];
}  float3x3;

struct SimParams
{
    float3 cursorPos;
    float3 cubeOrigin;
    float3 cubeDim;
    uint   gridResolution;
    float3 voxelSize;
    float3x3  K;
    float4x4  T,iT,Tbaseline;
    uint winWidth;
    uint winHeight;
    float minDist;
    float maxDist;
};


