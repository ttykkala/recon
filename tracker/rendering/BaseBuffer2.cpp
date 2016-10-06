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

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <string>
#include <GL/glew.h> // GLEW Library
// CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <assert.h>
#include "BaseBuffer2.h"
#include <stdio.h>
#include <tracker/basic_math.h>

BaseBuffer2::BaseBuffer2() {
    vbuffer = NULL;
}


BaseBuffer2::~BaseBuffer2() {
    if (vbuffer != NULL) delete vbuffer; vbuffer = NULL;
}


void BaseBuffer2::release() {
    vbuffer->release();
}

void BaseBuffer2::initialize() {
    if (vbuffer == NULL) {
        vbuffer = new VertexBuffer2(10014+320*240*2,true,BASEBUFFER_STRIDE);
        vbuffer->addLine(0,0,0,300,0,0,0,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,0,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,0);
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,1,0,0);
        vbuffer->addLine(0,0,0,0,0,300,1,0,0);
        vbuffer->addLine(0,0,0,0,0,0,1,0,0);
/*
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,1,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,1);
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,1,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,1);
        vbuffer->addLine(0,0,0,0,0,0,1,1,0);
  */

        // src and dst warped 2d points and 10000 slots for trajectory
        for (int i = 0; i < (10000+320*240*2); i++) {
           vbuffer->addVertexWithoutElement(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        }
        vbuffer->upload();
       vbuffer->setVertexAmount(14);
    }
    identity4x4(&invT[0]);
    identity4x4(&T[0]);
}

void BaseBuffer2::reset() {
    if (vbuffer != NULL) {
        vbuffer->setVertexAmount(0);
        vbuffer->addLine(0,0,0,300,0,0,0,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,0,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,0);
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,1,0,0);
        vbuffer->addLine(0,0,0,0,0,300,1,0,0);
        vbuffer->addLine(0,0,0,0,0,0,1,0,0);
/*
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,1,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,1);
        vbuffer->addLine(0,0,0,300,0,0,1,0,0);
        vbuffer->addLine(0,0,0,0,300,0,0,1,0);
        vbuffer->addLine(0,0,0,0,0,300,0,0,1);
        vbuffer->addLine(0,0,0,0,0,0,1,1,0);
  */

        // src and dst warped 2d points and 10000 slots for trajectory
        for (int i = 0; i < (10000+320*240*2); i++) {
            vbuffer->addVertexWithoutElement(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
        }
        vbuffer->upload();
        vbuffer->setVertexAmount(14);
    }
    identity4x4(&invT[0]);
    identity4x4(&T[0]);
}

void BaseBuffer2::renderBase() {
    if (vbuffer == NULL) return;
    vbuffer->renderBaseBufferLines();
}

void BaseBuffer2::renderSrcPoints(int cnt) {
    if (vbuffer == NULL) return;
    vbuffer->renderPointRange2d(10014,10014+cnt);
}

void BaseBuffer2::renderDstPoints(int cnt) {
    if (vbuffer == NULL) return;
    vbuffer->renderPointRange2d(10014+320*240,10014+320*240+cnt);
}

float *BaseBuffer2::getCurrentPose() {
    float *m = &T[0];
    float *mInv = &invT[0];
    float tmp[16];
    invertRT4(mInv,&tmp[0]);
    transpose4x4(&tmp[0],m);
    return m;
}

void BaseBuffer2::downloadBaseCPU(float *devT) {
    cudaMemcpy(&invT[0],devT,16*sizeof(float),cudaMemcpyDeviceToHost);
}

