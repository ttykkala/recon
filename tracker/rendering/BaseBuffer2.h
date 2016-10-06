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

#include "VertexBuffer2.h"

/*
BASEBUFFER VERTEX FORMAT (scratch bufferi, 2d overlay-graffaa varten kuten basebuffer + 3d trajectory)
0 x
1 y
2 z
3 r
4 g
5 b
*/

class BaseBuffer2 {
private:
    VertexBuffer2 *vbuffer;
    float T[16];
    float invT[16];
public:
      BaseBuffer2();
     ~BaseBuffer2();
     void release();
     void reset();
     void initialize();
     void renderBase();
     void renderSrcPoints(int cnt);
     void renderDstPoints(int cnt);
     VertexBuffer2 *getVBuffer() { return vbuffer; }
     float *getCurrentPose();
     void downloadBaseCPU(float *devT);
};
