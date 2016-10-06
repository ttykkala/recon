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


#include "ResultTxt.h"
#include <string.h>
#include <tracker/basic_math.h>

ResultTxt::ResultTxt() {
    data = new float[16*MAX_POSE_COUNT];
    nRows = 0;
}


void ResultTxt::reset(bool addIdentityFlag) {
    nRows = 0;
    // the first pose is always identity as trajectory is represented as canonical sequence
    if (addIdentityFlag) {
        float mtx[16]; identity4x4(mtx);
        addPose(mtx);
    }
}

void ResultTxt::init(const char *inputFileName, bool addIdentityFlag) {
    reset(addIdentityFlag);
    strcpy(fn,inputFileName);
}

void ResultTxt::addPose(float *m4x4) {
    if (nRows >= 0 && nRows < MAX_POSE_COUNT) { memcpy(&data[nRows*16],m4x4,sizeof(float)*16); nRows++; }
}

ResultTxt::~ResultTxt() {
    if (data != NULL) delete[] data; data = NULL;
}

void ResultTxt::canonize() {
    float mi[16];
    invertRT4(&data[0],mi);
    for (int i = 0; i < nRows; i++) {
        matrixMult4x4(mi,&data[i*16],&data[i*16]);
    }
}

void ResultTxt::save(int transpose, int inverse) {
    if (nRows <= 0) return;
    canonize();
    FILE *f = fopen(fn,"wb");
    if (f == NULL) {
        printf("unable to open %s\n",fn);
        return;
    }
    printf("saving: %s\n",fn);
    for (int i = 0; i < nRows; i++) {
        float *m4x4 = &data[i*16];
        float mat[16],matT[16];
        if (inverse) invertRT4(&m4x4[0],&matT[0]); else memcpy(&matT[0],&m4x4[0],sizeof(float)*16);
        if (transpose) transpose4x4(matT,mat); else memcpy(&mat[0],&matT[0],sizeof(float)*16);
        fprintf(f,"%d: %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",i+1,mat[0],mat[4],mat[8],mat[12],mat[1],mat[5],mat[9],mat[13],mat[2],mat[6],mat[10],mat[14],mat[3],mat[7],mat[11],mat[15]);
    }
    if (f != NULL) fclose(f);
}

