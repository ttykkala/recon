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


#include "TrackingParams.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

TrackingParams::TrackingParams() {
    nParams = 0;
    params = NULL;

    defaultParams.trackMode = GEOMETRIC;
}

TrackingParams::~TrackingParams() {
    if (params) delete[] params;
}

char *TrackingParams::nextLine(char *ptr) {
    while (*ptr != '\n') ptr++;
    return ptr+1;
}

int TrackingParams::countLines(char *basePtr, int sz) {
    int nLines = 0;
    char *ptr = basePtr;
    while (ptr-basePtr < sz) {
        if (*ptr == '\n') nLines++;
        ptr++;
    }
    return nLines;
}

void TrackingParams::init(const char *fn) {
    FILE *f = fopen(fn,"rb");
    if (f == NULL) {
        printf("%s not found!\n",fn);
        return;
    }
    fseek(f,0,SEEK_END);
    long int sz = ftell(f);
    fseek(f,0,SEEK_SET);
    char *paramBuf = new char[sz];
    fread(paramBuf,sz,1,f);
    fclose(f);

    nParams = countLines(paramBuf, sz);
    params = new TRACK_PARAMS[nParams];
    char *ptr = paramBuf;
    for (int i = 0; i < nParams; i++)
    {
        int frame = 0;
        char buf[512];
        float error = 0.0f;
        sscanf(ptr,"%d %s %e\n",&frame,&buf[0],&error);
        params[i].frame = frame;
        if (strcmp(buf,"geometric") == 0) {
            params[i].trackMode = GEOMETRIC;
            params[i].lambda = 1;
            printf("frame %04d, geometric, lambda%1.2f\n",frame, 1.0);
        }
        else if (strcmp(buf,"photometric") == 0) {
            params[i].trackMode = PHOTOMETRIC;
            params[i].lambda = 0;
            printf("frame %04d, photometric, lambda%1.2f\n",frame, 0.0);
        }
        else if (strncmp(buf,"biobjective",strlen("biobjective")) == 0) {
            params[i].trackMode = BIOBJECTIVE;
            params[i].lambda = atof(&buf[strlen("biobjective-lambda")]);
            printf("frame %04d, biobjective-lambda%1.2f\n",frame, params[i].lambda);
        }
        else { printf("TrackingParams::init() - invalid mode!\n"); fflush(stdout); }
//        printf("%04d %d %s\n",params[i].frame,params[i].trackMode, buf); fflush(stdout);
        fflush(stdout);
        ptr = nextLine(ptr);
    }
    delete[] paramBuf;
}

TRACK_PARAMS *TrackingParams::getParams(int frameKey) {
    if (!params) {
        return &defaultParams;
    } else {
        int nearestIndex = 0;
        int nearestDist  = INT_MAX;
        for (int i = 0; i < nParams; i++) {
            int dist = abs(params[i].frame-frameKey);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestIndex = i;
            }
        }
        return &params[nearestIndex];
    }
}
