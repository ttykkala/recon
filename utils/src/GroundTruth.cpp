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

#include <stdio.h>
#include "basic_math.h"
#include "GroundTruth.h"
#include <string.h>

namespace localFuncs {
char *skipLine(char *ptr)
{	
	int i = 0;
	while (ptr[i] != '\n') {i++;}
	while (ptr[i] == '\n' || ptr[i] == '\r') {i++;}
	//printf("newline start: %c%c%c\n",ptr[0],ptr[1],ptr[2]);
	return &ptr[i];
}

unsigned int getRowCount(char *ptr,int numChars)
{
	int nRows = 0;
	for (int i = 0; i < numChars; i++)
		if (ptr[i] == '\n') nRows++;
	return nRows;
}
}

using namespace localFuncs;

Eigen::Matrix4f *loadCameraMatrices(const char *cameraMatrixFileName, int *numPoses, int skip, float scale)
{
	*numPoses=0;
	int tmp4=0;
	printf("camera matrix file: %s\n",cameraMatrixFileName);
	// load camera matrix data
	unsigned int cameraSize = 0;
	char *cameraData = NULL;
	FILE *g = fopen(cameraMatrixFileName,"rb");
	if (g == NULL) return NULL;
	fseek(g,0,SEEK_END); cameraSize = ftell(g); fseek(g,0,SEEK_SET);
    if (cameraSize == 0) {
        fclose(g);
        return NULL;
    }
	cameraData = new char[cameraSize];
    int ret = fread(cameraData,1,cameraSize,g);
	fclose(g);
	char *cPtr = cameraData;

	int numRows = getRowCount(cameraData,cameraSize);
	if (numRows == 0) {
        delete[] cameraData;
        return NULL;
    }
	printf("numRows:%d\n",numRows);
	Eigen::Matrix4f *matrixData = new Eigen::Matrix4f[numRows];
	float tmp[16];
	//float baseT[16]; identity4x4(baseT);
	printf("parsing camera matrix data..");
	*numPoses = numRows/skip;
	for (int i = 0; i < *numPoses; i++) {
		sscanf(cPtr,"%d: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",&tmp4,&tmp[0],&tmp[1],&tmp[2],&tmp[3],&tmp[4],&tmp[5],&tmp[6],&tmp[7],&tmp[8],&tmp[9],&tmp[10],&tmp[11],&tmp[12],&tmp[13],&tmp[14],&tmp[15]);
        matrixData[i](0,0) = tmp[0];  matrixData[i](0,1) = tmp[1];  matrixData[i](0,2) = tmp[2];   matrixData[i](0,3) = tmp[3]*scale;
        matrixData[i](1,0) = tmp[4];  matrixData[i](1,1) = tmp[5];  matrixData[i](1,2) = tmp[6];   matrixData[i](1,3) = tmp[7]*scale;
        matrixData[i](2,0) = tmp[8];  matrixData[i](2,1) = tmp[9];  matrixData[i](2,2) = tmp[10];  matrixData[i](2,3) = tmp[11]*scale;
        matrixData[i](3,0) = tmp[12]; matrixData[i](3,1) = tmp[13]; matrixData[i](3,2) = tmp[14];  matrixData[i](3,3) = tmp[15];
        //printf("%f %f %f\n",matrixData[i](0,3),matrixData[i](1,3),matrixData[i](2,3));
		//memcpy(&matrixData[i*16],tmpMatrixData,16*sizeof(float));
		for (int li = 0; li < skip; li++) {
			cPtr = skipLine(cPtr);
		}
	}
	printf("done!\n");
	delete[] cameraData;
	return matrixData;
}

void canonizeTrajectory(float *matrixData, int numFrames) {
        float mi[16];
        invertRT4(matrixData,mi);
        for (int i = 0; i < numFrames; i++) {
                matrixMult4x4(mi,&matrixData[i*16],&matrixData[i*16]);
        }
}
