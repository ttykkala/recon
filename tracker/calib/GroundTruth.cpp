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
#include <tracker/basic_math.h>
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

float *loadCameraMatrices(const char *cameraMatrixFileName, int *numRows, int relative)
{
	*numRows=0;
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

	*numRows = getRowCount(cameraData,cameraSize);
    if (*numRows == 0) {
        delete[] cameraData;
        return NULL;
    }
	printf("numFrames:%d\n",*numRows);
	float *matrixData = new float[16*(*numRows)];
	float tmpMatrixData[16];
	float baseT[16]; identity4x4(baseT);
	printf("parsing camera matrix data..");
	for (int i = 0; i < *numRows; i++) {
		sscanf(cPtr,"%d: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",&tmp4,&tmpMatrixData[0],&tmpMatrixData[1],&tmpMatrixData[2],&tmpMatrixData[3],&tmpMatrixData[4],&tmpMatrixData[5],&tmpMatrixData[6],&tmpMatrixData[7],&tmpMatrixData[8],&tmpMatrixData[9],&tmpMatrixData[10],&tmpMatrixData[11],&tmpMatrixData[12],&tmpMatrixData[13],&tmpMatrixData[14],&tmpMatrixData[15]);
		if (relative) {
			matrixMult4x4(baseT,tmpMatrixData,baseT);
			memcpy(&matrixData[i*16],baseT,16*sizeof(float));
		} else { 
			memcpy(&matrixData[i*16],tmpMatrixData,16*sizeof(float));
		}
		cPtr = skipLine(cPtr); 
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
