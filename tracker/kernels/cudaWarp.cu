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
#include "kernelUtils.h"
#include <cwchar>

// Kernel that executes on the CUDA device
__global__ void warpKernel( float *pR, float *pL, int N, float *tKR, float *T1, float *T2, float *T3, float *iKL, float *K3, float *outPoints)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 h1,h2,h3,l,ltmp,P,R;
	float H[9];
	if ( idx < N ) {
		ltmp = make_float3(1,0,-pR[idx*2]);
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
		
		H[0+0*3] = h1.x; H[1+0*3] = h2.x; H[2+0*3] = h3.x;
		H[0+1*3] = h1.y; H[1+1*3] = h2.y; H[2+1*3] = h3.y;
		H[0+2*3] = h1.z; H[1+2*3] = h2.z; H[2+2*3] = h3.z;
	
		matrixMultVec2(iKL, &pL[idx*2], &P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);
	
		outPoints[idx*2+0] = P.x/P.z;
		outPoints[idx*2+1] = P.y/P.z;
	}	
}

__global__ void warpPointsLineKernel( float *pR, float *linesL0, int N, float *tKR, float *T1, float *T2, float *T3, float *iKL, float *K3, float *linesL1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 h1,h2,h3,l,ltmp,P,R;
	float H[9];
	if ( idx < N ) {
		ltmp = make_float3(1,0,-pR[idx*2]);
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
		
		H[0+0*3] = h1.x; H[1+0*3] = h2.x; H[2+0*3] = h3.x;
		H[0+1*3] = h1.y; H[1+1*3] = h2.y; H[2+1*3] = h3.y;
		H[0+2*3] = h1.z; H[1+2*3] = h2.z; H[2+2*3] = h3.z;
	
		matrixMultVec2(iKL, &linesL0[idx*4], &P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);
	
		linesL1[idx*4+0] = P.x/P.z;
		linesL1[idx*4+1] = P.y/P.z;

		matrixMultVec2(iKL, &linesL0[idx*4+2], &P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);
	
		linesL1[idx*4+2] = P.x/P.z;
		linesL1[idx*4+3] = P.y/P.z;
	}	
}



__global__ void warpKernelG( float *pR, float *pL, int N, float *tKR, float *T1, float *T2, float *T3, float *iKL, float *K3, float *outPoints, float *gPointN, float *gPointS, float *gPointE, float *gPointW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 h1,h2,h3,l,ltmp,P,R;
	if ( idx < N ) {
                // center point
		ltmp = make_float3(1,0,-pR[idx*2]);
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
		
                float HC[9];
		HC[0+0*3] = h1.x; HC[1+0*3] = h2.x; HC[2+0*3] = h3.x;
		HC[0+1*3] = h1.y; HC[1+1*3] = h2.y; HC[2+1*3] = h3.y;
		HC[0+2*3] = h1.z; HC[1+2*3] = h2.z; HC[2+2*3] = h3.z;

		// previus point
		ltmp = make_float3(1,0,-(pR[idx*2]-1));
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
/*
                float HP[9]
                HP[0+0*3] = h1.x; HP[1+0*3] = h2.x; HP[2+0*3] = h3.x;
		HP[0+1*3] = h1.y; HP[1+1*3] = h2.y; HP[2+1*3] = h3.y;
		HP[0+2*3] = h1.z; HP[1+2*3] = h2.z; HP[2+2*3] = h3.z;
*/
		// next point
		ltmp = make_float3(1,0,-(pR[idx*2]+1));
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
		
/*                float HN[9];
		HN[0+0*3] = h1.x; HN[1+0*3] = h2.x; HN[2+0*3] = h3.x;
		HN[0+1*3] = h1.y; HN[1+1*3] = h2.y; HN[2+1*3] = h3.y;
		HN[0+2*3] = h1.z; HN[1+2*3] = h2.z; HN[2+2*3] = h3.z;
*/
		// apply homography
		matrixMultVec2(iKL, &pL[idx*2], &P);
		matrixMultVec3(HC, P, R);        
		matrixMultVec3(K3, R, P);
	
		outPoints[idx*2+0] = P.x/P.z;
		outPoints[idx*2+1] = P.y/P.z;
		/*
		// N gpoint
		ltmp = make_float3(1,0,-(pR[idx*2]-1));
		matrixMultVec3(tKR, ltmp, l);
		matrixMultVec3(T1, l, h1);
		matrixMultVec3(T2, l, h2);
		matrixMultVec3(T3, l, h3);
		
		H[0+0*3] = h1.x; H[1+0*3] = h2.x; H[2+0*3] = h3.x;
		H[0+1*3] = h1.y; H[1+1*3] = h2.y; H[2+1*3] = h3.y;
		H[0+2*3] = h1.z; H[1+2*3] = h2.z; H[2+2*3] = h3.z;
	
		matrixMultVec2(iKL, &pL[idx*2], &P);
		matrixMultVec3(H, P, R);        
		matrixMultVec3(K3, R, P);
	
		outPoints[idx*2+0] = P.x/P.z;
		outPoints[idx*2+1] = P.y/P.z;
		*/
	}	
}

extern "C" void	cudaPointLineWarp(float *pR, float *linesL0, int nPoints,float *matrixData,float *linesL1) {
	if (pR == 0 || linesL0 == 0 || nPoints <= 0 || matrixData == 0 || linesL1 == 0) return;
	dim3 threadsPerBlock(512,1,1);
	dim3 numBlocks(nPoints / threadsPerBlock.x + ( nPoints % threadsPerBlock.x == 0 ? 0 : 1 ),1,1);    
	float *tKR = &matrixData[0*9];
	float *T1 = &matrixData[1*9];
	float *T2 = &matrixData[2*9];
	float *T3 = &matrixData[3*9];
	float *iKL = &matrixData[4*9];
	float *K3 = &matrixData[5*9];
	warpPointsLineKernel <<< numBlocks, threadsPerBlock >>> ( pR,linesL0,nPoints,tKR,T1,T2,T3,iKL,K3,linesL1);

}


extern "C" void cudaWarp(float *pR, float *pL, int nPoints, float *matrixData, float *outPoints)
{
	if (pR == 0 || pL == 0 || nPoints <= 0 || matrixData == 0 || outPoints == 0) return;
	/*cudaEvent_t startcu, stopcu;
    float time;	
	cudaEventCreate(&startcu);
    cudaEventCreate(&stopcu);
	cudaEventRecord(startcu, 0);
*/
	dim3 threadsPerBlock(512,1,1);
	dim3 numBlocks(nPoints / threadsPerBlock.x + ( nPoints % threadsPerBlock.x == 0 ? 0 : 1 ),1,1);    
	float *tKR = &matrixData[0*9];
	float *T1 = &matrixData[1*9];
	float *T2 = &matrixData[2*9];
	float *T3 = &matrixData[3*9];
	float *iKL = &matrixData[4*9];
	float *K3 = &matrixData[5*9];
	//cudaThreadSynchronize();
	//if (gPointsNDev == 0 || gPointsSDev == 0 || gPointsEDev == 0 || gPointsWDev == 0) {
	warpKernel <<< numBlocks, threadsPerBlock >>> ( pR,pL,nPoints,tKR,T1,T2,T3,iKL,K3,outPoints);
	/*} else { 
		//printf("running warpKernelG!\n");
		warpKernelG <<< numBlocks, threadsPerBlock >>> ( pR,pL,nPoints,tKR,T1,T2,T3,iKL,K3,outPoints,gPointsNDev,gPointsSDev,gPointsEDev,gPointsWDev);
	}*/
//	cudaEventRecord(stopcu, 0);
	//cudaThreadSynchronize();
	// get elapsed time of device executiom
  //  cudaEventElapsedTime(&time, startcu, stopcu);
    // destroy event objects
    //cudaEventDestroy(startcu);
    //cudaEventDestroy(stopcu);
    //printf("warp time cuda:%f\n",time);
}
