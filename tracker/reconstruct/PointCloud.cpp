#include "PointCloud.h"
// CUDA
#include <cuda_runtime.h>
#include <cutil_inline.h>


PointCloud::PointCloud( int numMaxPoints )
{
	nSamplePoints = 0;
	nSupportPoints = 0;
	nMaxPoints = numMaxPoints;
	pointsR0 = new float[numMaxPoints*2]; supportR0 = new float[numMaxPoints*2];
	pointsL0 = new float[numMaxPoints*2]; supportL0 = new float[numMaxPoints*2];
	pointsR1 = new float[numMaxPoints*2]; supportR1 = new float[numMaxPoints*2];
	pointsL1 = new float[numMaxPoints*2]; supportL1 = new float[numMaxPoints*2];

	cudaMalloc( (void **)&pointsR0Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&pointsR1Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&pointsL0Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&pointsL1Dev, nMaxPoints*sizeof(float)*2); 

	supportR0Dev = NULL;
	supportL0Dev = NULL;
	supportR1Dev = NULL;
	supportL1Dev = NULL;
/*
	cudaMalloc( (void **)&supportR0Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&supportR1Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&supportL0Dev, nMaxPoints*sizeof(float)*2); 
	cudaMalloc( (void **)&supportL1Dev, nMaxPoints*sizeof(float)*2); 
*/
	pointsR0_dx1 = new float[numMaxPoints*2]; 
	pointsR0_dx2 = new float[numMaxPoints*2]; 
	pointsR0_dx3 = new float[numMaxPoints*2]; 
	pointsR0_dx4 = new float[numMaxPoints*2]; 
	pointsR0_dx5 = new float[numMaxPoints*2]; 
	pointsR0_dx6 = new float[numMaxPoints*2]; 
	pointsR0_dxd = new float[numMaxPoints*2];
	differentialsComputed = false;
}

PointCloud::~PointCloud()
{
	delete[] pointsR0; delete[] supportR0; 
	delete[] pointsR1; delete[] supportR1;
	delete[] pointsL0; delete[] supportL0;
	delete[] pointsL1; delete[] supportL1;

	cudaFree(pointsR0Dev); //cudaFree(supportR0Dev);
	cudaFree(pointsR1Dev); //cudaFree(supportR1Dev);
	cudaFree(pointsL0Dev); //cudaFree(supportL0Dev);
	cudaFree(pointsL1Dev); //cudaFree(supportL1Dev);

	delete[] pointsR0_dx1;
	delete[] pointsR0_dx2;
	delete[] pointsR0_dx3;
	delete[] pointsR0_dx4;
	delete[] pointsR0_dx5;
	delete[] pointsR0_dx6;
	delete[] pointsR0_dxd;
}


void PointCloud::addRefPoint( float xR, float yR, float xL, float yL )
{
	pointsR0[nSamplePoints*2+0] = xR;
	pointsR0[nSamplePoints*2+1] = yR;
	pointsL0[nSamplePoints*2+0] = xL;//[4];
	pointsL0[nSamplePoints*2+1] = yL;//[4];
	nSamplePoints++;
}

void PointCloud::addRefPointSupport( float xR, float yR, float xL, float yL )
{
	supportR0[nSupportPoints*2+0] = xR;
	supportR0[nSupportPoints*2+1] = yR;
	supportL0[nSupportPoints*2+0] = xL;
	supportL0[nSupportPoints*2+1] = yL;
	nSupportPoints++;
}

void PointCloud::reset() {
	nSamplePoints = 0;
	nSupportPoints = 0;
	differentialsComputed = false;
}

void PointCloud::updateRefDevice() {
	if (nSamplePoints > 0) {
		cudaMemcpy( pointsR0Dev, pointsR0, nSamplePoints*sizeof(float)*2, cudaMemcpyHostToDevice );
		cudaMemcpy( pointsL0Dev, pointsL0, nSamplePoints*sizeof(float)*2, cudaMemcpyHostToDevice );
	}
	if (nSupportPoints > 0) {
		// copy support points on device to same memory buffer with sample points 
		// this allows single warp call for all points
		cudaMemcpy(pointsR0Dev+nSamplePoints*2,supportR0,nSupportPoints*sizeof(float)*2,cudaMemcpyHostToDevice);
		cudaMemcpy(pointsL0Dev+nSamplePoints*2,supportL0,nSupportPoints*sizeof(float)*2,cudaMemcpyHostToDevice);
		supportR0Dev = pointsR0Dev+nSamplePoints*2;
		supportL0Dev = pointsL0Dev+nSamplePoints*2;
		supportR1Dev = pointsR1Dev+nSamplePoints*2;
		supportL1Dev = pointsL1Dev+nSamplePoints*2;
	//	cudaMemcpy( supportR0Dev, supportR0, nSupportPoints*sizeof(float)*2, cudaMemcpyHostToDevice );
	//	cudaMemcpy( supportL0Dev, supportL0, nSupportPoints*sizeof(float)*2, cudaMemcpyHostToDevice );
	}
}