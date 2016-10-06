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


//NOTE: differences to Kinfu
// - zmaps used instead of distances (depthScaled)
// - multiple iterations allowed for intersection point estimation

#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <float.h>
#include <cuda_gl_interop.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "particles_kernel.cuh"
#include <math_constants.h>
#include <helper_math.h>
__constant__ SimParams params;
#include "VoxelGrid.cuh"


namespace voxutils {
    #include "kernelUtils.h"
}
using namespace voxutils;

extern "C"
{
    texture<float4, 2, cudaReadModeElementType> rgbdTex;

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    int iDivUp(int a, int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    inline float frand()
    {
        return rand() / (float) RAND_MAX;
    }


    void sortParticles(float *sortKeys, uint *indices, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
                            thrust::device_ptr<float>(sortKeys + numParticles),
                            thrust::device_ptr<uint>(indices));
    }

    __device__ void transformRT3CUDA(float *M1, float4 *V, float3 *R) {
        R->x = M1[0]*V->x+M1[1]*V->y+M1[2]*V->z+M1[3];
        R->y = M1[4]*V->x+M1[5]*V->y+M1[6]*V->z+M1[7];
        R->z = M1[8]*V->x+M1[9]*V->y+M1[10]*V->z+M1[11];
    }
    __device__ void transformRT4CUDA(float *M1, float4 *V, float4 *R) {
        R->x = M1[0]*V->x+M1[1]*V->y+M1[2]*V->z+M1[3];
        R->y = M1[4]*V->x+M1[5]*V->y+M1[6]*V->z+M1[7];
        R->z = M1[8]*V->x+M1[9]*V->y+M1[10]*V->z+M1[11];
        R->w = 1;
    }

    __global__ void projectionKernel(float4 *vData, float2 *vData2D)
    {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        float *K  = &params.K.m[0];
        float *T  = &params.T.m[0];

        float4 p3  = vData[idx];
        float2 *p2 = &vData2D[idx];

        float3 r;
        transformRT3CUDA(T,&p3,&r); r.x /= r.z; r.y /= r.z; r.z = 1.0f;
        float2 uv;
        uv.x = (r.x*K[0]+K[2])*params.winWidth;
        uv.y = (r.y*K[4]+K[5])*params.winHeight;
        *p2 = uv;
    }

    __global__ void resetKernel(float2 *tsdfData, float v1, float v2)
    {
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        tsdfData[idx].x = v1;
        tsdfData[idx].y = v2;
    }

    __device__ void initRay(int xi, int yi, float *K, float *T, float3 &rayOrigin, float3 &camRayDir, float3 &rayDir, float3 &invDir, int *sign) {
        float2 pd;
        pd.x = (float(xi)/params.winWidth);
        pd.y = (float(yi)/params.winHeight);
        camRayDir.x = -(pd.x-K[2])/K[0];
        camRayDir.y = -(pd.y-K[5])/K[4];
        camRayDir.z =  -1;
        normalize3Cuda(&camRayDir);
        rotate3Cuda(&T[0],&camRayDir,&rayDir);

        rayOrigin.x = T[3];
        rayOrigin.y = T[7];
        rayOrigin.z = T[11];

        sign[0] = rayDir.x<0;
        sign[1] = rayDir.y<0;
        sign[2] = rayDir.z<0;

        if (fabs(rayDir.x) > 1e-6f) invDir.x = 1.0f / rayDir.x; else invDir.x = (1-2*sign[0])*FLT_MAX;
        if (fabs(rayDir.y) > 1e-6f) invDir.y = 1.0f / rayDir.y; else invDir.y = (1-2*sign[1])*FLT_MAX;
        if (fabs(rayDir.z) > 1e-6f) invDir.z = 1.0f / rayDir.z; else invDir.z = (1-2*sign[2])*FLT_MAX;

    }

 //   __device__ void swap(float &v1, float &v2) { float tmp = v1; v1 = v2; v2 = tmp; }

    __device__ bool intersectVoxelGridCuda(const float3 &origin, const float3 &dir, const float3 &invdir, int *sign, const float3 *bounds, float *enterT, float *exitT)
    {
        float x0 = (bounds[sign[0]].x - origin.x) * invdir.x;
        float x1 = (bounds[1-sign[0]].x - origin.x) * invdir.x;
        float y0 = (bounds[sign[1]].y - origin.y) * invdir.y;
        float y1 = (bounds[1-sign[1]].y - origin.y) * invdir.y;
        float z0 = (bounds[sign[2]].z - origin.z) * invdir.z;
        float z1 = (bounds[1-sign[2]].z - origin.z) * invdir.z;

        float tmin = max(x0,max(y0,z0));//vmin.max(x0,x1,vmin.max(y0,y1,vmin.max(z0,z1,t0)));
        float tmax = min(x1,min(y1,z1));//vmax.min(x0,x1,vmax.min(y0,y1,vmax.min(z0,z1,t1)));

        // if not intersection or its behind the camera exit:
        if ((tmin >= tmax) || tmax <= 0) return false;
        // otherwise return the range inside voxel volume
        //if (tmin < 0.0f) return false;//tmin = 0.0f;
        *enterT = tmin;
        *exitT  = tmax;
        return true;
    }

    __device__ uint3 voxelize(const float3 &p, const float3 &voxelDim, float3 &frac) {
        float3 invVoxelDim;
        invVoxelDim.x = 1.0f / voxelDim.x;
        invVoxelDim.y = 1.0f / voxelDim.y;
        invVoxelDim.z = 1.0f / voxelDim.z;
        float3 samplePoint;
        samplePoint.x = p.x*invVoxelDim.x;
        samplePoint.y = p.y*invVoxelDim.y;
        samplePoint.z = p.z*invVoxelDim.z;
        uint3 voff = make_uint3(samplePoint.x,samplePoint.y,samplePoint.z);
        frac.x = samplePoint.x - voff.x;
        frac.y = samplePoint.y - voff.y;
        frac.z = samplePoint.z - voff.z;
        return voff;
    }

    __device__ float trilinearDensity(const float2 *tsdfData, const float3 &p3d) {
        uint3 index = make_uint3(p3d.x,p3d.y,p3d.z);
        float3 frac = make_float3(p3d.x-index.x,p3d.y-index.y,p3d.z-index.z);

      /*  if (index.x > params.gridResolution-2) return 1.0e5f;
        if (index.y > params.gridResolution-2) return 1.0e5f;
        if (index.z > params.gridResolution-2) return 1.0e5f;
        */
        int xstep = 1;
        int ystep = params.gridResolution;
        int zstep = params.gridResolution*params.gridResolution;

        int   offset = index.x+index.y*ystep+index.z*zstep;
        float d000 = tsdfData[offset].x;
        float d001 = tsdfData[offset+zstep].x;
        float d010 = tsdfData[offset+ystep].x;
        float d011 = tsdfData[offset+ystep+zstep].x;
        float d100 = tsdfData[offset+xstep].x;
        float d101 = tsdfData[offset+xstep+zstep].x;
        float d110 = tsdfData[offset+xstep+ystep].x;
        float d111 = tsdfData[offset+xstep+ystep+zstep].x;


        float x00  = (1-frac.x)*d000+frac.x*d100;
        float x01  = (1-frac.x)*d001+frac.x*d101;
        float x10  = (1-frac.x)*d010+frac.x*d110;
        float x11  = (1-frac.x)*d011+frac.x*d111;

        float z0   = (1-frac.y)*x00+frac.y*x10;
        float z1   = (1-frac.y)*x01+frac.y*x11;

        return (1-frac.z)*z0+frac.z*z1;
/*
        return (1-frac.x) * (1-frac.y) * (1-frac.z) * d000+
               (1-frac.x) * (1-frac.y) *    frac.z  * d001+
               (1-frac.x) *    frac.y  * (1-frac.z) * d010+
               (1-frac.x) *    frac.y  *    frac.z  * d011+
                  frac.x  * (1-frac.y) * (1-frac.z) * d100+
                  frac.x  * (1-frac.y) *    frac.z  * d101+
                  frac.x  *    frac.y  * (1-frac.z) * d110+
                  frac.x  *    frac.y  *    frac.z  * d111;*/
    }

    __device__ float trilinearWeight(const float2 *tsdfData, const float3 &p3d) {
        uint3 index = make_uint3(p3d.x,p3d.y,p3d.z);
        float3 frac = make_float3(p3d.x-index.x,p3d.y-index.y,p3d.z-index.z);

      /*  if (index.x > params.gridResolution-2) return 1.0e5f;
        if (index.y > params.gridResolution-2) return 1.0e5f;
        if (index.z > params.gridResolution-2) return 1.0e5f;
        */
        int xstep = 1;
        int ystep = params.gridResolution;
        int zstep = params.gridResolution*params.gridResolution;

        int   offset = index.x+index.y*ystep+index.z*zstep;
        float d000 = tsdfData[offset].y;
        float d001 = tsdfData[offset+zstep].y;
        float d010 = tsdfData[offset+ystep].y;
        float d011 = tsdfData[offset+ystep+zstep].y;
        float d100 = tsdfData[offset+xstep].y;
        float d101 = tsdfData[offset+xstep+zstep].y;
        float d110 = tsdfData[offset+xstep+ystep].y;
        float d111 = tsdfData[offset+xstep+ystep+zstep].y;


        float x00  = (1-frac.x)*d000+frac.x*d100;
        float x01  = (1-frac.x)*d001+frac.x*d101;
        float x10  = (1-frac.x)*d010+frac.x*d110;
        float x11  = (1-frac.x)*d011+frac.x*d111;

        float z0   = (1-frac.y)*x00+frac.y*x10;
        float z1   = (1-frac.y)*x01+frac.y*x11;

        return (1-frac.z)*z0+frac.z*z1;
/*
        return (1-frac.x) * (1-frac.y) * (1-frac.z) * d000+
               (1-frac.x) * (1-frac.y) *    frac.z  * d001+
               (1-frac.x) *    frac.y  * (1-frac.z) * d010+
               (1-frac.x) *    frac.y  *    frac.z  * d011+
                  frac.x  * (1-frac.y) * (1-frac.z) * d100+
                  frac.x  * (1-frac.y) *    frac.z  * d101+
                  frac.x  *    frac.y  * (1-frac.z) * d110+
                  frac.x  *    frac.y  *    frac.z  * d111;*/
    }


    /*
     *  Utility functions to find cubic and quartic roots,
     *  coefficients are passed like this:
     *
     *      c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4 = 0
     *
     *  The functions return the number of non-complex roots and
     *  put the values into the s array.
     *
     *  Author:         Jochen Schwarze (schwarze@isa.de)
     */

//    extern float   sqrt(), cbrt(), cos(), acos();

    /* epsilon surrounding for near zero values */

    #define     EQN_EPS     1e-8f
    #define	    IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)

//    #ifdef NOCBRT
//    #define     cbrt(x)     ((x) > 0.0 ? pow((float)(x), 1.0f/3.0f) : \
 //                             ((x) < 0.0 ? -pow((float)-(x), 1.0f/3.0f) : 0.0))
 //   #endif

    __device__ int solveQuadric(float *c, float *s)
    {
        float p, q, D;

        /* normal form: x^2 + px + q = 0 */

        p = c[ 1 ] / (2 * c[ 2 ]);
        q = c[ 0 ] / c[ 2 ];

        D = p * p - q;

        if (IsZero(D))
        {
            s[ 0 ] = - p;
            return 1;
        }
        else if (D < 0)
        {
            return 0;
        }
        else if (D > 0)
        {
            float sqrt_D = sqrt(D);

            s[ 0 ] =   sqrt_D - p;
            s[ 1 ] = - sqrt_D - p;
            return 2;
        }
        return 0;
    }


    __device__ int solveCubic(float *c, float *s)
    {
        int     i, num;
        float  sub;
        float  A, B, C;
        float  sq_A, p, q;
        float  cb_p, D;

        /* normal form: x^3 + Ax^2 + Bx + C = 0 */

        A = c[ 2 ] / c[ 3 ];
        B = c[ 1 ] / c[ 3 ];
        C = c[ 0 ] / c[ 3 ];

        /*  substitute x = y - A/3 to eliminate quadric term:
        x^3 +px + q = 0 */

        sq_A = A * A;
        p = 1.0/3 * (- 1.0/3 * sq_A + B);
        q = 1.0/2 * (2.0/27 * A * sq_A - 1.0/3 * A * B + C);

        /* use Cardano's formula */

        cb_p = p * p * p;
        D = q * q + cb_p;

        if (IsZero(D))
        {
        if (IsZero(q)) /* one triple solution */
        {
            s[ 0 ] = 0;
            num = 1;
        }
        else /* one single and one float solution */
        {
            float u = cbrt(-q);
            s[ 0 ] = 2.0f * u;
            s[ 1 ] = - u;
            num = 2;
        }
        }
        else if (D < 0) /* Casus irreducibilis: three real solutions */
        {
        float phi = 1.0f/3.0f * acos(-q / sqrt(-cb_p));
        float t = 2.0f * sqrt(-p);

        s[ 0 ] =   t * cos(phi);
        s[ 1 ] = - t * cos(phi + M_PI / 3.0f);
        s[ 2 ] = - t * cos(phi - M_PI / 3.0f);
        num = 3;
        }
        else /* one real solution */
        {
        float sqrt_D = sqrt(D);
        float u = cbrt(sqrt_D - q);
        float v = - cbrt(sqrt_D + q);

        s[ 0 ] = u + v;
        num = 1;
        }

        /* resubstitute */

        sub = 1.0f/3.0f * A;

        for (i = 0; i < num; ++i)
        s[ i ] -= sub;

        return num;
    }

    __device__ void submat3x3CUDA( float *mr, float *mb, int i, int j ) {
      int di, dj, si, sj;
      // loop through 3x3 submatrix
      for( di = 0; di < 3; di ++ ) {
        for( dj = 0; dj < 3; dj ++ ) {
          // map 3x3 element (destination) to 4x4 element (source)
          si = di + ( ( di >= i ) ? 1 : 0 );
          sj = dj + ( ( dj >= j ) ? 1 : 0 );
          // copy element
          mb[di * 3 + dj] = mr[si * 4 + sj];
        }
      }
    }

    __device__ float det3x3CUDA(float *mat )
    {
        float det;
        det = mat[0] * ( mat[4]*mat[8] - mat[7]*mat[5] )
            - mat[1] * ( mat[3]*mat[8] - mat[6]*mat[5] )
            + mat[2] * ( mat[3]*mat[7] - mat[6]*mat[4] );
        return det;
    }

    __device__ float det4x4CUDA( float *mr )
    {
        float det,result = 0, i = 1;
        float msub3[9];
        for ( int n = 0; n < 4; n++, i *= -1 )
        {
            submat3x3CUDA( mr, msub3, 0, n );
            det     = det3x3CUDA( msub3 );
            result += mr[n] * det * i;
        }
        return result;
    }

    __device__ int invert4x4CUDA( float *ma, float *mr )
    {
        float mr_tmp[16];
        float mdet = det4x4CUDA( ma );
        float mtemp[9];
        int     i, j, sign;
        if ( fabs( mdet ) < 0.0005f ) {
            return 0;
        }
        for ( i = 0; i < 4; i++ ) {
            for ( j = 0; j < 4; j++ )
            {
                sign = 1 - ( (i + j) % 2 ) * 2;
                submat3x3CUDA( ma, mtemp, i, j );
                mr_tmp[i+j*4] = ( det3x3CUDA( mtemp ) * sign ) / mdet;
            }
        }
        for ( i = 0; i < 16; i++) mr[i] = mr_tmp[i];
        return 1;
    }

    __device__ void matrixMultVec4CUDA(float *M, float *v, float *r)
    {
        float t[4];
        t[0] = M[0] *v[0] + M[1] *v[1] + M[2] *v[2] + M[3] * v[3];
        t[1] = M[4] *v[0] + M[5] *v[1] + M[6] *v[2] + M[7] * v[3];
        t[2] = M[8] *v[0] + M[9] *v[1] + M[10]*v[2] + M[11]* v[3];
        t[3] = M[12]*v[0] + M[13]*v[1] + M[14]*v[2] + M[15]* v[3];
        r[0] = t[0];
        r[1] = t[1];
        r[2] = t[2];
        r[3] = t[3];
    }

    //TODO: random numerical problems result in random blinking pixels, using doubles not possible ?
    __device__ void estimateTrilinearZero(const float2 *tsdfData, const float3 &origin, const float3 &dir, float t, const float3 &minBound, const float3 &voxelDim, float *tHitZero)
    {
        // compute the voxel we are in:
        float3 frac;
        uint3 voxelIndex = voxelize(origin+t*dir-minBound,voxelDim,frac);

        *tHitZero = t;

        // check its in the grid bounds
        if (voxelIndex.x > params.gridResolution-2) return;
        if (voxelIndex.y > params.gridResolution-2) return;
        if (voxelIndex.z > params.gridResolution-2) return;

        int xstep = 1;
        int ystep = params.gridResolution;
        int zstep = params.gridResolution*params.gridResolution;

        // fetch voxel box corner intensities
        float d[2][2][2];
        int   offset = voxelIndex.x+voxelIndex.y*ystep+voxelIndex.z*zstep;
        d[0][0][0] = tsdfData[offset].x;
        d[0][0][1] = tsdfData[offset+zstep].x;
        d[0][1][0] = tsdfData[offset+ystep].x;
        d[0][1][1] = tsdfData[offset+ystep+zstep].x;
        d[1][0][0] = tsdfData[offset+xstep].x;
        d[1][0][1] = tsdfData[offset+xstep+zstep].x;
        d[1][1][0] = tsdfData[offset+xstep+ystep].x;
        d[1][1][1] = tsdfData[offset+xstep+ystep+zstep].x;


        float3 voxelBounds[2];
        // voxel min bound
        voxelBounds[0] = minBound + voxelDim * make_float3(float(voxelIndex.x),float(voxelIndex.y),float(voxelIndex.z));

        // voxel max bound
        voxelBounds[1] = minBound + voxelDim * make_float3(float(voxelIndex.x+1),float(voxelIndex.y+1),float(voxelIndex.z+1));

        float fracA[2][3];
        float fracB[2][3];

        fracA[0][0] = (voxelBounds[1].x - origin.x)/voxelDim.x;
        fracA[0][1] = (voxelBounds[1].y - origin.y)/voxelDim.y;
        fracA[0][2] = (voxelBounds[1].z - origin.z)/voxelDim.z;

        fracB[0][0] = -dir.x / voxelDim.x;
        fracB[0][1] = -dir.y / voxelDim.y;
        fracB[0][2] = -dir.z / voxelDim.z;

        fracA[1][0] = (origin.x - voxelBounds[0].x)/voxelDim.x;
        fracA[1][1] = (origin.y - voxelBounds[0].y)/voxelDim.y;
        fracA[1][2] = (origin.z - voxelBounds[0].z)/voxelDim.z;

        fracB[1][0] = dir.x / voxelDim.x;
        fracB[1][1] = dir.y / voxelDim.y;
        fracB[1][2] = dir.z / voxelDim.z;

        float c[4] = {0,0,0,0};
        float s[3] = {0,0,0};
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    c[3] += ( fracB[i][0]*fracB[j][1]*fracB[k][2] ) * d[i][j][k];
                    c[2] += ( fracA[i][0]*fracB[j][1]*fracB[k][2] + fracB[i][0]*fracA[j][1]*fracB[k][2] + fracB[i][0]*fracB[j][1]*fracA[k][2] )*d[i][j][k];
                    c[1] += ( fracB[i][0]*fracA[j][1]*fracA[k][2] + fracA[i][0]*fracB[j][1]*fracA[k][2] + fracA[i][0]*fracA[j][1]*fracB[k][2] )*d[i][j][k];
                    c[0] += ( fracA[i][0]*fracA[j][1]*fracA[k][2] ) * d[i][j][k];
                }
            }
        }

        int nZeros = 0;
        if (fabs(c[3]) < 1e-6f) nZeros = solveQuadric(c,s);
        else nZeros = solveCubic(c,s);
        if (nZeros > 0) {
            float minT = 0;//FLT_MAX;
            float nSamples = 0;
            for (int i = 0; i < nZeros; i++) {
                if (/*(s[i] < minT) &&*/ (s[i] > 0) && fabs(s[i]-t)<voxelDim.x/2) {
                    float3 testVoxel = ((origin+s[i]*dir)-minBound)/voxelDim;
                    uint3 testVoxelIndex = make_uint3(uint(testVoxel.x),uint(testVoxel.y),uint(testVoxel.z));
                    if ((testVoxelIndex.x == voxelIndex.x) && (testVoxelIndex.y == voxelIndex.y) && (testVoxelIndex.z == voxelIndex.z)) {
                        minT += s[i];
                        nSamples++;
                    }
                }
            }
            if (nSamples > 0) *tHitZero = minT/nSamples;
//            if (minT != FLT_MAX) *tHitZero = minT;
        }
        return;
    }

    // assume densityA > 0 && densityB < 0
    __device__ void estimateLinearZero(float tA, float densityA, float tB, float densityB, float *tHitZero) {
        float sLinear = clamp(densityA/(densityA-densityB),0.0f,1.0f);
        *tHitZero = tA+sLinear*(tB-tA);
    }

    // assume densityA > 0 && densityD < 0
    // TODO: invert4x4CUDA fails! singular matrix, why?
    __device__ void estimateCubicZero(float tA, float densityA, float tB, float densityB, float tC, float densityC, float tD, float densityD, float *tHitZero)
    {
//        float tLinearZero;
//        estimateLinearZero(tA,densityA,tD,densityD,&tLinearZero);
        float M[16],iM[16];

        float a = 10.0f;
        float b = -tA*a;
        tA = a*tA+b;
        tB = a*tB+b;
        tC = a*tC+b;
        tD = a*tD+b;

        M[0]  = tA*tA*tA; M[1]   = tA*tA; M[2]   = tA; M[3]    = 1;
        M[4]  = tB*tB*tB; M[5]   = tB*tB; M[6]   = tB; M[7]    = 1;
        M[8]  = tC*tC*tC; M[9]   = tC*tC; M[10]  = tC; M[11]   = 1;
        M[12] = tD*tD*tD; M[13]  = tD*tD; M[14]  = tD; M[15]   = 1;
        float B[4];
        B[0] = densityA;
        B[1] = densityB;
        B[2] = densityC;
        B[3] = densityD;

        tA = (tA-b)/a;
        tB = (tB-b)/a;
        tC = (tC-b)/a;
        tD = (tD-b)/a;

        int inversionSucceeded = invert4x4CUDA(&M[0],&iM[0]);
        if (!inversionSucceeded) {
 //           *tHitZero = 0;
            estimateLinearZero(tA,densityA,tD,densityD,tHitZero);
            return;
        }

//       estimateLinearZero(tA,densityA,tD,densityD,tHitZero);
 //       return;

        float coeff[4];
        matrixMultVec4CUDA(iM,B,coeff);

        float tOptions[4];
        int nRoots = solveCubic(&coeff[0],&tOptions[0]);
        if (nRoots < 1) {
            *tHitZero = 0;
//            estimateLinearZero(tA,densityA,tD,densityD,tHitZero);
            return;
        }
//        estimateLinearZero(tA,densityA,tD,densityD,tHitZero);
//        return;

        float minT = tOptions[0];
        for (int i = 1; i < nRoots; i++) {
            if (tOptions[i] < minT) minT = tOptions[i];
        }

        *tHitZero = (minT-b)/a;
    }


    __device__ bool checkGridBounds(uint3 &voff) {
        bool ok = true;
        if (voff.x > params.gridResolution-1) ok = false;
        if (voff.y > params.gridResolution-1) ok = false;
        if (voff.z > params.gridResolution-1) ok = false;
        return ok;
    }

    __device__ bool traverseVoxelsCuda(const float3 &origin,const float3 &rayDir, const float3 &invDir, int *signInt, float enterT, float exitT, const float3 *interpolationBounds, const float3 &voxelDim, float2 *tsdfData, float *tHitZero, float *hitCnt, bool filtering)
    {        
        // no need to check trilinear filtering bounds in each iteration:
        enterT += voxelDim.x*2;
        exitT  -= voxelDim.x*2;
        if (enterT < 0) enterT = 0;
        if (enterT + voxelDim.x*2 > exitT) return false;

        // determine current voxel coordinate:
        float3 frac;
        uint3 voff = voxelize(origin + enterT * rayDir - interpolationBounds[0], voxelDim, frac);
        float3 invVoxelDim = 1.0f / voxelDim;
        float3 voxelPoint = interpolationBounds[0] + make_float3(voff.x,voff.y,voff.z)*voxelDim;
        float3 sign = make_float3(1 - 2*signInt[0], 1 - 2*signInt[1], 1 - 2*signInt[2]);
        // determine voxel-sized steps into ray direction:
        float3 gridStep = make_float3(sign.x*voxelDim.x, sign.y*voxelDim.y, sign.z*voxelDim.z);

        // compute intersections to voxel walls:
        float3 tNext = (voxelPoint + gridStep - origin) * invDir;
        // compute steps in to the next plane intersection
        float3 tStep = gridStep * invDir;

        int iter = 0;
        if ( filtering ) iter = 3;

        float t = enterT;
        // traverse voxels: always proceed to next planar intersection point until outside grid
        // assume: first voxel density > 0
        while (t < exitT) {
//            if (!checkGridBounds(voff)) return false;
            int offset = voff.x + voff.y * params.gridResolution + voff.z * params.gridResolution * params.gridResolution;
            // coarse TSDF intersection test:
            float voxelDensity = tsdfData[offset].x;
            if (voxelDensity < 0.0f) {
            //if (trilinearDensity(voxelGrid,voxelPoint,bounds[0],voxelDim) < 0.0f) {
                float t1        = clamp(dot(voxelPoint-origin,rayDir),enterT,exitT);
                float e1        = trilinearDensity(tsdfData,(origin+t1*rayDir-interpolationBounds[0])*invVoxelDim);
                // fine TSDF intersection test:
                if (e1 < 0.0f) {
                    float t0 = t1 - voxelDim.x/2;
                    while ( t0 > enterT ) {
                        float e0         = trilinearDensity(tsdfData,(origin+t0*rayDir-interpolationBounds[0])*invVoxelDim);
                        if (e0 > 0.0f ) {
                            float tx;
                            estimateLinearZero(t0, e0, t1, e1, &tx);
                            while (iter > 0) {
                                float ex = trilinearDensity(tsdfData,(origin+tx*rayDir-interpolationBounds[0])*invVoxelDim);
                                if (ex < 0.0f) {
                                    t1 = tx;
                                    e1 = ex;
                                } else {
                                    t0 = tx;
                                    e0 = ex;
                                }
                                estimateLinearZero(t0, e0, t1, e1, &tx);
                                iter--;
                            }
                            *tHitZero = tx;
                            *hitCnt = tsdfData[offset].y;
                            return true;
                        } else if (e0 > e1) {
                            t1 = t0;
                            e1 = e0;
                        }
                        t0 -= voxelDim.x*0.5f;
                    }
                    // surface not found, but supposed to? -> exit
                    return false;
                }
            }
            if (tNext.x < tNext.y) {
                if (tNext.x < tNext.z) {
                    t = tNext.x;
                    voxelPoint.x += gridStep.x;
                    voff.x += sign.x;
                    tNext.x += tStep.x;
                } else {
                    t = tNext.z;
                    voxelPoint.z += gridStep.z;
                    voff.z += sign.z;
                    tNext.z += tStep.z;
                }
            } else {
                if (tNext.y < tNext.z) {
                    t = tNext.y;
                    voxelPoint.y += gridStep.y;
                    voff.y += sign.y;
                    tNext.y += tStep.y;
                } else {
                    t = tNext.z;
                    voxelPoint.z += gridStep.z;
                    voff.z += sign.z;
                    tNext.z += tStep.z;
                }
            }
        }
        return false;
    }



    __global__ void  rayCastVoxelsKernel(float2 *tsdfData, uint width, uint height, bool useCubicFilter, float4 *cudaRayCastImage) {
        int xi     = blockIdx.x*blockDim.x+threadIdx.x;
        int yi     = blockIdx.y*blockDim.y+threadIdx.y;
        int offset = xi+yi*width;
        float *K  = &params.K.m[0];
        float *T  = &params.iT.m[0];
        float minDist = params.minDist;
        float maxDist = params.maxDist;
        float3 bounds3d[2];
        float3 interpolationBounds[2];
        bounds3d[0] = params.cubeOrigin-params.cubeDim;
        bounds3d[1] = params.cubeOrigin+params.cubeDim;
        interpolationBounds[0] = params.cubeOrigin-params.cubeDim+params.voxelSize;
        //interpolationBounds[1] = params.cubeOrigin+params.cubeDim-params.voxelSize;

        float3 voxelDim = 2.0f*params.voxelSize;
        float3 rayOrigin, camRayDir, rayDir, invRayDir; int sign[3];

        initRay(xi,yi,K,T,rayOrigin,camRayDir,rayDir,invRayDir,&sign[0]);
        float4 *pixel = &cudaRayCastImage[offset];
        float enterT,exitT,tHitZero,hitCnt;
        if (intersectVoxelGridCuda(rayOrigin,rayDir,invRayDir,&sign[0],&bounds3d[0],&enterT,&exitT)) {
            if (traverseVoxelsCuda(rayOrigin,rayDir,invRayDir,&sign[0], enterT,exitT,&interpolationBounds[0],voxelDim, tsdfData, &tHitZero, &hitCnt, useCubicFilter))
            {
                float3 ip = tHitZero*camRayDir;
                pixel->x = ip.x;
                pixel->y = ip.y;
                pixel->z = ip.z;
                pixel->w = (tHitZero > minDist && tHitZero < maxDist /* && hitCnt >= 1.0f*/) ? 1.0f : 0.0f;
                return;
            }
        }
        pixel->x = 0.0f;
        pixel->y = 0.0f;
        pixel->z = 0.0f;
        pixel->w = 0.0f;
    }

    __device__ bool filter3(float4 *p, int width, float *w, float4 &filtered) {
        if (p->w <= 0.0f) return false;
        filtered = *p;
        return true; /*
        filtered = make_float4(0,0,0,0);
        int off = 0;
        int nsamples = 0;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++,off++) {
                float4 sample = p[j*width+i];
                if (sample.w > 0.0f) {
                    filtered += sample*w[off]; nsamples++;
                }
            }
        }
        return nsamples > 0;
        */
    }


    __global__ void  smoothingKernel(float4 *cudaRayCastImageXYZ, uint width, uint height, float4 *cudaSmoothXYZ) {
        int xi     = blockIdx.x*blockDim.x+threadIdx.x;
        int yi     = blockIdx.y*blockDim.y+threadIdx.y;
        int offset = xi+yi*width;

        float4 *pixel = &cudaSmoothXYZ[offset];
        pixel->x = 0.0f;
        pixel->y = 0.0f;
        pixel->z = 0.0f;
        pixel->w = 0.0f;

        if (xi >= (width-2) || yi >= (height-2)) return;
        if (xi <  2 || yi < 2) return;

        float4 *p   = &cudaRayCastImageXYZ[offset];         if (p->w <= 0.0f) return;

        float zpivot      = p->z;
        float zfiltered   = 0.0f;
        float sumw        = 0.0f;
/*
        float gauss5x5[] = {
            0.01,0.02,0.04,0.02,0.01,
            0.02,0.04,0.08,0.04,0.02,
            0.04,0.08,0.16,0.08,0.04,
            0.02,0.04,0.08,0.04,0.02,
            0.01,0.02,0.04,0.02,0.01,
        };
*/
        float gauss3x3[] = {
            1/15.0f, 2/15.0f, 1/15.0f,
            2/15.0f, 4/15.0f, 2/15.0f,
            1/15.0f, 2/15.0f, 1/15.0f
        };

        int off=0;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++,off++) {
                float4 *p = &cudaRayCastImageXYZ[offset+i+j*width];
                if (p->w <= 0.0f) continue;
                float z = p->z;
                float zdiff = z-zpivot; zdiff *= zdiff; zdiff=zdiff*1000;
                float weight = gauss3x3[off]*expf(-zdiff);///zdiff;
                zfiltered += z*weight;
                sumw += weight;
            }
        }
        pixel->x = p->x;
        pixel->y = p->y;
        pixel->z = zfiltered/sumw;
        pixel->w = p->w;
    }

    __global__ void  shadingKernel(float4 *cudaRayCastImageXYZ, uint width, uint height, float4 *cudaRayCastImage, float4 *cudaRayCastNormalImage) {
        int xi     = blockIdx.x*blockDim.x+threadIdx.x;
        int yi     = blockIdx.y*blockDim.y+threadIdx.y;
        int offset = xi+yi*width;

        float4 *pixel  = &cudaRayCastImage[offset];
        float4 *normal = &cudaRayCastNormalImage[offset];

        pixel->x = 0.0f;
        pixel->y = 0.0f;
        pixel->z = 0.0f;
        pixel->w = 1.0f;
        normal->x = 0.0f;
        normal->y = 0.0f;
        normal->z = 0.0f;
        normal->w = 0.0f;

        if (xi >= (width-3) || yi >= (height-3)) return;
        if (xi <  3 || yi < 3) return;

        float gauss3x3[] = {
            1/15.0f, 2/15.0f, 1/15.0f,
            2/15.0f, 4/15.0f, 2/15.0f,
            1/15.0f, 2/15.0f, 1/15.0f
        };

        float4 p1x,p0x,p1y,p0y;
        if (!filter3(&cudaRayCastImageXYZ[offset+1],width,&gauss3x3[0],p1x))     return;
        if (!filter3(&cudaRayCastImageXYZ[offset-1],width,&gauss3x3[0],p0x))     return;
        if (!filter3(&cudaRayCastImageXYZ[offset+width],width,&gauss3x3[0],p1y)) return;
        if (!filter3(&cudaRayCastImageXYZ[offset-width],width,&gauss3x3[0],p0y)) return;

        float4 u = p1x - p0x;
        float4 v = (p0y - p1y);
        float3 pointNormal;
        pointNormal.x = u.y*v.z - u.z*v.y;
        pointNormal.y = -(u.x*v.z - u.z*v.x);
        pointNormal.z = u.x*v.y - u.y*v.x;
        normalize3Cuda(&pointNormal);

        float color = dot(pointNormal,make_float3(0.0f,0.0f,1.0f));
        if (color > 0.0f)
        {
            pixel->x = color;
            pixel->y = color;
            pixel->z = color;
            normal->x = pointNormal.x;
            normal->y = pointNormal.y;
            normal->z = pointNormal.z;
        }
    }

    __global__ void resampleVoxelsKernel(float2 *tsdfDst, float2 *tsdfSrc) {
        unsigned int ix = blockIdx.x*blockDim.x+threadIdx.x;
        unsigned int iy = blockIdx.y*blockDim.y+threadIdx.y;
        unsigned int iz = blockIdx.z*blockDim.z+threadIdx.z;

        if (ix > params.gridResolution-1) return;
        if (iy > params.gridResolution-1) return;
        if (iz > params.gridResolution-1) return;

        int idx = ix + iy*params.gridResolution + iz*params.gridResolution*params.gridResolution;
        //float *K   = &params.K.m[0];
        //float *T   = &params.T.m[0];
        float *iT  = &params.iT.m[0];

        float3 voxelSize = make_float3(params.voxelSize.x,params.voxelSize.y,params.voxelSize.z);
        float3 start = make_float3(
                    params.cubeOrigin.x - params.cubeDim.x + voxelSize.x,
                    params.cubeOrigin.y - params.cubeDim.y + voxelSize.y,
                    params.cubeOrigin.z - params.cubeDim.z + voxelSize.z);

        voxelSize.x *= 2; voxelSize.y *= 2; voxelSize.z *= 2;

        float4 p3 = make_float4(start.x + ix*voxelSize.x,start.y+iy*voxelSize.y,start.z+iz*voxelSize.z,1);
        float2 *dtsdf  = &tsdfDst[idx];

        float3 r;
        transformRT3CUDA(iT,&p3,&r);
        float3 iv;
        iv.x = (r.x-start.x)/voxelSize.x;
        iv.y = (r.y-start.y)/voxelSize.y;
        iv.z = (r.z-start.z)/voxelSize.z;

        bool outOfBounds = false;
        if (iv.x < 0 || iv.x > params.gridResolution-2) outOfBounds = true;
        if (iv.y < 0 || iv.y > params.gridResolution-2) outOfBounds = true;
        if (iv.z < 0 || iv.z > params.gridResolution-2) outOfBounds = true;

        if (!outOfBounds) {
            dtsdf->x = trilinearDensity(tsdfSrc,iv);
            dtsdf->y = trilinearWeight(tsdfSrc, iv);
        } else {
            dtsdf->x = 1;
            dtsdf->y = 0;
        }
    }


    //TODO: weight by max(dot(src.normal,dst.normal),0);
    __global__ void updateVoxelsKernel(float2 *tsdfData, float *distMap)
    {
        unsigned int ix = blockIdx.x*blockDim.x+threadIdx.x;
        unsigned int iy = blockIdx.y*blockDim.y+threadIdx.y;
        unsigned int iz = blockIdx.z*blockDim.z+threadIdx.z;

        if (ix > params.gridResolution-1) return;
        if (iy > params.gridResolution-1) return;
        if (iz > params.gridResolution-1) return;

        int idx = ix + iy*params.gridResolution + iz*params.gridResolution*params.gridResolution;
        float *K   = &params.K.m[0];
        float *T   = &params.T.m[0];
        //float *iT  = &params.iT.m[0];
        int w   = int(params.winWidth);
        int h   = int(params.winHeight);

        float3 voxelSize = make_float3(params.voxelSize.x,params.voxelSize.y,params.voxelSize.z);
        float3 start = make_float3(
                    params.cubeOrigin.x - params.cubeDim.x + voxelSize.x,
                    params.cubeOrigin.y - params.cubeDim.y + voxelSize.y,
                    params.cubeOrigin.z - params.cubeDim.z + voxelSize.z);

        voxelSize.x *= 2; voxelSize.y *= 2; voxelSize.z *= 2;

        float4 p3 = make_float4(start.x + ix*voxelSize.x,start.y+iy*voxelSize.y,start.z+iz*voxelSize.z,1);
        float2 *tsdf  = &tsdfData[idx];

        float3 r; float2 uv;
        transformRT3CUDA(T,&p3,&r);
        float voxelDist =  -r.z;//sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        r.x /= r.z; r.y /= r.z;
        uv.x = ( r.x*K[0]+K[2])*params.winWidth;
        uv.y = ( r.y*K[4]+K[5])*params.winHeight;

        int ui = int(uv.x);   int   vi = int(uv.y);

        int safeBorder = 1;//16;
        if (ui < safeBorder || ui >= (w-1-safeBorder) || vi < safeBorder || vi >= (h-1-safeBorder)) return;

        int offset = ui+vi*w;
        float m3a = distMap[offset];
        float m3b = distMap[offset+1];
        float m3c = distMap[offset+w];
        float m3d = distMap[offset+w+1];

         if (m3a < params.voxelSize.x*2) return;
         if (m3b < params.voxelSize.x*2) return;
         if (m3c < params.voxelSize.x*2) return;
         if (m3d < params.voxelSize.x*2) return;

        float fu = uv.x - ui;
        float fv = uv.y - vi;
        float dist = ((1-fu)*(1-fv)*m3a + fu*(1-fv)*m3b + (1-fu)*fv*m3c + fu*fv*m3d);
        //float dist = m3a;//((1-fu)*(1-fv)*m3a + fu*(1-fv)*m3b + (1-fu)*fv*m3c + fu*fv*m3d);
        //      float zerr2 = (z-m3a)*(z-m3a)+(z-m3b)*(z-m3b)+(z-m3c)*(z-m3c)+(z-m3d)*(z-m3d);
        if (fabs(dist-m3a) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3b) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3c) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3d) > params.cubeDim.x/200.0f) return;

        float newWeight = 1.0f;
        // what is the voxel depth difference to the new measurement?
        float slopeRange = params.cubeDim.x*0.030f;//cubeDim.x*(0.02f+0.01f*min(z/250.0f,0.0f));
        float sdf =  -(voxelDist - dist)/slopeRange;
        if (sdf < -5.0/4.0f) return;
        float newTsdf = fmax(fmin(1.0f,sdf),-1.0f);
        tsdf->x = (tsdf->y*tsdf->x + newWeight*newTsdf)/(tsdf->y+newWeight);
        // always allow slow update to field :
        tsdf->y = fmin(newWeight+tsdf->y,16.0f);
    }

    //TODO: weight by max(dot(src.normal,dst.normal),0);
    __global__ void updateVoxelsXYZKernel(float2 *tsdfData, float4 *xyzMap, float strength)
    {
        unsigned int ix = blockIdx.x*blockDim.x+threadIdx.x;
        unsigned int iy = blockIdx.y*blockDim.y+threadIdx.y;
        unsigned int iz = blockIdx.z*blockDim.z+threadIdx.z;

        if (ix > params.gridResolution-1) return;
        if (iy > params.gridResolution-1) return;
        if (iz > params.gridResolution-1) return;

        int idx = ix + iy*params.gridResolution + iz*params.gridResolution*params.gridResolution;
        float *K   = &params.K.m[0];
        float *T   = &params.T.m[0];
        //float *iT  = &params.iT.m[0];
        int w   = int(params.winWidth);
        int h   = int(params.winHeight);

        float3 voxelSize = make_float3(params.voxelSize.x,params.voxelSize.y,params.voxelSize.z);
        float3 start = make_float3(
                    params.cubeOrigin.x - params.cubeDim.x + voxelSize.x,
                    params.cubeOrigin.y - params.cubeDim.y + voxelSize.y,
                    params.cubeOrigin.z - params.cubeDim.z + voxelSize.z);

        voxelSize.x *= 2; voxelSize.y *= 2; voxelSize.z *= 2;

        float4 p3 = make_float4(start.x + ix*voxelSize.x,start.y+iy*voxelSize.y,start.z+iz*voxelSize.z,1);
        float2 *tsdf  = &tsdfData[idx];

        float3 r; float2 uv;
        transformRT3CUDA(T,&p3,&r);
        float voxelDist =  -r.z;//sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        r.x /= r.z; r.y /= r.z;
        uv.x = ( r.x*K[0]+K[2])*params.winWidth;
        uv.y = ( r.y*K[4]+K[5])*params.winHeight;

        int ui = int(uv.x);   int   vi = int(uv.y);

        int safeBorder = 1;//16;
        if (ui < safeBorder || ui >= (w-1-safeBorder) || vi < safeBorder || vi >= (h-1-safeBorder)) return;

        int offset = ui+vi*w;
        float m3a = -xyzMap[offset].z;
        float m3b = -xyzMap[offset+1].z;
        float m3c = -xyzMap[offset+w].z;
        float m3d = -xyzMap[offset+w+1].z;

         if (m3a < params.voxelSize.x*2) return;
         if (m3b < params.voxelSize.x*2) return;
         if (m3c < params.voxelSize.x*2) return;
         if (m3d < params.voxelSize.x*2) return;

        float fu = uv.x - ui;
        float fv = uv.y - vi;
        float dist = ((1-fu)*(1-fv)*m3a + fu*(1-fv)*m3b + (1-fu)*fv*m3c + fu*fv*m3d);
        //float dist = m3a;//((1-fu)*(1-fv)*m3a + fu*(1-fv)*m3b + (1-fu)*fv*m3c + fu*fv*m3d);
        //      float zerr2 = (z-m3a)*(z-m3a)+(z-m3b)*(z-m3b)+(z-m3c)*(z-m3c)+(z-m3d)*(z-m3d);
        if (fabs(dist-m3a) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3b) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3c) > params.cubeDim.x/200.0f) return;
        if (fabs(dist-m3d) > params.cubeDim.x/200.0f) return;

        float newWeight = strength;
        // what is the voxel depth difference to the new measurement?
        float slopeRange = params.cubeDim.x*0.050f;//cubeDim.x*(0.02f+0.01f*min(z/250.0f,0.0f));
        float sdf =  -(voxelDist - dist)/slopeRange;
        if (sdf < -5.0/4.0f) return;
        float newTsdf = fmax(fmin(1.0f,sdf),-1.0f);
        tsdf->x = (tsdf->y*tsdf->x + newWeight*newTsdf)/(tsdf->y+newWeight);
        // always allow slow update to field :
        tsdf->y = fmin(newWeight+tsdf->y,16.0f);
    }


    __global__ void reconstructionKernel(float *vData, int width, int height)
    {
        int xi     = blockIdx.x*blockDim.x+threadIdx.x;
        int yi     = blockIdx.y*blockDim.y+threadIdx.y;
        int offset = xi+yi*width;
        //float *K    = &params.K.m[0];
        float maxDist = params.maxDist;
        float minDist = params.minDist;
        float4 rgbd = tex2D(rgbdTex, xi, height-1-yi);
        float tol = 1e-4f;
        if (rgbd.w > tol && rgbd.w < (1.0f-tol)) {
            //float *dist  = &vData[offset];
            float z = rgbd.w*(maxDist-minDist)+minDist;
           /* float2 pd;
            pd.x = float(xi)/width;
            pd.y = float(yi)/height;
            float3 xy;
            xy.x =  (pd.x-K[2])/K[0];
            xy.y =  (pd.y-K[5])/K[4];*/
            vData[offset] = z;//*sqrtf(xy.x*xy.x + xy.y*xy.y + 1);
        } else {
            vData[offset] = 0;
        }
    }

    void projectVoxelGrid(float4 *voxelVert3D, float2 *voxelVert2D, uint nPoints) {
        if (voxelVert3D == NULL || voxelVert2D == NULL) return;

        // enforce multiple of 1024 for element count -> max performance
        if (nPoints%1024 != 0) {
            printf("projectVoxelGrid: nPoints must be multiple of 1024!\n");
            return;
        }
        dim3 cudaBlockSize(1024,1,1);
        dim3 cudaGridSize(nPoints/cudaBlockSize.x,1,1);
        projectionKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(voxelVert3D,voxelVert2D);
    }

    void resetVoxelGrid(float2 *tsdfData, uint nCubes, float v1, float v2) {
        if (tsdfData == NULL || nCubes <= 0) return;

        // enforce multiple of 1024 for element count -> max performance
        if (nCubes%1024 != 0) {
            printf("resetVoxelGrid: nCubes must be multiple of 1024!\n");
            return;
        }
        dim3 cudaBlockSize(1024,1,1);
        dim3 cudaGridSize(nCubes/cudaBlockSize.x,1,1);
        resetKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(tsdfData,v1,v2);
    }

    void resampleVoxelGrid(float2 *tsdfDataDst, float2 *tsdfDataSrc, uint gridResoX, uint gridResoY, uint gridResoZ) {
        if (tsdfDataDst == NULL || tsdfDataSrc == NULL) return;

        dim3 cudaBlockSize(8,8,8);
        dim3 cudaGridSize(gridResoX/cudaBlockSize.x,gridResoY/cudaBlockSize.y,gridResoZ/cudaBlockSize.z);
        resampleVoxelsKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(tsdfDataDst, tsdfDataSrc);
    }



    void rayCastVoxels(float2 *tsdfData, uint width, uint height, float4 *cudaRayCastImage, bool useCubicFilter, float4 *cudaRayCastImageXYZ, float4 *cudaRayCastNormalImage) {
        if (tsdfData == NULL) return;

        dim3 cudaBlockSize(32, 20, 1);
        dim3 cudaGridSize(width/cudaBlockSize.x, height/cudaBlockSize.y, 1);

        //printf("use cubic filter: %d\n", int(useCubicFilter));
       // cudaMemset(cudaRayCastImage,255,nPoints*sizeof(float)*4);
        rayCastVoxelsKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(tsdfData, width, height, useCubicFilter, cudaRayCastImageXYZ);
        cudaThreadSynchronize();
        shadingKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(cudaRayCastImageXYZ, width, height, cudaRayCastImage, cudaRayCastNormalImage);
     //   smoothingKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(cudaRayCastImageXYZ, width, height, cudaSmoothRayCastImageXYZ);
     //   shadingKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(cudaSmoothRayCastImageXYZ, width, height, cudaRayCastImage);
    }


    void updateVoxelGrid(float2 *tsdf, uint gridResolutionX, uint gridResolutionY, uint gridResolutionZ, float *distMap) {
        if (tsdf == NULL) return;
        dim3 cudaBlockSize(8,8,8);
        dim3 cudaGridSize(gridResolutionX/cudaBlockSize.x,gridResolutionY/cudaBlockSize.y,gridResolutionZ/cudaBlockSize.z);
        updateVoxelsKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(tsdf, distMap);
    }

    void updateVoxelGridXYZ(float2 *tsdf, uint gridResoX, uint gridResoY, uint gridResoZ, float4 *xyzMap) {
        if (tsdf == NULL) return;
        dim3 cudaBlockSize(8,8,8);
        dim3 cudaGridSize(gridResoX/cudaBlockSize.x,gridResoY/cudaBlockSize.y,gridResoZ/cudaBlockSize.z);
        updateVoxelsXYZKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(tsdf, xyzMap, 16);
    }


    void reconstructCuda(cudaArray *rgbdImage, float *outData, int width, int height) {
        dim3 block(32, 32, 1);
        dim3 grid(width/block.x, height/block.y, 1);
        if (rgbdImage == NULL) return;

       checkCudaErrors(cudaBindTextureToArray(rgbdTex, rgbdImage));
       struct cudaChannelFormatDesc desc;
       checkCudaErrors(cudaGetChannelDesc(&desc, rgbdImage));
#if 0
       printf("CUDA Array channel descriptor, bits per component:\n");
       printf("X %d Y %d Z %d W %d, kind %d\n",
              desc.x,desc.y,desc.z,desc.w,desc.f);

       printf("Possible values for channel format kind: i %d, u%d, f%d:\n",
              cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned,
              cudaChannelFormatKindFloat);
#endif
       reconstructionKernel<<<grid,block,0,0>>>(outData,width,height);
 //       projectionKernel<<<grid,block,0,0>>>(rgbdImage,outData, size);
    }

}   // extern "C"
