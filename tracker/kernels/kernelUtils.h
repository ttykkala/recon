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

__device__ void normalizeCuda(float *v) {
    float len = 0.0f;
    len += v[0]*v[0];
    len += v[1]*v[1];
    len += v[2]*v[2];
    len = 1.0f/sqrtf(len);
    v[0] *= len;
    v[1] *= len;
    v[2] *= len;
}

__device__ void normalizeCudaSafe(float *v) {
    float len = 0.0f;
    len += v[0]*v[0];
    len += v[1]*v[1];
    len += v[2]*v[2];
    len = 1.0f/sqrtf(len+1e-7f);
    v[0] *= len;
    v[1] *= len;
    v[2] *= len;
}

__device__ void normalize3Cuda(float3 *v) {
    float len = 0.0f;
    len += v->x*v->x;
    len += v->y*v->y;
    len += v->z*v->z;
    len = 1.0f/sqrtf(len);
    v->x *= len;
    v->y *= len;
    v->z *= len;
}

__device__ void normalize3CudaSafe(float3 *v) {
    float len = 0.0f;
    len += v->x*v->x;
    len += v->y*v->y;
    len += v->z*v->z;
    len = 1.0f/sqrtf(len+1e-7f);
    v->x *= len;
    v->y *= len;
    v->z *= len;
}

__device__ void rotateCuda(float *T, float *v, float *r) {
    r[0] = T[0]*v[0]+T[1]*v[1]+T[2]*v[2];
    r[1] = T[4]*v[0]+T[5]*v[1]+T[6]*v[2];
    r[2] = T[8]*v[0]+T[9]*v[1]+T[10]*v[2];
}

__device__ void rotate3Cuda(float *T, float3 *v, float3 *r) {
    r->x = T[0]*v->x+T[1]*v->y+T[2]*v->z;
    r->y = T[4]*v->x+T[5]*v->y+T[6]*v->z;
    r->z = T[8]*v->x+T[9]*v->y+T[10]*v->z;
}

__device__ void matrixMult4(float *M1, float *M2, float *R)
{
        float Rtmp[16];
        Rtmp[0] = M1[0]*M2[0]+M1[1]*M2[4]+M1[2]*M2[8]+M1[3]*M2[12];
        Rtmp[1] = M1[0]*M2[1]+M1[1]*M2[5]+M1[2]*M2[9]+M1[3]*M2[13];
        Rtmp[2] = M1[0]*M2[2]+M1[1]*M2[6]+M1[2]*M2[10]+M1[3]*M2[14];
        Rtmp[3] = M1[0]*M2[3]+M1[1]*M2[7]+M1[2]*M2[11]+M1[3]*M2[15];

        Rtmp[4] = M1[4]*M2[0]+M1[5]*M2[4]+M1[6]*M2[8]+M1[7]*M2[12];
        Rtmp[5] = M1[4]*M2[1]+M1[5]*M2[5]+M1[6]*M2[9]+M1[7]*M2[13];
        Rtmp[6] = M1[4]*M2[2]+M1[5]*M2[6]+M1[6]*M2[10]+M1[7]*M2[14];
        Rtmp[7] = M1[4]*M2[3]+M1[5]*M2[7]+M1[6]*M2[11]+M1[7]*M2[15];

        Rtmp[8]  = M1[8]*M2[0]+M1[9]*M2[4]+M1[10]*M2[8]+M1[11]*M2[12];
        Rtmp[9]  = M1[8]*M2[1]+M1[9]*M2[5]+M1[10]*M2[9]+M1[11]*M2[13];
        Rtmp[10] = M1[8]*M2[2]+M1[9]*M2[6]+M1[10]*M2[10]+M1[11]*M2[14];
        Rtmp[11] = M1[8]*M2[3]+M1[9]*M2[7]+M1[10]*M2[11]+M1[11]*M2[15];

        Rtmp[12] = M1[12]*M2[0]+M1[13]*M2[4]+M1[14]*M2[8]+M1[15]*M2[12];
        Rtmp[13] = M1[12]*M2[1]+M1[13]*M2[5]+M1[14]*M2[9]+M1[15]*M2[13];
        Rtmp[14] = M1[12]*M2[2]+M1[13]*M2[6]+M1[14]*M2[10]+M1[15]*M2[14];
        Rtmp[15] = M1[12]*M2[3]+M1[13]*M2[7]+M1[14]*M2[11]+M1[15]*M2[15];
        for (int i = 0; i < 16; i++) R[i] = Rtmp[i];
}

__device__ void invertRT4Cuda( float *M, float *Mi )
{
//    for (int i = 0; i < 16; i++) Mi[i] = M[i];
    Mi[0]  = M[0]; Mi[1]  = M[4]; Mi[2]  = M[8];   Mi[3]  = -(M[0]*M[3]+M[4]*M[7]+M[8]*M[11]);
    Mi[4]  = M[1]; Mi[5]  = M[5]; Mi[6]  = M[9];   Mi[7]  = -(M[1]*M[3]+M[5]*M[7]+M[9]*M[11]);
    Mi[8]  = M[2]; Mi[9]  = M[6]; Mi[10] = M[10];  Mi[11] = -(M[2]*M[3]+M[6]*M[7]+M[10]*M[11]);
    Mi[12] = 0;     Mi[13] = 0;     Mi[14] = 0;    Mi[15] = 1;
}


__device__ void normalizeQuaternionCuda(float *q) {
    float len = 1e-7f;
    len += q[0]*q[0];
    len += q[1]*q[1];
    len += q[2]*q[2];
    len += q[3]*q[3];
    len = sqrtf(len);
    q[0] /= len;
    q[1] /= len;
    q[2] /= len;
    q[3] /= len;
}

__device__ void rot2QuaternionCuda(float *m, float *q) {
    float tr = m[0]+m[5]+m[10];
    if (tr > 0) {
        float s = 0.5f / sqrtf(tr+1.0f);
        q[0] = 0.25f / s;
        q[1] = (m[9] - m[6])*s;
        q[2] = (m[2] - m[8])*s; //(t(0,2) - t(2,0))*s;
        q[3] = (m[4] - m[1])*s; //(t(1,0) - t(0,1))*s;
    } else {
        if (m[0] > m[5] && m[0] > m[10]) {
            float s = 2.0f * sqrtf(1.0f+m[0]-m[5]-m[10]);
            q[0] = (m[9]-m[6]) / s;
            q[1] = 0.25f * s;
            q[2] = (m[1] + m[4]) / s;
            q[3] = (m[2]+m[8]) / s;
        } else if (m[5] > m[10]) {
            float s = 2.0f * sqrtf(1.0f + m[5] - m[0] - m[10]);
            q[0] = (m[2]-m[8]) / s;
            q[1] = (m[1] + m[4]) / s;
            q[2] = 0.25f * s;
            q[3] = (m[6]+m[9]) / s;
        } else {
            float s = 2.0f * sqrtf(1.0f + m[10] - m[0] - m[5]);
            q[0] = (m[4]-m[1]) / s;
            q[1] = (m[2]+m[8]) / s;
            q[2] = (m[6]+m[9]) / s;
            q[3] = 0.25f * s;
        }
    }
    normalizeQuaternionCuda(q);
}

__device__ void quaternion2AxisAngleCuda(float *q, float *axisAngle) {

    float ca = q[0];
    axisAngle[3] = acosf(ca) * 2.0f * 180.0f / 3.141592653f;
    float sa  = (float)sqrtf( 1.0f - ca * ca );
    if ( fabs( sa ) < 0.0005f ) sa = 1.0f;
    axisAngle[0] = q[1] / sa;
    axisAngle[1] = q[2] / sa;
    axisAngle[2] = q[3] / sa;
}

__device__ void quaternion2RotCuda(float *q, float *m) {
    float w = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];

    m[0] = w*w+x*x-y*y-z*z;
    m[1] = 2*(x*y-w*z);
    m[2] = 2*(x*z+w*y);
    m[3] = 0;

    m[4] = 2*(x*y+w*z);
    m[5] = w*w-x*x+y*y-z*z;
    m[6] = 2*(y*z-w*x);
    m[7] = 0;

    m[8] = 2*(x*z-w*y);
    m[9] = 2*(y*z+w*x);
    m[10] = w*w-x*x-y*y+z*z;
    m[11] = 0;

    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;
}


__device__ void axisAngle2RotCuda(float *axisAngle, float *T) {
    float q[4];
    float ca = cos(3.141592653f*axisAngle[3]/(2.0f*180.0f));
    q[0] = ca;
    float sa  = (float)sqrtf( 1.0f - ca * ca );
    if ( fabs( sa ) < 0.0005f ) sa = 1.0f;

    q[1] = axisAngle[0]*sa;
    q[2] = axisAngle[1]*sa;
    q[3] = axisAngle[2]*sa;

    normalizeQuaternionCuda(q);
    quaternion2RotCuda(q,T);
}


__device__ void matrixMultVec2(float *M1, float *V, float3 *R)
{
	R->x = M1[0]*V[0]+M1[1]*V[1]+M1[2];
	R->y = M1[3]*V[0]+M1[4]*V[1]+M1[5];
	R->z = M1[6]*V[0]+M1[7]*V[1]+M1[8];
}

__device__ void matrixMultVec3(float *M1, float3 &V, float3 &R)
{
	R.x = M1[0]*V.x+M1[1]*V.y+M1[2]*V.z;
	R.y = M1[3]*V.x+M1[4]*V.y+M1[5]*V.z;
	R.z = M1[6]*V.x+M1[7]*V.y+M1[8]*V.z;
}

__device__ void matrixMultVec4(float *M1, float3 &V, float3 &R)
{
	R.x = M1[0]*V.x+M1[1]*V.y+M1[2]*V.z + M1[3];
	R.y = M1[4]*V.x+M1[5]*V.y+M1[6]*V.z + M1[7];
	R.z = M1[8]*V.x+M1[9]*V.y+M1[10]*V.z + M1[11];
}

__device__ void matrixRot4(float *M1, float3 &V, float3 &R)
{
    R.x = M1[0]*V.x+M1[1]*V.y+M1[2]*V.z;
    R.y = M1[4]*V.x+M1[5]*V.y+M1[6]*V.z;
    R.z = M1[8]*V.x+M1[9]*V.y+M1[10]*V.z;
}

__device__ void normalizeVec3(float *v) {
    float len = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0] /= len;  v[1] /= len; v[2] /= len;
}

__device__ void crossproduct(float *u, float *v, float *w) {
    w[0] = u[1]*v[2]-u[2]*v[1];
    w[1] = -(u[0]*v[2]-u[2]*v[0]);
    w[2] = u[0]*v[1]-u[1]*v[0];
}

__device__ void normalizeMat4(float *T) {
    // extract column vectors SO3
    float u[3],v[3],w[3];
    u[0] = T[0]; u[1] = T[4]; u[2] = T[8];
    v[0] = T[1]; v[1] = T[5]; v[2] = T[9];
    w[0] = T[2]; w[1] = T[6]; w[2] = T[10];

    normalizeVec3(w);
    crossproduct(v,w,u); normalizeVec3(u);
    crossproduct(w,u,v);

    // store column vectors SO3
    T[0] = u[0]; T[4] = u[1]; T[8]  = u[2];
    T[1] = v[0]; T[5] = v[1]; T[9]  = v[2];
    T[2] = w[0]; T[6] = w[1]; T[10] = w[2];
 }

__device__ void bilinearInterpolation(float3 &p, int width, int height, unsigned char *srcPtr, unsigned char &color) {
	// bi-linear interpolation
	int xdi = (int)p.x;
	int ydi = (int)p.y;
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offset2 = xdi+ydi*width;
		float v0 = (float)srcPtr[offset2];       float v1 = (float)srcPtr[offset2+1];
		float v2 = (float)srcPtr[offset2+width]; float v3 = (float)srcPtr[offset2+width+1];
		float fx = p.x - xdi;
		float fy = p.y - ydi;
		color = (unsigned char)((1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3); 
		return;
	}
	color = 0;
}

__device__ void bilinearInterpolation(float3 &p, int width, int height, float *srcPtr, float &color) {
	// bi-linear interpolation
	int xdi = (int)p.x;
	int ydi = (int)p.y;
	if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
		int offset2 = xdi+ydi*width;
		float v0 = srcPtr[offset2];       float v1 = srcPtr[offset2+1];
		float v2 = srcPtr[offset2+width]; float v3 = srcPtr[offset2+width+1];
		float fx = p.x - xdi;
		float fy = p.y - ydi;
		color = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3; 
		return;
	}
	color = 0;
}

__device__ void bilinearInterpolation(int xdi, int ydi, float fx, float fy, int width, float *srcPtr, float &color) {
    int offset2 = xdi+ydi*width;
    float v0 = srcPtr[offset2];       float v1 = srcPtr[offset2+1];
    float v2 = srcPtr[offset2+width]; float v3 = srcPtr[offset2+width+1];
    color = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
}

__device__ void bilinearInterpolation6(float *srcPtr6, float fx, float fy, int pitch, float3 &p) {
    float v0,v1,v2,v3;
    v0 = srcPtr6[0];       v1 = srcPtr6[6];
    v2 = srcPtr6[pitch]; v3 = srcPtr6[pitch+6];
    p.x = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[1];       v1 = srcPtr6[7];
    v2 = srcPtr6[1+pitch]; v3 = srcPtr6[pitch+7];
    p.y = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[2];       v1 = srcPtr6[8];
    v2 = srcPtr6[2+pitch]; v3 = srcPtr6[pitch+8];
    p.z = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
}

__device__ void bilinearInterpolation6N(float *srcPtr6, float fx, float fy, int pitch, float3 &p, float3 &n) {
    float v0,v1,v2,v3;
    v0 = srcPtr6[0];       v1 = srcPtr6[6];
    v2 = srcPtr6[pitch];   v3 = srcPtr6[pitch+6];
    p.x = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[1];       v1 = srcPtr6[7];
    v2 = srcPtr6[1+pitch]; v3 = srcPtr6[pitch+7];
    p.y = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[2];       v1 = srcPtr6[8];
    v2 = srcPtr6[2+pitch]; v3 = srcPtr6[pitch+8];
    p.z = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[3];       v1 = srcPtr6[9];
    v2 = srcPtr6[3+pitch]; v3 = srcPtr6[pitch+9];
    n.x = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[4];       v1 = srcPtr6[10];
    v2 = srcPtr6[4+pitch]; v3 = srcPtr6[pitch+10];
    n.y = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
    v0 = srcPtr6[5];       v1 = srcPtr6[11];
    v2 = srcPtr6[5+pitch]; v3 = srcPtr6[pitch+11];
    n.z = (1-fx)*(1-fy)*v0 + fx*(1-fy)*v1 + (1-fx)*fy*v2 + fx*fy*v3;
}

// this takes into account bilinear filtering bounds and low resolution coordinate domain
__device__ bool inBounds(float2 &p, int w, int h) {
    if (p.x >= 10 && p.x < (w-10) && p.y >= 10 && p.y < (h-10)) return true;
    return false;
}

__device__ void distortPointWithGradient(float2 &pu, float *kc, float *K, float2 &pd, float2 &pn, float2 &pe, float2 &ps, float2 &pw) {
    // distort point
    float ifx = 1.0f/K[0];
    float ify = 1.0f/K[4];
    float r2,r4,r6;
    float radialDist;
    float dx0,dx,dx1;
    float dy0,dy,dy1;

    // generate normalized gradient point coordinates
    pn.x = pu.x +   0; pn.y = pu.y - ify;
    ps.x = pu.x +   0; ps.y = pu.y + ify;
    pw.x = pu.x - ifx; pw.y = pu.y +   0;
    pe.x = pu.x + ifx; pe.y = pu.y +   0;

    // generate r2 components
    dx  = pu.x*pu.x; dy  = pu.y*pu.y;
    dx0 = pw.x*pw.x; dy0 = pn.y*pn.y;
    dx1 = pe.x*pe.x; dy1 = ps.y*ps.y;

    // generate distorted coordinates
    r2 = dx+dy; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pd.x = K[0]*pu.x*radialDist+K[2];
    pd.y = K[4]*pu.y*radialDist+K[5];

    r2 = dx+dy0; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pn.x = K[0]*pn.x*radialDist+K[2];
    pn.y = K[4]*pn.y*radialDist+K[5];

    r2 = dx+dy1; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    ps.x = K[0]*ps.x*radialDist+K[2];
    ps.y = K[4]*ps.y*radialDist+K[5];

    r2 = dx0+dy; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pw.x = K[0]*pw.x*radialDist+K[2];
    pw.y = K[4]*pw.y*radialDist+K[5];

    r2 = dx1+dy; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pe.x = K[0]*pe.x*radialDist+K[2];
    pe.y = K[4]*pe.y*radialDist+K[5];
}


__device__ void distortPoint(float2 &pu, float *kc, float *K, float2 &pd) {
    // distort point
    float r2,r4,r6;
    float radialDist;
    float dx;
    float dy;

    // generate r2 components
    dx  = pu.x*pu.x; dy  = pu.y*pu.y;
    // generate distorted coordinates
    r2 = dx+dy; r4 = r2*r2; r6 = r4 * r2;
    radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    pd.x = K[0]*pu.x*radialDist+K[2];
    pd.y = K[4]*pu.y*radialDist+K[5];
}


__device__ void bilinearInterpolation(float3 &p, int width, int height, float *srcPtr, float &colorR, float &colorG, float &colorB) {
	// bi-linear interpolation
	unsigned int xdi = (unsigned int)p.x;
	unsigned int ydi = (unsigned int)p.y;
	if (xdi < width-2 && ydi < height-2) {
		int pitch = width*3;
		int offsetR2 = 3*xdi+ydi*pitch;
		int offsetG2 = offsetR2+1;
		int offsetB2 = offsetR2+2;
		float fx = p.x - xdi;
		float fy = p.y - ydi;

		float a = (1-fx)*(1-fy);
		float b = fx*(1-fy);
		float c = (1-fx)*fy;
		float d = fx*fy; 

		float v0 = srcPtr[offsetR2];       float v1 = srcPtr[offsetR2+3];
		float v2 = srcPtr[offsetR2+pitch]; float v3 = srcPtr[offsetR2+pitch+3];
		colorR = a*v0 + b*v1 + c*v2 + d*v3;
		
		v0 = srcPtr[offsetG2];       v1 = srcPtr[offsetG2+3];
		v2 = srcPtr[offsetG2+pitch]; v3 = srcPtr[offsetG2+pitch+3];
		colorG = a*v0 + b*v1 + c*v2 + d*v3;

		v0 = srcPtr[offsetB2];       v1 = srcPtr[offsetB2+3];
		v2 = srcPtr[offsetB2+pitch]; v3 = srcPtr[offsetB2+pitch+3];
		colorB = a*v0 + b*v1 + c*v2 + d*v3;

		return;
	}
	colorR = 0;
	colorG = 0;
	colorB = 0;
}

__device__ void bilinearInterpolation(int xdi, int ydi, float fx, float fy, int width, float *srcPtr, float &colorR, float &colorG, float &colorB) {
    int pitch = width*3;
    int offsetR2 = 3*xdi+ydi*pitch;
    int offsetG2 = offsetR2+1;
    int offsetB2 = offsetR2+2;

    float a = (1-fx)*(1-fy);
    float b = fx*(1-fy);
    float c = (1-fx)*fy;
    float d = fx*fy;

    float v0 = srcPtr[offsetR2];       float v1 = srcPtr[offsetR2+3];
    float v2 = srcPtr[offsetR2+pitch]; float v3 = srcPtr[offsetR2+pitch+3];
    colorR = a*v0 + b*v1 + c*v2 + d*v3;

    v0 = srcPtr[offsetG2];       v1 = srcPtr[offsetG2+3];
    v2 = srcPtr[offsetG2+pitch]; v3 = srcPtr[offsetG2+pitch+3];
    colorG = a*v0 + b*v1 + c*v2 + d*v3;

    v0 = srcPtr[offsetB2];       v1 = srcPtr[offsetB2+3];
    v2 = srcPtr[offsetB2+pitch]; v3 = srcPtr[offsetB2+pitch+3];
    colorB = a*v0 + b*v1 + c*v2 + d*v3;
}

__device__ float interpolateFloatPixel(unsigned char *I, int w, int h, float x, float y)
{
	if (x < 0 || y < 0) return 0;
	if (x >= w-2 || y >= h-2) return 0;

	int xi = int(x);
	int yi = int(y);
	float fracX = x-xi;
	float fracY = y-yi;

	float i1 = float(I[xi+yi*w]);
	float i2 = float(I[xi+1+yi*w]);
	float i3 = float(I[xi+(yi+1)*w]);
	float i4 = float(I[(xi+1)+(yi+1)*w]);
	return (1-fracX)*(1-fracY)*i1 + fracX*(1-fracY)*i2 + (1-fracX)*fracY*i3 + fracX*fracY*i4;
}
