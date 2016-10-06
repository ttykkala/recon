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


#if !defined(__CMATH_H__)
#define __CMATH_H__

typedef struct {
    float rx2,ry2;
    float rx,ry,rz;
    float px,py,pz;
    unsigned char magGrad;
    float color;
    float colorR;
    float colorG;
    float colorB;
} ProjectData;

float rad2deg(float x);
float deg2rad(float x);
void normalize(float *vec);
void normalize(double a[3]);
void scaleVector3(double v[3],double scale);
double dot3(double a[3], double b[3]);
double lensq(double v[3]);
float length3(float *vec);
void get3x3(double matrix4x3[3*4], double matrix3x3[3*3]);
double det3x3(double mat[9] );
float det3x3(float mat[9] );
//float dotProduct6(float *a, float *b);
void identity3x3(double m[9]);
void identity3x3(float m[9]);
void identity4x4(double *M4x4);
void identity4x4(float *M4x4);
void identity3x4(double *m);
void matrixMult(double *K, double *CM, double *P, int nRows, int nCols);
void matrixMult4x4(float *M1, float *M2, float *R);
void matrixMult4x4(double *M1, double *M2, double *R);
void matrixMult4x4(float *M1, double *M2, float *R);
void matrixMult4x4(double *M1, float *M2, float *R);
void matrixMult4x4RT(float *M1, float *M2, float *R);
void transpose3x3(double *M3x3, double *R3x3);
void transpose3x3(float *M3x3, float *R3x3);
void transpose4x4(double *M4x4, double *R4x4);
void transpose4x4(float *M4x4, float *R4x4);
void copy3x3(double *M3x4, double *R3x3);
void copy3x3f(double *M3x4, float *R3x3);
void copy3x3(float *M3x4, float *R3x3);
void copyT3x1(float *M4, float *T3x1);
void matrixMultVec3(float *M1, float *V, float *R);
void matrixMultVec3(double *M1, double *V, double *R);
void invertRT( double *R, double *t, double *Ri, double *ti );
void invertRT( float *R, float *t, float *Ri, float *ti );
void invertRT4( double *M, double *Mi );
void invertRT4( float *M, float *Mi );
void transformRT3(double *M1, float *V, float *R);
void transformRT3(float *M1, float *V, float *R);
void rotate3(double *M1, double *V, double *R);
void rotate3(float *M1, float *V, float *R);
void transformVector(double *CM, double *v, int nRows, int nCols, double *w);
int inverse3x3( double ma[9], double mr[9]);
int inverse3x3( float ma[9], float mr[9]);
void transpose(double ma[9], double mr[9]);
void submat3x3( double mr[16], double mb[9], int i, int j );
double det4x4( double mr[16] );
int inverse4x4( double ma[16], double mr[16] );
void getTrans(double m[3*4], double t[3]);
void transformVector3(double m[9], double v[3]);
void transformVector4(double m[12], double v[4]);
void buildTx(double t[3],double Tx[9]);
void buildTx(float *t, float *Tx);
void matrix_mult3(double m1[9], double m2[9], double res[9]);
void matrixMult3(float *m1, float *m2, float *res);
void matrixMult3(double *m1, double *m2, double *res);
void printMatrix3(double m[9]);
void rodrigues(double x, double y, double z, double m[9]);
void rodrigues(double x, double y, double z, double tx, double ty, double tz, double m[12]);
void matrix_from_euler(double xang, double yang, double zang, double m[9]);
void matrix_from_euler4(float xang, float yang, float zang, float *m);
void rot2euler(double *m, double *theta);
void pseudoInverse3x4(double *m, double *mr);
void matrixMultMxN(double *A, double *B, double *R, int nRows1, int nCols1, int nCols2);
void matrixMultMxN_diagonal(double *A, double *B, double *R, int nRows1, int nCols1, int nCols2);
void matrixMultMxN_diagonal(float *A, float *B, float *R, int nRows1, int nCols1, int nCols2);
void capZero(double *A, int cnt);
void capZero(float *A, int cnt);
void projectionMatrix(double *K, double *cameraMatrix, double *P);
bool pointInTriangle(float rx2, float ry2, float ax, float ay, float bx, float by, float cx, float cy);
void matrixMultVec4(double *M1, float *V, float *R);
void rot2Quaternion(float *m, int n, float *q);
void rot2Quaternion(double *m, int n, double *q);
void quaternion2Rot(float *q, float *m);
void quaternion2Rot(double *q, double *m);
void normalizeQuaternion(float *qr );
void normalizeQuaternion(double *qr );
void dumpMatrix(const char *str, const float *M, int rows, int cols);
void dumpMatrix(const char *str, const double *M, int rows, int cols);
void  poseDistance(float *dT, double *dist, double *angle);
float poseDistanceProjection(float *relT, float *K, float *testObj, int nVertices);
int generateGridObject(float **gridObj,int nStepsX, int nStepsY, int nLayers, float size, float viewDistanceMin, float zRange, float fovAngleX);
float quickMedian(float *arr, int n);
void double_precision4(float *m4, double *m4_dbl);
void float_precision4(double *m4, float *m4_flt);
float qdot(float *q1, float *q2);
void slerp(float *q1, float *q2, float t, float *qt);
void lerp(float *p1, float w1, float *p2, float w2, int n, float *pt);
void get3DPoint(float x, float y, float z, float *iK, float *xc, float *yc, float *zc);
void get3DRay(float x, float y, float *iK, float *xr, float *yr, float *zr);
// this initializes projection from current view k into reference view
void projectInit(float *dstK, float *T, float *P);
void projectInitZ(float *srcT, float *dstK, float *dstT, float *P, float *Tz);
void projectInitXYZ(float *srcT, float *dstK, float *dstT, float *P, float *Tx, float *Ty, float *Tz);
void projectFast(float *x3d, float *x2d, float *P);
void projectFastZ(float *x3d, float *x2d, float *z, float *P, float *Tz);
void projectFastXYZ(float *x3d, float *x2d, float *p3, float *P, float *Tx, float *Ty, float *Tz);
void distortPointCPU(float *pu, float *kc, float *K, float *pd);
void relativeTransform(float *srcT, float *dstT, float *T);
void normalizeRT4(float *T);
double dotProduct6CPU(double *a, double *b);
void matrixMultVec6CPU(double *A, double *x, double *r);
int solveQuadric(double *c, double *s);
int solveCubic(double *c, double *s);
int solveQuadric(double *c, double *s);


float quickMedian(float *arr, int n);

int analyzeConvergence(float *A, float *b, double tol, int maxSteps);

#endif //!__CMATH_H__

