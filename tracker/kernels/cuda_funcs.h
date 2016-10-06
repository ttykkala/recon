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

#include <GL/glew.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <rendering/VertexBuffer2.h>

#include "histogram_common.h"

extern "C" void rgb2GrayCuda(Image2 *rgbImage, Image2 *grayImage);
extern "C" void rgbTex2GrayCuda(cudaArray *rgbdImage, Image2 *grayImage, float *zMap, float *calibDataDev, cudaStream_t stream=0);
extern "C" void rgbdTex2DepthCuda(cudaArray *rgbdFrame, int w, int h, float *zMap, float *calibDataDev, bool normalizeDepth, cudaStream_t stream=0);
extern "C" void xyz2DepthCuda(float4 *xyzDev, int width, int height, float *zMap, float *calibDataDev, bool normalizeDepth);
extern "C" void undistortCuda(Image2 *srcImage, Image2 *dstImage, float *calibDataDev);
extern "C" void undistortRGBCuda(Image2 *srcImage, Image2 *dstImage, float *calibDataDev);
extern "C" void d2ZCuda(unsigned short *disparity16U, Image2 *zImage, float *calibDataDev, float xOff, float yOff);
extern "C" void d2ZCudaHdr(float *disparityHdr, Image2 *zImage, float *calibDataDev, float xOff, float yOff);
extern "C" void convertZmapToXYZCuda(float *zReferenceDev, float4 *refPointsDev, float *calibDataDev, int width, int height, cudaStream_t stream=0);
extern "C" void convertZmapToXYZ6Cuda(float *zReferenceDev, float *refPointsDev, int *selectionMask, float *calibDataDev, int width, int height, cudaStream_t stream=0);
extern "C" void xyz2CloudCuda(float4 *xyzImage, float *calibDataDev, VertexBuffer2 *vbuffer, ImagePyramid2 *grayPyramid, cudaStream_t stream=0);
extern "C" void xyz2DiffCuda(VertexBuffer2 *vbuffer, int vWidth, int vHeight, float *calibDataDev, float *TrelDev, float *diffImage, int width, int height, int layer, ImagePyramid2 *grayPyramidCur, cudaStream_t stream=0);
extern "C" void z2CloudCuda(Image2 *zImageIR, float *calibDataDev, VertexBuffer2 *vbuffer, Image2 *rgbImage, ImagePyramid2 *grayPyramid, Image2 *zImage, bool computeGradient);
extern "C" void genPointCloud6Cuda(float *zCurrentDev,int texWidth,int texHeigth,float *curPointsDev,int *selectionMaskDev, float *calibDataDev, cudaStream_t stream=0);
//extern "C" void xyz6Select(float *refPointsDev,int width, int height, int *selectionMaskRefDev);
extern "C" void downSampleCloud6Cuda(float *xyz6DevSrc,int *maskSrc, float *xyz6DevDst, int *maskDst,int dstTexWidth,int dstTexHeight, cudaStream_t stream=0);
extern "C" void pointCloud6DiffCuda(float *xyz6Dev1, int *selectDev1, float *T, float *calibDataDev, float *xyz6Dev2, int *selectDev2, int width, int height, int layer, float *diffImage,bool normalize, cudaStream_t stream=0);
extern "C" void pointCloud6ToDepthCuda(float *xyz6Dev, int width, int height, float *zMap, float *calibDataDev, bool normalizeDepth, cudaStream_t stream=0);
extern "C" void generateOrientedPoints6Cuda(float4 *newRefPointsDev, float4 *newRefPointNormalsDev, float *refPointsDev, int *selectionMask, float *calibDataDev, int width, int height, cudaStream_t stream = 0);
extern "C" void gradientCuda(ImagePyramid2 &pyramid, ImagePyramid2 &gradXPyramid, ImagePyramid2 &gradYPyramid, int baseLayer);
extern "C" void convert2FloatCuda(Image2 *rgbInput, Image2 *imRGB);
extern "C" void convertToHDRCuda(Image2 *imRGB, Image2 *imRGBHDR);
extern "C" void downSample2Pyramid(ImagePyramid2 &pyramid, cudaStream_t stream=0);
extern "C" void downSample2Cuda(Image2 *hires, Image2 *lowres);
extern "C" void undistortDisparityCuda(unsigned short *disparity16U, float *uPtr, float *calibDataDev, int width, int height, cudaStream_t stream = 0);
extern "C" void generateOrientedPointsCuda(float *depthMapDev,float4 *refPointsDev,float *calibDataDev,unsigned char *normalStatusDev);
extern "C" void setNormalsCuda(VertexBuffer2 *vbuffer, float *normalImage, float scale);
extern "C" void extractGradientMagnitudes(VertexBuffer2 *vbuffer, float *gradientScratchDev, cudaStream_t stream=0);
extern "C" double cudaHista(float *src, float *hist, int length, int bins, float *d_hist, cudaStream_t stream=0);
extern "C" double cudaHistb(float *src, float *hist, int length, int bins);
extern "C" void computeNormals6Cuda(float *refPointsDev, int *selectionMask, int texWidth,int texHeight, cudaStream_t stream=0);
extern "C" void addVertexAttributesCuda(Image2 *zImage, float *calibDataDev, VertexBuffer2 *vbuffer, ImagePyramid2 *grayPyramid, cudaStream_t stream=0);
extern "C" void addVertexAttributesWithoutNormalsCuda(float *calibDataDev, VertexBuffer2 *vbuffer, ImagePyramid2 *grayPyramid, cudaStream_t stream=0);
extern "C" void listSelectedRefCuda(VertexBuffer2 *vbuffer, float *selectionPointsDev, float *selectionColorsDev);
extern "C" void listSelectedCurCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *TrelDev, float *selectionPointsDev, float *selectionColorsDev, cudaStream_t stream);
extern "C" void listDepthSelectedCuda(float *curPointsDev, int *select, int nSelect, float *TDev, float *calibDataDev, float *selectionPointsDev, float *selectionColorsDev, cudaStream_t stream=0);
extern "C" void icpResidualMaskCuda(float *curPointsDev,int *selectionMask,int width,int height,float *TDev,float *calibDataDev,float *refPointsDev, int *selectionMaskRef, int layer, float *residualDev, int *residualMaskDev, cudaStream_t stream=0);
extern "C" void icpResidualCuda(float *curPointsDev, int *select, int nSelect, int width, int height, int layer, float *TDev, float *calibDataDev, float *refPointsDev, int *refMask, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev, cudaStream_t stream=0);
extern "C" void icpResidualMaskCuda2(float *refPointsDev,int *selectionMask,int width,int height,float *TDev,float *calibDataDev,float *curPointsDev, int *selectionMaskRef, int layer, float *residualDev, int *residualMaskDev, cudaStream_t stream=0);
extern "C" void icpResidualCuda2(float *refPointsDev, int *select, int nSelect, int width, int height, int layer, float *TDev, float *calibDataDev, float *curPointsDev, int *curMask, int nPadded, float *residualDev, float *jacobianDev, float *weightsDev, float optScaleIn, cudaStream_t stream=0);
extern "C" void generateStudentTWeights(float *residualDev,int nElems,float *partialSumDev, int maxBlocks, float *weightsDev,cudaStream_t stream=0);
extern "C" void selectValidIndex(int *residualMaskDev, int nElems,int *partialSumIntDev, int maxBlocks, int *selectionIndexDev, cudaStream_t stream=0);
extern "C" void packIndex(int *selectionIndexDev,int totalSelected,float *fullResidualDev,float *residualDev,float *residual2Dev, cudaStream_t stream=0);
extern "C" void filterIndices3(VertexBuffer2 *vbuffer, float *gradientData, uint *histogramDev, int pixelSelectionAmount);
extern "C" void filterIndices4(VertexBuffer2 *vbuffer, float *gradientData, float *histogramDev, int pixelSelectionAmount, int nbins, cudaStream_t stream=0);
extern "C" void warpPoints(VertexBuffer2 *vbuffer, float *weightsDev, float *T, float *calibDataDev, VertexBuffer2 *baseBuf,ImagePyramid2 *grayPyramid);
extern "C" void interpolateResidual(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, ImagePyramid2 &grayPyramid, int layer, float *residual, float *zweightsDev);
extern "C" void interpolateResidual2(VertexBuffer2 *vbuffer, float *T, float *calibDataDev, ImagePyramid2 &grayPyramid, int layer, float *zCurrentDev, float *zWeightsDev, float *residual, cudaStream_t cudaStream);
extern "C" void precomputeJacobianCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3, float optScaleIn);
extern "C" void precomputeJacobian4Cuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3, float *jacobianDev4, float optScaleIn, cudaStream_t stream=0);
//extern "C" void precomputeJacobianUncompressedCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3);
extern "C" void dotProductCuda(float *vecA, float *vecB, int count, float *result, cudaStream_t stream=0);
extern "C" void JTJCuda(float *JT,int count, float *JtJDev, cudaStream_t stream=0);
extern "C" void JTresidualCuda(float *JT, float *residual, int len, float *result6, cudaStream_t stream=0);
extern "C" void initCudaDotProduct();
extern "C" void releaseCudaDotProduct();
extern "C" void solveMotionCuda(float *JtJDev, float *x, float *b, float optScaleOut, cudaStream_t stream=0);
extern "C" void warpBase(VertexBuffer2 *vbuffer,float *T);
extern "C" void matrixMult4Cuda(float *A, float *B, float *C);
extern "C" void matrixMult4NormalizedCuda(float *TrelDev, float *TabsDev, float *TnextDev);
extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev, float *weightedResidualDev, cudaStream_t stream=0);
extern "C" void weightJacobian(float *jacobianTDev, float *weights, int count, float *weightedJacobianTDev, cudaStream_t stream=0);
extern "C" void compressVertexBuffer(VertexBuffer2 *vbufferSrc, VertexBuffer2 *vbufferDst, bool rgbVisualization = false);
extern "C" void compressVertexBuffer2(int *indicesExt,float *verticesExt,int pixelSelectionAmount,int srcStride, VertexBuffer2 *vbuffer);
extern "C" void collectPointsCuda(VertexBuffer2 *vbufferSrc, float *Tsrc, int collectedPoints256, VertexBuffer2 *vbufferDst, float *Tdst);
extern "C" void collectPointsCuda2(VertexBuffer2 *vbufferSrc, float *Tsrc,  int collectedPoints256, float *vertexImageDev, float *Tdst);
extern "C" void setPointIntensityCuda(VertexBuffer2 *vbuffer, float *Tsrc,float *Tdst,ImagePyramid2 *grayPyramid);
extern "C" void collectPointsIntoImageCuda(VertexBuffer2 *vbufferSrc, float *Tsrc, int collectedPoints256, float *vertexImageDev, float *Tdst, int width, int height, float *calibDataDev);
extern "C" void invertPoseCuda(float *A, float *iA, int N, cudaStream_t stream=0);
extern "C" void convertMatrixToPosAxisAngleCuda(float *A, float *posAxisAngle, int N);
extern "C" void filterPoseCuda(float *posAxisAngle, float *weightsDev, int N, float *T);
extern "C" void filterDepthIIR(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, float *weightsDev, int width, int height, float weightThreshold);
extern "C" void vectorProductCuda(float *vecA,float *vecB,int count,float *result, cudaStream_t stream=0);
//extern "C" void cudaWarp(float *pR, float *pL, int nPoints, float *matrixData, float *outPoints);
//extern "C" void	cudaPointLineWarp(float *pR, float *linesL0, int nPoints,float *matrixData,float *linesL1);
extern "C" void linearFuseCuda(float *JtJDevExt,float *residual6DevExt, float weight1, int N1, float *JtJDev, float *residual6Dev, float weight2, int N2, cudaStream_t stream=0);

#include "hostUtils.h"
