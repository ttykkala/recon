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


#include <tracker/shapetracker.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <multithreading.h>
#include <tracker/basic_math.h>
#include <reconstruct/TrackFrame.h>
#include <calib/calib.h>
#include <multicore/multicore.h>
#include <cuda_funcs.h>

static CUTThread threadID;

//static const int geometryResidualSize = 8*1024;//8*1024;//40*1024;
// these constant scales are only to make sure that parameter vector X contains uniform magnitudes in optimization
static float optScaleIn = 1e-3f;
static float optScaleOut = 1e+3f;

#define MAX_BLOCKS_TO_REDUCE 4096

void ShapeTracker::setReferenceUpdating(bool flag) {
    referenceUpdatingFlag = flag;
}

void ShapeTracker::setFilteredReference(bool flag) {
    useRawMeasurementsForPoseEstimation = !flag;
}

bool ShapeTracker::getFilteredReference() {
    return !useRawMeasurementsForPoseEstimation;
}

ShapeTracker::ShapeTracker(int cudaDeviceID,int resoX, int resoY, Calibration *calib, float *TrelExtHost, float *TrelDevExt, float optScaleInExt, float optScaleOutExt, cudaStream_t stream) {
    this->deviceID = cudaDeviceID;
    this->texWidth = resoX;
    this->texHeight = resoY;
    this->m_incrementalMode = false;
    //this->rgbdFrame = NULL;
    //this->refFrame = NULL;
    this->debugImage = NULL;
    this->referenceExists  = false;
    this->referencePrepared = false;
    this->TidentityDev = NULL;
    this->TDev = TrelDevExt;
    this->iTDev = NULL;
    this->Text = TrelExtHost;
    this->fullResidualDev = NULL;
    this->residualDev = NULL;
    this->residual2Dev = NULL;
    this->residualMaskDev = NULL;
    this->residual6Dev = NULL;
    this->JtJDev = NULL;
    this->zCurrentDevExt = NULL;
    this->zReferenceDev = NULL;
    this->selectionPointsDev = NULL;
    this->refSelectionPointsDev = NULL;
    this->selectionColorsDev = NULL;
    this->selectionPoints = NULL;
    this->selectionColors = NULL;
    this->refSelectionPoints = NULL;
    this->calibrationUpdateSignaled = false;
    this->useRawMeasurementsForPoseEstimation = true;//false;//true;//false;
    this->poseDistanceThreshold = 125.0f;//350.0f;
    this->poseAngleThreshold = 12.5f;//10.0f;//20.0f;
    this->jacobianDev = NULL;
    this->referenceUpdateCount = 0;
    this->normalStatusDev = NULL;
    this->partialSumDev = NULL;
    this->partialSumIntDev = NULL;
    this->nLayers = 4;
    this->optScaleIn = optScaleInExt;
    this->optScaleOut = optScaleOutExt;
    this->geometryResidualSize = texWidth*texHeight;
    this->stream = stream;

    for (int i = 0; i < nLayers; i++) {
        refPointsDev[i] = NULL;
        curPointsDev[i] = NULL;
        debugImage1C[i] = NULL;
        refImage1C[i]   = NULL;
        selectionMaskCurDev[i] = NULL;
        selectionMaskRefDev[i] = NULL;
        selectionIndexDev[i]   = NULL;
        weightsDev[i] = NULL;
        nSelected[i] = 0;
    }
//    nIterations[0] = 10; nIterations[1] = 3; nIterations[2] = 1;
//    nIterations[0] = 1;  nIterations[1] = 2; nIterations[2] = 3; nIterations[3] = 5;
///    nIterations[0] = 2;  nIterations[1] = 2; nIterations[2] = 4; nIterations[3] = 0;
    nIterations[0] = 1;  nIterations[1] = 3; nIterations[2] = 10; nIterations[3] = 5;

    //    nIterations[0] = 0;  nIterations[1] = 0; nIterations[2] = 1; nIterations[3] = 0;

    // calib->setupCalibDataBuffer(texWidth,texHeight);
    float *calibDataExt = calib->getCalibData();
    this->calibData = new float[CALIB_SIZE];
    memcpy(this->calibData,calibDataExt,sizeof(float)*CALIB_SIZE);

  //  printf("shapetracker initialized!\n");
    resetTracking();
}

ShapeTracker::~ShapeTracker() {
    if (calibData) delete[] calibData;
  //  printf("shapetracker removed!\n");
}

int ShapeTracker::numReferenceUpdates() {
    return referenceUpdateCount;
}

int ShapeTracker::getDeviceID() {
    return deviceID;
}

void ShapeTracker::updateCalibration(float *calibData) {
    memcpy(this->calibData,calibData,sizeof(float)*CALIB_SIZE);
    cudaMemcpy(calibDataDev,calibData,CALIB_SIZE*sizeof(float),cudaMemcpyHostToDevice);
}

void ShapeTracker::resetTracking() {
    referenceExists = false;
    referencePrepared    = false;
    referenceUpdateCount = 0;
    printf("shapetracker reset.\n");
}

void ShapeTracker::setCurrentImage(float *zCurrentDev) {
    this->zCurrentDevExt = zCurrentDev;
    //this->rgbdFrame = currentFrame;
  //  printf("shape tracker receives current image!\n"); fflush(stdout);
    currentExists = true;
}

bool ShapeTracker::poseDifferenceExceeded(float *dT, float translationThreshold, float rotationThreshold) {
    double dx = dT[3];
    double dy = dT[7];
    double dz = dT[11];
    float dist = (float)sqrt(dx*dx+dy*dy+dz*dz+1e-12f);
    if (dist > translationThreshold) return true;
    float angle = 0;
    // check identity matrix as a special case:
    if ((fabs(dT[0]-1.0f) > 1e-8) || (fabs(dT[5]-1.0f) > 1e-8) || (fabs(dT[10]-1.0f) > 1e-8)) {
        float q[4];
        rot2Quaternion(dT, 4, q);
        normalizeQuaternion(q);
        double ca = (double)q[0];
        angle = rad2deg(acos(ca)) * 2.0f;
    }
    if (angle > rotationThreshold) return true;
    return false;
}

bool ShapeTracker::referenceOutdated() {
    if (!referenceExists) return true;
    if (m_incrementalMode) return true;
    if (referenceUpdatingFlag && poseDifferenceExceeded(&Text[0],poseDistanceThreshold,poseAngleThreshold)) return true;
    return false;
}

void ShapeTracker::setReferencePoints(float4 *newRefPointsDev, float4 *newRefPointNormalsDev, cudaArray *newRefFrame) {
    if (useRawMeasurementsForPoseEstimation && newRefFrame == NULL) { printf("shapetracker: bad reference given!\n"); fflush(stdout); return; }
    if (!useRawMeasurementsForPoseEstimation && (newRefPointsDev == NULL || newRefPointNormalsDev == NULL)) { printf("shapetracker: bad reference given!\n"); fflush(stdout); return; }

    if (!useRawMeasurementsForPoseEstimation) {
        generateOrientedPoints6Cuda(newRefPointsDev,newRefPointNormalsDev,refPointsDev[0], selectionMaskRefDev[0],calibDataDev, texWidth,texHeight, stream);
    } else {
        //checkCudaErrors(cudaMemcpy2DArrayToArray(refFrame,0,0,newRefFrame,0,0,texWidth*sizeof(float)*4,texHeight,cudaMemcpyDeviceToDevice));
        rgbdTex2DepthCuda(newRefFrame, texWidth,texHeight, zReferenceDev,  calibDataDev,false,stream);
        convertZmapToXYZ6Cuda(zReferenceDev,refPointsDev[0],selectionMaskRefDev[0],calibDataDev,texWidth,texHeight,stream);
        computeNormals6Cuda(refPointsDev[0],selectionMaskRefDev[0],texWidth,texHeight,stream);
        cudaStreamSynchronize(stream);
    }
    referenceExists   = true;
    referencePrepared = false;
    referenceUpdateCount++;
}

bool ShapeTracker::isReferencePrepared() {
    return referencePrepared;
}

void ShapeTracker::lock() {

}

void ShapeTracker::unlock() {

}

void ShapeTracker::updateRefImages(float **xyzMapDev, int nLayers) {
    for (int i = 0; i < nLayers; i++) {
        pointCloud6ToDepthCuda(xyzMapDev[i], texWidth>>i,texHeight>>i, refImage1C[i],calibDataDev,true,stream);
    }
}

void ShapeTracker::updateDebugDiffImages(float **xyzMapDev1,int **select1Dev, float *T, float *calibDataDev, float **xyzMapDev2, int **select2Dev, int nLayers) {
    for (int i = 0; i < nLayers; i++) {
        pointCloud6DiffCuda(xyzMapDev1[i], select1Dev[i], T, calibDataDev, xyzMapDev2[i], select2Dev[i], texWidth>>i,texHeight>>i, i, debugImage1C[i],true,stream);
    }
}

void ShapeTracker::updateErrorImage() {
    updateDebugDiffImages(refPointsDev,selectionMaskRefDev,TDev,calibDataDev,curPointsDev,selectionMaskCurDev,nLayers);
}


void ShapeTracker::setIncrementalMode(bool flag) {
    m_incrementalMode = flag;
}

void ShapeTracker::clearDebugImages() {
    for (int i = 0; i < nLayers; i++) {
        cudaMemsetAsync(debugImage1C[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(float),stream);
        cudaMemsetAsync(refImage1C[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(float),stream);
    }
}


void ShapeTracker::generatePyramids(float **xyzMapDev, int **selectionMasks, int nLayers) {
    for (int i = 1; i < nLayers; i++) {
        downSampleCloud6Cuda(xyzMapDev[i-1],selectionMasks[i-1],xyzMapDev[i],selectionMasks[i],(texWidth>>i),(texHeight>>i),stream);
    }
}

void ShapeTracker::prepareCurrent() {
//    rgbdTex2DepthCuda(cudaArray *rgbdFrame, int w, int h, float *zMap, float *calibDataDev, bool normalizeDepth);
    //rgbdTex2DepthCuda(rgbdFrame,texWidth,texHeight,zCurrentDev, calibDataDev,false);
    genPointCloud6Cuda(zCurrentDevExt,texWidth,texHeight,curPointsDev[0],selectionMaskCurDev[0],calibDataDev,stream);
 //   pointCloud6ToDepthCuda(curPointsDev[0], texWidth, texHeight, debugImage1C[0], calibDataDev, true);
    /* note: current normals are not used for anything!
    computeNormals6Cuda(curPointsDev[0],selectionMaskCurDev[0],texWidth,texHeight,stream);
    */
    generatePyramids(curPointsDev,selectionMaskCurDev,nLayers);
    //updateDebugImages(curPointsDev, nLayers);
//    printf("preparing shape tracker current!\n"); fflush(stdout);
}

void ShapeTracker::prepareReference() {

    if (!referenceExists || referencePrepared) return;
    // clear relative pose
    cudaMemcpyAsync((void*)TDev,  (void*)TidentityDev, 16*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    //identity4x4(&Tinc[0]);
    if (Text != NULL) {
        identity4x4(Text);
    } else {
        printf("Text == NULL!\n"); fflush(stdout);
    }
    generatePyramids(&refPointsDev[0],selectionMaskRefDev,nLayers);
    updateRefImages(refPointsDev, nLayers);
    referencePrepared = true;
  //  printf("shape reference prepared!\n"); fflush(stdout);
}

void ShapeTracker::listSelected(int layer, float **selected, float **refSelected, float **color, int *nSelect) {
    *nSelect = 0;
    *selected = NULL;
    *refSelected = NULL;
    *color = NULL;
    if (!currentExists || !referencePrepared) return;

//    listDepthSelectedCuda(refPointsDev[0], selectionMaskRefDev[0], texWidth, texHeight, T1Dev, T2Dev, calibDataDev, selectionPointsDev, selectionColorsDev);
//    icpResidualCuda(curPointsDev[0],selectionMaskCurDev[0],texWidth,texHeight,T1Dev,T2Dev,calibDataDev,refPointsDev[0],selectionMaskRefDev[0],0,fullResidualDev,residualMaskDev);
 /*   selectValidIndex(residualMaskDev,texWidth*texHeight,partialSumIntDev,MAX_BLOCKS_TO_REDUCE,selectionIndexDev[0]);
    cudaThreadSynchronize();
    int totalSelected = 0;
    cudaMemcpy(&totalSelected,partialSumIntDev,sizeof(int),cudaMemcpyDeviceToHost);
    printf("totalSelected:%d\n",totalSelected); fflush(stdout);*/
/*
    if (totalSelected > 0) {
        packIndex(selectionIndexDev[0],totalSelected,fullResidualDev,residualDev,residual2Dev);
        generateStudentTWeights(residual2Dev,totalSelected,partialSumDev,MAX_BLOCKS_TO_REDUCE,selectionColorsDev);
        listDepthSelectedCuda(curPointsDev[0], selectionIndexDev[0], totalSelected, T1Dev, T2Dev, calibDataDev, selectionPointsDev, weightsDev);//selectionColorsDev);//weightsDev);
        cudaMemcpy(selectionPoints,selectionPointsDev,2*sizeof(float)*totalSelected,cudaMemcpyDeviceToHost);
        cudaMemcpy(selectionColors,selectionColorsDev,sizeof(float)*totalSelected,cudaMemcpyDeviceToHost);

        *nSelect  = totalSelected;
        *selected = selectionPoints;
        *color    = selectionColors;

    }*/
    if (nSelected[layer] > 0) {
        cudaMemcpyAsync(selectionColorsDev,weightsDev[layer],sizeof(float)*nSelected[layer],cudaMemcpyDeviceToDevice,stream);
  //      invertPoseCuda(TDev,iTDev,1,stream);
        // note: the following 2 commands: weigtsDev is written garbage:
        listDepthSelectedCuda(refPointsDev[layer], selectionIndexDev[layer], nSelected[layer], TidentityDev, calibDataDev, refSelectionPointsDev, weightsDev[layer], stream);//selectionColorsDev);//weightsDev);
        listDepthSelectedCuda(refPointsDev[layer], selectionIndexDev[layer], nSelected[layer], TDev,         calibDataDev, selectionPointsDev,    weightsDev[layer], stream);//selectionColorsDev);//weightsDev);

        //updateDebugDiffImages(refPointsDev,selectionMaskRefDev,TDev,calibDataDev,curPointsDev,selectionMaskCurDev,nLayers);
        //updateDebugDiffImages(curPointsDev,selectionMaskCurDev,TDev,calibDataDev,refPointsDev,selectionMaskRefDev,nLayers);
        cudaMemcpyAsync(selectionPoints,selectionPointsDev,      2*sizeof(float)*nSelected[layer],cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(refSelectionPoints,refSelectionPointsDev,2*sizeof(float)*nSelected[layer],cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(selectionColors,selectionColorsDev,sizeof(float)*nSelected[layer],cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        *nSelect  = nSelected[layer];
        *selected = selectionPoints;
        *refSelected = refSelectionPoints;
        *color    = selectionColors;
    }
    return;
}

void ShapeTracker::setIterationCounts(int *nIter) {
    for (int i = 0; i < nLayers; i++) nIterations[i] = nIter[i];
}

// for selected points:
    // parallel: warp points + interpolate -> residual
    // 1thread: m-estimator weights
    // parallel: Jpre -> reweight jacobian -> Jw
    // parallel: Jw^T * Jw, Jw^T * residual (system reduction)
    // 1thread: inv6x6, solve x
    // 1thread: expm -> Trel update


int getPaddedLength(int n) {
    int nPadded = 1024*int(n/1024); if (n%1024 != 0) nPadded += 1024;
    return nPadded;
}


bool ShapeTracker::updateLinearSystem(int layer, int iteration) {
    // assume the main loop iterates through layers and iterations
    if (iteration == 0) {
        icpResidualMaskCuda2(refPointsDev[layer],selectionMaskRefDev[layer],texWidth>>layer,texHeight>>layer,TDev,calibDataDev,curPointsDev[layer],selectionMaskCurDev[layer],layer,fullResidualDev,residualMaskDev,stream);
        //        icpResidualMaskCuda(curPointsDev[layer],selectionMaskCurDev[layer],texWidth>>i,texHeight>>i,TDev,calibDataDev,refPointsDev[layer],selectionMaskRefDev[layer],layer,fullResidualDev,residualMaskDev,stream);
        selectValidIndex(residualMaskDev,(texWidth>>layer)*(texHeight>>layer),partialSumIntDev,MAX_BLOCKS_TO_REDUCE,selectionIndexDev[layer],stream);
        cudaMemcpyAsync(&nSelected[layer],partialSumIntDev,sizeof(int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        if (nSelected[layer] <= 0) { printf("shapetracker optimization layer %d: not enough points!\n",layer); fflush(stdout); return false;}
        packIndex(selectionIndexDev[layer],nSelected[layer],fullResidualDev,residualDev,residual2Dev,stream);
        generateStudentTWeights(residual2Dev,nSelected[layer],partialSumDev,MAX_BLOCKS_TO_REDUCE, weightsDev[layer],stream);

        // padded residual / jacobian length:
        int nPadded = getPaddedLength(nSelected[layer]);
        cudaMemsetAsync(residualDev,0,sizeof(float)*nPadded,stream);
        cudaMemsetAsync(jacobianDev,0,sizeof(float)*6*nPadded,stream);
    }
    // padded residual / jacobian length:
    int nPadded = getPaddedLength(nSelected[layer]);
    icpResidualCuda2(refPointsDev[layer],selectionIndexDev[layer], nSelected[layer], texWidth>>layer, texHeight>>layer, layer, TDev, calibDataDev, curPointsDev[layer], selectionMaskCurDev[layer], nPadded,residualDev, jacobianDev, weightsDev[layer],optScaleIn,stream);
    //           icpResidualCuda(curPointsDev[layer],selectionIndexDev[layer], nSelected[layer], texWidth>>i, texHeight>>i, i, TDev, calibDataDev, refPointsDev[layer], selectionMaskRefDev[layer], nPadded[layer],residualDev, jacobianDev, weightsDev[layer],stream);
    JTJCuda(jacobianDev,nPadded,JtJDev,stream);
    //printf("nPadded[%d]:%d\n",i,nPadded[layer]); fflush(stdout);
    JTresidualCuda(jacobianDev,residualDev,nPadded,residual6Dev,stream);
    return true;
}

void ShapeTracker::getIterationCounts(int *niter) {
    memcpy(niter,&nIterations[0],sizeof(int)*4);
}

void ShapeTracker::estimateIncrement() {
    solveMotionCuda(JtJDev,residual6Dev,TDev,optScaleOut,stream);
}

void ShapeTracker::linearFuse(float *JtJDevExt, float *residual6DevExt, int nPixels, int layer, float lambda) {
    linearFuseCuda(JtJDevExt,residual6DevExt, (1-lambda), nPixels, JtJDev, residual6Dev,lambda,nSelected[layer],stream);
}

int ShapeTracker::getRecentSelected(int layer) {
    return nSelected[layer];
}

void ShapeTracker::copyLinearSystemTo(float *JtJDevExt, float *residual6DevExt) {
    cudaMemcpyAsync(JtJDevExt,JtJDev,sizeof(float)*36,cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(residual6DevExt,residual6Dev,sizeof(float)*6,cudaMemcpyDeviceToDevice,stream);
}


void ShapeTracker::signalCalibrationUpdate(float *calibDataExt) {
    memcpy(this->calibData,calibDataExt,sizeof(float)*CALIB_SIZE);
    calibrationUpdateSignaled = true;
}

bool ShapeTracker::readyToTrack() {
    return (referenceExists && currentExists);
}

float4 *ShapeTracker::getDebugImage() {
    return debugImage;
}

float *ShapeTracker::getDebugImage1C(int layer) {
    return debugImage1C[layer];
}

float *ShapeTracker::getRefImage1C(int layer) {
    return refImage1C[layer];
}

void ShapeTracker::allocMemory() {
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
    //checkCudaErrors(cudaMallocArray(&this->refFrame,   &channelDesc, texWidth, texHeight));
    checkCudaErrors(cudaMalloc(&this->debugImage,      texWidth*texHeight*sizeof(float4)));
    checkCudaErrors(cudaMalloc(&this->normalStatusDev,  texWidth*texHeight*sizeof(char)));

    if (texWidth == 640) {
        nLayers = 4;
    } else if (texWidth == 320) {
        nLayers = 3;
    } else {
        // wtf?
        nLayers = 3;
    }
    printf("nlayers: %d\n",nLayers);
    for (int i = 0; i < nLayers; i++) {
        checkCudaErrors(cudaMalloc((void **)&refPointsDev[i], (texWidth>>i)*(texHeight>>i)*sizeof(float)*6));
        checkCudaErrors(cudaMalloc((void **)&curPointsDev[i], (texWidth>>i)*(texHeight>>i)*sizeof(float)*6));
        checkCudaErrors(cudaMalloc(&this->debugImage1C[i],    (texWidth>>i)*(texHeight>>i)*sizeof(float)));       cudaMemset(debugImage1C[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(float));
        checkCudaErrors(cudaMalloc(&this->refImage1C[i],      (texWidth>>i)*(texHeight>>i)*sizeof(float)));       cudaMemset(refImage1C[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(float));
        checkCudaErrors(cudaMalloc((void **)&selectionMaskCurDev[i],  (texWidth>>i)*(texHeight>>i)*sizeof(int))); cudaMemset(selectionMaskCurDev[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(int));
        checkCudaErrors(cudaMalloc((void **)&selectionMaskRefDev[i],  (texWidth>>i)*(texHeight>>i)*sizeof(int))); cudaMemset(selectionMaskRefDev[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(int));
        checkCudaErrors(cudaMalloc((void **)&selectionIndexDev[i],    (texWidth>>i)*(texHeight>>i)*sizeof(int))); cudaMemset(selectionIndexDev[i],0,(texWidth>>i)*(texHeight>>i)*sizeof(int));
        checkCudaErrors(cudaMalloc((void **)&weightsDev[i], (texWidth>>i)*(texHeight>>i)*sizeof(float)));         cudaMemset(weightsDev[i], 0,(texWidth>>i)*(texHeight>>i)*sizeof(float));
    }
    // allocate worst-case amount of memory for jacobian scratch
    cudaMalloc((void **)&jacobianDev, 6*geometryResidualSize*sizeof(float)); cudaMemset(jacobianDev,0,6*geometryResidualSize*sizeof(float));

    // temporary debug image content
    float4 *ptr = new float4[texWidth*texHeight];
    for (int i = 0; i < texWidth*texHeight; i++) {
        ptr[i].x = 1.0f;
        ptr[i].y = 0.0f;
        ptr[i].z = 0.0f;
        ptr[i].w = 1.0f;
    }
    cudaMemcpy(debugImage,ptr,texWidth*texHeight*sizeof(float4),cudaMemcpyHostToDevice);
    cudaMalloc(&calibDataDev,CALIB_SIZE*sizeof(float));
    cudaMemcpy(calibDataDev,calibData,CALIB_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    delete[] ptr;

    initCudaDotProduct();


    cudaMalloc((void **)&residualDev,  geometryResidualSize*sizeof(float));            cudaMemset(residualDev,0,geometryResidualSize*sizeof(float));
    cudaMalloc((void **)&residual2Dev, geometryResidualSize*sizeof(float));            cudaMemset(residual2Dev,0,geometryResidualSize*sizeof(float));
    cudaMalloc((void **)&residualMaskDev, geometryResidualSize*sizeof(int));           cudaMemset(residualMaskDev,0,geometryResidualSize*sizeof(int));
    cudaMalloc((void **)&fullResidualDev,  geometryResidualSize*sizeof(float));        cudaMemset(fullResidualDev,0,geometryResidualSize*sizeof(float));
    cudaMalloc((void **)&residual6Dev, 6*sizeof(float));                               cudaMemset(residual6Dev,0,6*sizeof(float));
    cudaMalloc((void **)&JtJDev, 6*6*sizeof(float));                                   cudaMemset(JtJDev,0,6*6*sizeof(float));
    cudaMalloc((void **)&selectionPointsDev,    geometryResidualSize*sizeof(float)*2); cudaMemset(selectionPointsDev,0,geometryResidualSize*sizeof(float)*2);
    cudaMalloc((void **)&refSelectionPointsDev, geometryResidualSize*sizeof(float)*2); cudaMemset(refSelectionPointsDev,0,geometryResidualSize*sizeof(float)*2);
    cudaMalloc((void **)&selectionColorsDev, geometryResidualSize*sizeof(float));      cudaMemset(selectionColorsDev,0,geometryResidualSize*sizeof(float));
    //cudaMalloc((void**)&zCurrentDev,texWidth*texHeight*sizeof(float));               cudaMemset(zCurrentDev,0,texWidth*texHeight*sizeof(float));
    cudaMalloc((void**)&zReferenceDev,texWidth*texHeight*sizeof(float));               cudaMemset(zReferenceDev,0,texWidth*texHeight*sizeof(float));
    cudaMalloc((void**)&partialSumDev,MAX_BLOCKS_TO_REDUCE*sizeof(float));             cudaMemset(partialSumDev,0,MAX_BLOCKS_TO_REDUCE*sizeof(float));
    cudaMalloc((void**)&partialSumIntDev,MAX_BLOCKS_TO_REDUCE*sizeof(int));            cudaMemset(partialSumIntDev,0,MAX_BLOCKS_TO_REDUCE*sizeof(int));

    float identityMatrix[16];
    identity4x4(&identityMatrix[0]);
    //identity4x4(&T[0]);
//    identity4x4(&Tbase[0]);
    //identity4x4(&Tinc[0]);
    cudaMalloc((void **)&TidentityDev, 16 * sizeof(float));  cudaMemcpy((void*)TidentityDev,&identityMatrix[0],16*sizeof(float),cudaMemcpyHostToDevice);
    // store T1 and T2 into neighboring memory slots
 //   cudaMalloc((void **)&TDev,  16 * sizeof(float));
    cudaMalloc((void **)&iTDev,  16 * sizeof(float));
   // cudaMemcpy((void*)TDev, (void*)TidentityDev,16*sizeof(float),   cudaMemcpyDeviceToDevice);
//    cudaMalloc((void **)&T2Dev,  16 * sizeof(float));

    selectionPoints = new float[texWidth*texHeight*2];
    refSelectionPoints = new float[texWidth*texHeight*2];
    selectionColors = new float[texWidth*texHeight];
   }

void ShapeTracker::freeMemory() {
    if (normalStatusDev) cudaFree(normalStatusDev);
     //if (refFrame) cudaFreeArray(refFrame);
    if (selectionPoints) delete[] selectionPoints;
    if (refSelectionPoints) delete[] refSelectionPoints;
    if (selectionColors) delete[] selectionColors;
    if (debugImage) cudaFree(debugImage);     
    for (int i = 0; i < 4; i++) {
        if (refPointsDev[i]) cudaFree(refPointsDev[i]);
        if (curPointsDev[i]) cudaFree(curPointsDev[i]);
        if (debugImage1C[i]) cudaFree(debugImage1C[i]);
        if (refImage1C[i])   cudaFree(refImage1C[i]);
        if (selectionMaskCurDev[i]) cudaFree(selectionMaskCurDev[i]);
        if (selectionMaskRefDev[i]) cudaFree(selectionMaskRefDev[i]);
        if (selectionIndexDev[i])   cudaFree(selectionIndexDev[i]);
        if (weightsDev[i])          cudaFree(weightsDev[i]);
    }
    if (calibDataDev) cudaFree(calibDataDev);
    if (fullResidualDev) cudaFree(fullResidualDev);
    if (residualDev) cudaFree(residualDev);
    if (residual2Dev) cudaFree(residual2Dev);
    if (residualMaskDev) cudaFree(residualMaskDev);
    if (residual6Dev) cudaFree(residual6Dev);
    if (JtJDev) cudaFree(JtJDev);
    if (selectionPointsDev)    cudaFree(selectionPointsDev);
    if (refSelectionPointsDev) cudaFree(refSelectionPointsDev);
    if (selectionColorsDev) cudaFree(selectionColorsDev);
    //if (zCurrentDev) cudaFree(zCurrentDev);
    if (zReferenceDev) cudaFree(zReferenceDev);
    if (jacobianDev) cudaFree(jacobianDev);
  //  if (TDev) cudaFree(TDev);
    if (iTDev) cudaFree(iTDev);
//    if (T2Dev) cudaFree(T2Dev);
    if (TidentityDev) cudaFree(TidentityDev);
    if (partialSumDev) cudaFree(partialSumDev);
    if (partialSumIntDev) cudaFree(partialSumIntDev);
    releaseCudaDotProduct();
}

void ShapeTracker::release() {
    freeMemory();
    printf("released ShapeTracker!\n");
}
