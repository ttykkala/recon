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

#include <tracker/phototracker.h>
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
#include <tracker/basic_math.h>
#include <reconstruct/TrackFrame.h>
#include <calib/calib.h>
#include <multicore/multicore.h>
#include <cuda_funcs.h>
#include "devices.cuh"
#include <tracker/expm.h>

static CUTThread threadID;

static const int photometricResidualSize = 8*1024;
// these constant scales are only to make sure that parameter vector X contains uniform magnitudes in optimization
const float optScaleIn = 1e-3f;
const float optScaleOut = 1e+3f;
static PhotoTracker *photoTracker = NULL;

typedef struct {
    int deviceID;
    int width;
    int height;
    Calibration *calib;
} INPUT_PARAMS;


void PhotoTracker::setReferenceUpdating(bool flag) {
    referenceUpdatingFlag = flag;
/*    if (!referenceUpdatingFlag) {
        referenceExists = false;
        referencePrepared = false;
    }*/
    if (shapeTracker) shapeTracker->setReferenceUpdating(flag);
}

void PhotoTracker::setFilteredReference(bool flag) {
    // always use raw measurements with photometric side
    if (shapeTracker) shapeTracker->setFilteredReference(flag);
}

void PhotoTracker::setIncrementalMode(bool flag) {
    m_incrementalMode = flag;
    // always use raw measurements with photometric side
    if (shapeTracker) shapeTracker->setIncrementalMode(flag);
}

bool PhotoTracker::getFilteredReference() {
    return shapeTracker->getFilteredReference();
}

PhotoTracker::PhotoTracker(int cudaDeviceID,barrierSync *barrier, int resoX, int resoY, Calibration *calib) {
    this->deviceID = cudaDeviceID;
    this->threadActiveFlag = false;
    this->barrier = barrier;
    this->estimateOnCPU=false;
    this->m_incrementalMode = false;
    this->texWidth = resoX;
    this->texHeight = resoY;
    //this->rgbdFrame = NULL;
    this->refFrame = NULL;
    this->debugImage = NULL;
    this->debugImage1C = NULL;
    this->debugRefImage1C = NULL;
    this->refPointsDev = NULL;
    this->referenceExists  = false;
    this->referencePrepared    = false;
    this->histogramFloatDev = NULL;
    this->gradientScratchDev = NULL;
    this->d_hist = NULL;
    this->TidentityDev = NULL;
    this->TrelDev = NULL;
    this->T = new float[16];
    this->Tinc = new double[16];
    this->residualDev = NULL;
    this->weightedJacobianTDev = NULL;
    this->weightedResidualDev = NULL;
    this->residual6Dev = NULL;
    this->weightsDev = NULL;
    this->zWeightsDev = NULL;
    this->JtJDev = NULL;
    this->zCurrentDev = NULL;
    this->zReferenceDev = NULL;
    this->selectionPointsDev = NULL;
    this->selectionColorsDev = NULL;
    this->refSelectionPointsDev = NULL;
    //this->refSelectionColorsDev = NULL;
    this->selectionPoints = NULL;
    this->selectionColors = NULL;
    this->refSelectionPoints = NULL;
    //this->refSelectionColors = NULL;
    this->calibrationUpdateSignaled = false;
    this->poseDistanceThreshold = 125.0f;//350.0f;
    this->poseAngleThreshold = 12.5f;//20.0f;
    this->nLayers = 4;
    this->shapeTracker = NULL;
    this->refFrameNumber = 0;
    this->mode = GEOMETRIC;
    this->stream = 0;
    this->lambda = 0.9f;
    this->saveCovariancesToDisk = false;//true;
    this->currentFrame = 0;

    sprintf(eigenFileName,"scratch/eigen.txt");
    remove(eigenFileName);
    identity4x4(T);
    identity4x4(Tinc);
    float identityMatrix[16];
    identity4x4(&identityMatrix[0]);
    cudaMalloc((void **)&TidentityDev, 16 * sizeof(float));  cudaMemcpy((void*)TidentityDev,&identityMatrix[0],16*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&TrelDev,  16 * sizeof(float));      cudaMemcpy((void*)TrelDev, (void*)TidentityDev,16*sizeof(float),cudaMemcpyDeviceToDevice);

    for (int i = 0; i < nLayers; i++) jacobianTDev[i] = NULL;
    nIterations[0] = 1;  nIterations[1] = 3; nIterations[2] = 10; nIterations[3] = 5;
    nIterationsBiObjective[0] = 1;  nIterationsBiObjective[1] = 3; nIterationsBiObjective[2] = 10; nIterationsBiObjective[3] = 5;

    float *calibDataExt = calib->getCalibData();
    this->calibData = new float[CALIB_SIZE];
    memcpy(this->calibData,calibDataExt,sizeof(float)*CALIB_SIZE);

    shapeTracker = new ShapeTracker(deviceID,texWidth,texHeight,calib,T,TrelDev,optScaleIn,optScaleOut,stream);
    resetTracking();
}

void PhotoTracker::setLambda(float lambda) {
    if (lambda < 0.0f) lambda = 0.0f;
    if (lambda > 1.0f) lambda = 1.0f;
    this->lambda = lambda;
    printf("phototracker: lambda set to %f\n",lambda); fflush(stdout);
}

float PhotoTracker::getLambda() {
    return this->lambda;
}


void PhotoTracker::setMode(TrackMode mode) {
    if (mode != this->mode) {
        if (mode == GEOMETRIC)   printf("mode switched (%d): geometric\n",  refFrameNumber); fflush(stdout);
        if (mode == PHOTOMETRIC) printf("mode switched (%d): photometric\n",refFrameNumber); fflush(stdout);
        if (mode == BIOBJECTIVE) printf("mode switched (%d): biobjective\n",refFrameNumber); fflush(stdout);
        this->mode = mode;
    }
}

TrackMode PhotoTracker::getMode() {
    return mode;
}
PhotoTracker::~PhotoTracker() {
    if (calibData) delete[] calibData;
}

int PhotoTracker::numReferenceUpdates() {
    return refFrameNumber;
}

int PhotoTracker::getDeviceID() {
    return deviceID;
}

void PhotoTracker::setThreadActive(bool flag) {
    threadActiveFlag = flag;
}

bool PhotoTracker::isActive() {
    return threadActiveFlag;
}

void selectGPU(int deviceID) {
    checkCudaErrors(cudaSetDevice(deviceID));
    printf("selected gpu: %d\n", deviceID); fflush(stdout);

    for (int i = 0; i < numCudaDevices; i++) {
        if (i != deviceID) setupPeerAccess(deviceID,i);
    }
    checkCudaError("selectGPU error");
}

void PhotoTracker::sync() {
    if (!barrier) return;
    if (isActive()) {
/*        static int joo = 0;
        printf("phototracker sync: %d\n",joo); joo++;*/
        barrier->sync();
        cudaDeviceSynchronize();
    }
}

double *PhotoTracker::getGlobalPose() {
    return &globalPose[0];
}

float *PhotoTracker::getRelativePose() {
    return &T[0];
}

void PhotoTracker::resetTracking(TrackMode mode) {
    clearPose();
    referenceExists      = false;
    currentExists        = false;
    referencePrepared    = false;
    refFrameNumber       = 0;
    currentFrame         = 0;
    setMode(mode);
    if (shapeTracker) {
        shapeTracker->resetTracking();
    }
}

void PhotoTracker::clearPose() {
    identity4x4(&globalPose[0]);
    if (Tinc != NULL) identity4x4(&Tinc[0]);
    if (T != NULL) identity4x4(&T[0]);
    if (TrelDev != NULL && TidentityDev != NULL) {
        cudaMemcpyAsync((void*)TrelDev,  (void*)TidentityDev, 16*sizeof(float), cudaMemcpyDeviceToDevice,stream);
        cudaStreamSynchronize(stream);
    }
    //if (shapeTracker) shapeTracker->clearPose();
}


void PhotoTracker::setCurrentImage(cudaArray *currentFrame) {
    if (!threadActiveFlag) return;
    rgbTex2GrayCuda(currentFrame,&grayPyramidCur.getImageRef(0),zCurrentDev,calibDataDev,stream);
    cudaStreamSynchronize(stream);
    /*if (!referenceExists) {
        checkCudaErrors(cudaMemcpy2DArrayToArray(refFrame,0,0,currentFrame,0,0,texWidth*sizeof(float)*4,texHeight,cudaMemcpyDeviceToDevice));
    }*/
    if (shapeTracker) shapeTracker->setCurrentImage(zCurrentDev);
    checkCudaError("cuda current image error");
    currentExists = true;
}

bool PhotoTracker::poseDifferenceExceeded(float *dT, float translationThreshold, float rotationThreshold) {  
   // dumpMatrix("dT",dT,4,4);
//    return true;
    double dx = dT[3];
    double dy = dT[7];
    double dz = dT[11];
    float dist = (float)sqrt(dx*dx+dy*dy+dz*dz+1e-12f);

    if (dist > translationThreshold) return true;
    float angle = 0;
    // check identity matrix as a special case:
    if ((fabs(dT[0]-1.0f) > 1e-5) || (fabs(dT[5]-1.0f) > 1e-5) || (fabs(dT[10]-1.0f) > 1e-5)) {
        float q[4];
        rot2Quaternion(dT, 4, q);
        normalizeQuaternion(q);
        double ca = (double)q[0];
        angle = acos(ca) * 2.0f * 180.0f / 3.141592653f;
    }
    if (angle > rotationThreshold) return true;
    return false;
}

bool PhotoTracker::referenceOutdated() {
    if (!referenceExists) return true;
    if (m_incrementalMode) return true;
  //  return true;
    if (referenceUpdatingFlag && poseDifferenceExceeded(&T[0],poseDistanceThreshold,poseAngleThreshold)) return true;
    return false;
}

void PhotoTracker::updatePhotometricReference(float4 *newRefPointsDev, cudaArray *newRefFrame) {
    checkCudaErrors(cudaMemcpy2DArrayToArray(refFrame,0,0,newRefFrame,0,0,texWidth*sizeof(float)*4,texHeight,cudaMemcpyDeviceToDevice));
    //cudaDeviceSynchronize();
/*    if (!useRawMeasurementsForPoseEstimation) {
        checkCudaErrors(cudaMemcpyAsync(refPointsDev,newRefPointsDev,sizeof(float)*4*texWidth*texHeight,cudaMemcpyDeviceToDevice,stream));
        cudaStreamSynchronize(stream);
        checkCudaError("set reference pointsB error");
    }*/
    referencePrepared = false;
    referenceExists   = true;
}

bool PhotoTracker::setReferencePoints(float4 *newRefPointsDev, float4 *newRefPointNormalsDev, cudaArray *newRefFrame, TrackMode suggestedMethod, float lambda) {
    if (!threadActiveFlag) return false;
    if (newRefFrame == NULL) return false;
    if (shapeTracker->getFilteredReference() && (newRefPointsDev == NULL || newRefPointNormalsDev == NULL)) return false;

    if (suggestedMethod == PHOTOMETRIC) {
        if (referenceOutdated()) {
            setMode(PHOTOMETRIC);
            updatePhotometricReference(newRefPointsDev,newRefFrame); refFrameNumber = currentFrame;
            return true;
        }
    } else if (suggestedMethod == GEOMETRIC){
        if (shapeTracker->referenceOutdated()) {
            setMode(GEOMETRIC);
            shapeTracker->setReferencePoints(newRefPointsDev,newRefPointNormalsDev,newRefFrame); refFrameNumber = currentFrame;
            return true;
        }
    } else if (suggestedMethod == BIOBJECTIVE) {
        if (referenceOutdated() || shapeTracker->referenceOutdated()) {
            setMode(BIOBJECTIVE);
            setLambda(lambda);
            updatePhotometricReference(newRefPointsDev,newRefFrame);  refFrameNumber = currentFrame;
            shapeTracker->setReferencePoints(newRefPointsDev,newRefPointNormalsDev,newRefFrame);
            return true;
        }
    }
    return false;
}

bool PhotoTracker::isReferencePrepared() {
    return referencePrepared;
}

void PhotoTracker::selectPixels(int pixelSelectionAmount) {
    extractGradientMagnitudes(&vbuffer,gradientScratchDev,stream);
    cudaHista(gradientScratchDev, histogramFloatDev, vbuffer.getVertexCount(), 256, d_hist, stream);
    filterIndices4(&vbuffer, gradientScratchDev, histogramFloatDev, pixelSelectionAmount,256,stream);
    addVertexAttributesWithoutNormalsCuda(calibDataDev, &vbuffer, &grayPyramidRef,stream);
    precomputeJacobian4Cuda(&vbuffer,calibDataDev,jacobianTDev[0],jacobianTDev[1],jacobianTDev[2],jacobianTDev[3],optScaleIn,stream);
    //        compressVertexBuffer(&vbufferExt,&vbuffer,rgbVisualization);
}

void PhotoTracker::lock() {
    vbuffer.lock(); vbuffer.lockIndex(); grayPyramidRef.lock(); grayPyramidCur.lock();
}

void PhotoTracker::unlock() {
    vbuffer.unlock(); vbuffer.unlockIndex(); grayPyramidRef.unlock(); grayPyramidCur.unlock();
}

void PhotoTracker::prepareReference() {
    if (!referenceExists || referencePrepared) return;

    // clear relative pose
    cudaMemcpyAsync((void*)TrelDev,  (void*)TidentityDev, 16*sizeof(float), cudaMemcpyDeviceToDevice,stream);
    //memcpy(&TbasePhoto[0],&photoEstimatedPose[0],sizeof(float)*16);
    if (T!=NULL) {
        identity4x4(&T[0]);
    } else {
        printf("phototracker: T == NULL!\n"); fflush(stdout);
    }

    rgbTex2GrayCuda(refFrame,&grayPyramidRef.getImageRef(0),zReferenceDev,calibDataDev,stream);
    grayPyramidRef.updateLayers(stream);

    //if (useRawMeasurementsForPoseEstimation)
    //{
    convertZmapToXYZCuda(zReferenceDev,refPointsDev,calibDataDev,texWidth,texHeight,stream);
    //}
    xyz2DepthCuda(refPointsDev, texWidth,texHeight, debugRefImage1C,calibDataDev,true); // update debugRefImage1C
    xyz2CloudCuda(refPointsDev, calibDataDev, &vbuffer, &grayPyramidRef,stream);

    selectPixels(photometricResidualSize);

   // printf("!!!tracker: preparing new reference! (%d)\n",vbuffer.getElementsCount());
    referencePrepared = true;
}

void PhotoTracker::updateErrorImage() {

    if (this->mode == PHOTOMETRIC) {
        int i = 0;
        vbuffer.lock(); vbuffer.lockIndex();
        xyz2DiffCuda(&vbuffer, texWidth, texHeight, calibDataDev, TrelDev, debugImage1C, texWidth>>i, texHeight>>i, i, &grayPyramidCur,stream);
        vbuffer.unlock(); vbuffer.unlockIndex();
    } if (this->mode == GEOMETRIC || this->mode == BIOBJECTIVE) {
        if (shapeTracker) shapeTracker->updateErrorImage();
    }
/*
    for (int i = 0; i < nLayers; i++) {
        xyz2DiffCuda(&vbuffer, calibDataDev, TrelDev, debugImage1C, texWidth>>i, texHeight>>i, i, &grayPyramidCur);
   }*/

}

void PhotoTracker::listSelected(int layer, float **selected, float **refSelected, float **color, int *nSelect) {
    *nSelect = 0;
    *selected    = NULL;
    *refSelected = NULL;
    *color       = NULL;

    int dstOffset = 0;
    if (this->mode == GEOMETRIC || this->mode == BIOBJECTIVE) {
        shapeTracker->listSelected(layer,selected,refSelected,color,nSelect);
        dstOffset = *nSelect;
        if (this->mode == GEOMETRIC) return;
    }

    if (!referencePrepared) return;
    vbuffer.lock(); vbuffer.lockIndex();
//    listSelectedRefCuda(&vbuffer, selectionPointsDev, selectionColorsDev);
    // note the order of these two commands: the first cmd will produce garbage to selectionColorsDev:
    listSelectedCurCuda(&vbuffer, calibDataDev, TidentityDev, refSelectionPointsDev, selectionColorsDev,stream);
    listSelectedCurCuda(&vbuffer, calibDataDev, TrelDev,         selectionPointsDev, selectionColorsDev,stream);
    if (mode == PHOTOMETRIC) {
        vectorProductCuda(weightsDev,zWeightsDev,vbuffer.getElementsCount(),selectionColorsDev,stream);
//        updateDebugDiffImages();
    }
    vbuffer.unlock(); vbuffer.unlockIndex();
    if (dstOffset > 0) {
        memcpy(selectionPoints,   *selected,   sizeof(float)*dstOffset*2);
        memcpy(refSelectionPoints,*refSelected,sizeof(float)*dstOffset*2);
        memcpy(selectionColors,   *color,      sizeof(float)*dstOffset);
   //     memcpy(refSelectionColors,*color,      sizeof(float)*dstOffset);
    }
    cudaMemcpyAsync(selectionPoints+dstOffset*2,   selectionPointsDev,   2*sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(refSelectionPoints+dstOffset*2,refSelectionPointsDev,2*sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(selectionColors+dstOffset,     selectionColorsDev,     sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
   // cudaMemcpyAsync(refSelectionColors+dstOffset,selectionColorsDev,  sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    *nSelect     = vbuffer.getElementsCount()+dstOffset;
    *selected    = selectionPoints;
    *refSelected = refSelectionPoints;
    *color       = selectionColors;
    return;
}

/*
void PhotoTracker::listRefSelected(int layer, float **selected, float **color, int *nSelect) {
    *nSelect = 0;
    *selected = NULL;
    *color = NULL;

    int dstOffset = 0;
    if (this->mode == GEOMETRIC || this->mode == BIOBJECTIVE) {
        //shapeTracker->listRefSelected(layer,selected,color,nSelect);
        dstOffset = *nSelect;
        if (this->mode == GEOMETRIC) return;
    }

    if (!referencePrepared) return;
    vbuffer.lock(); vbuffer.lockIndex();
    listSelectedCurCuda(&vbuffer, calibDataDev, TidentityDev, refSelectionPointsDev, refSelectionColorsDev,stream);
    if (mode == PHOTOMETRIC) {
        vectorProductCuda(weightsDev,zWeightsDev,vbuffer.getElementsCount(),selectionColorsDev,stream);
        //updateDebugDiffImages();
    }
    vbuffer.unlock(); vbuffer.unlockIndex();
    if (dstOffset > 0) {
        memcpy(refSelectionPoints,*selected,sizeof(float)*dstOffset*2);
        memcpy(refSelectionColors,*color,   sizeof(float)*dstOffset);
    }
    cudaMemcpyAsync(refSelectionPoints+dstOffset*2,refSelectionPointsDev,2*sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(refSelectionColors+dstOffset,  refSelectionColorsDev,  sizeof(float)*vbuffer.getElementsCount(),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    *nSelect  = vbuffer.getElementsCount()+dstOffset;
    *selected = refSelectionPoints;
    *color    = refSelectionColors;
    return;
}*/

void PhotoTracker::prepareCurrent() {
    grayPyramidCur.updateLayers(stream);
}

int PhotoTracker::getLayers() {
    return nLayers;
}

void PhotoTracker::setIterationCounts(int *nIter) {
    for (int i = 0; i < nLayers; i++) nIterations[i] = nIter[i];
}

// for selected points:
    // parallel: warp points + interpolate -> residual
    // 1thread: m-estimator weights
    // parallel: Jpre -> reweight jacobian -> Jw
    // parallel: Jw^T * Jw, Jw^T * residual (system reduction)
    // 1thread: inv6x6, solve x
    // 1thread: expm -> Trel update

// Tphotoref -> Tcur := TirDev * T_inc
// Tcur -> Ticpref := inv(TirDev*T_inc)*T_icp := inv(T_inc)*inv(TirDev)*T_icp
bool PhotoTracker::updateLinearSystem(int layer) {
    interpolateResidual2(&vbuffer,TrelDev,calibDataDev,grayPyramidCur,layer,zCurrentDev,zWeightsDev,residualDev,stream);
    generateWeights64(residualDev,vbuffer.getElementsCount(),weightsDev,zWeightsDev,weightedResidualDev,stream);
    weightJacobian(jacobianTDev[layer], weightsDev, vbuffer.getElementsCount(), weightedJacobianTDev,stream);
    JTJCuda(weightedJacobianTDev,vbuffer.getElementsCount(),JtJDev,stream);
    JTresidualCuda(weightedJacobianTDev,weightedResidualDev,vbuffer.getElementsCount(),residual6Dev,stream);
    return true;
}

void PhotoTracker::applyIncrement() {
    // store previous transform inverse (non-inverted dir: cur -> ref)
    double TdblPrev[16];   double_precision4(T,&TdblPrev[0]);
    double iTprev[16];     invertRT4(&TdblPrev[0],&iTprev[0]);

    float iT[16];
    cudaMemcpyAsync(&iT[0],TrelDev,sizeof(float)*16,cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    double iTdbl[16],Tdbl[16];
    double_precision4(&iT[0],&iTdbl[0]);
    invertRT4(&iTdbl[0],&Tdbl[0]);
    float_precision4(&Tdbl[0],T);
    matrixMult4x4(&iTprev[0],&Tdbl[0],&Tinc[0]);
    // cumulate previous motion into returnable estimate
//    memcpy(&estimatedPose[0],&Tdbl[0],sizeof(double)*16);
    matrixMult4x4(&globalPose[0],&Tinc[0],&globalPose[0]);
}
/*
void PhotoTracker::addShapeTrackerIncrement(float *Tinc) {
    if (Tinc != &this->Tinc[0]) {
        memcpy(&this->Tinc[0],Tinc,sizeof(float)*16);
    }
    float iT[16];
    matrixMult4x4(&T[0],&Tinc[0],&T[0]);
    invertRT4(&T[0],&iT[0]);
    // gpu mem has inverted base transform base->cur
    cudaMemcpyAsync((void*)TrelDev, (void*)&iT[0],16*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
}*/

void PhotoTracker::resetToTransform(double *inputT) {
   // memcpy(&estimatedPose[0],inputT,sizeof(float)*16);
    float_precision4(inputT,T);
    identity4x4(&Tinc[0]);

    float iT[16];
    invertRT4(&T[0],&iT[0]);
    cudaMemcpyAsync((void*)TrelDev, (void*)&iT[0],16*sizeof(float),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
/*
    if (shapeTracker) {
        shapeTracker->clearPose();
     //   shapeTracker->resetToTransform(inputT);
//        memcpy(&estimatedPose[0],&Tinc[0],sizeof(float)*16);
    }
    */
}


double *PhotoTracker::getIncrement() {
    return &Tinc[0];
}



void PhotoTracker::signalCalibrationUpdate(float *calibDataExt) {
    memcpy(this->calibData,calibDataExt,sizeof(float)*CALIB_SIZE);
    calibrationUpdateSignaled = true;
}

bool PhotoTracker::readyToTrack() {
    // also update calibration at the beginning of frame (if signaled)
    if (calibrationUpdateSignaled) {
        cudaMemcpyAsync(calibDataDev,calibData,CALIB_SIZE*sizeof(float),cudaMemcpyHostToDevice,stream);
        cudaStreamSynchronize(stream);
        if (shapeTracker) shapeTracker->updateCalibration(calibData);
        calibrationUpdateSignaled = false;
        printf("calibration updated!\n");
    }
    bool photometricReady = (referenceExists && currentExists);

    if (getMode() == PHOTOMETRIC) return photometricReady;
    else if (getMode() == GEOMETRIC) return shapeTracker->readyToTrack();
    else if (getMode() == BIOBJECTIVE) return shapeTracker->readyToTrack() && photometricReady;
    else { printf("phototracker::readyToTrack() is given invalid mode!\n"); fflush(stdout); return false; }
}

ShapeTracker *PhotoTracker::getShapeTracker() {
    return shapeTracker;
}

void PhotoTracker::gpuPreProcess() {
    int mode = getMode();
    if (mode == GEOMETRIC || mode == BIOBJECTIVE) {
        ShapeTracker *shapeTracker = getShapeTracker();
        if (!shapeTracker->isReferencePrepared()) {
            shapeTracker->prepareReference();
        }
        shapeTracker->prepareCurrent();
    }

    if (mode == PHOTOMETRIC || mode == BIOBJECTIVE) {
        if (!isReferencePrepared()) {
            prepareReference();
        }
        prepareCurrent();
/*
        // TODO!
        // save images and compare with reference frames
        // there is a mismatch with photometric results (gputracker says it better!)

        static int curFrame = 1;
        // save current image to disk for debug purposes
        Mat rgbImageF(240,320,CV_32FC1);
        Mat rgbImage(240,320,CV_8UC1);
        char buf[512];
        sprintf(buf,"scratch/cur-%04d.ppm",curFrame);
        cudaMemcpy(rgbImageF.ptr(),grayPyramidCur.getImageRef(0).devPtr,320*240*sizeof(float),cudaMemcpyDeviceToHost);
        unsigned char *dst = rgbImage.ptr();
        float *src = (float*)rgbImageF.ptr();
        for (int i = 0; i < 320*240; i++) dst[i] = src[i]*255.0f;
        imwrite(buf,rgbImage);

        sprintf(buf,"scratch/ref-%04d.ppm",curFrame);
        cudaMemcpy(rgbImageF.ptr(),grayPyramidRef.getImageRef(0).devPtr,320*240*sizeof(float),cudaMemcpyDeviceToHost);
        dst = rgbImage.ptr();
        src = (float*)rgbImageF.ptr();
        for (int i = 0; i < 320*240; i++) dst[i] = src[i]*255.0f;
        imwrite(buf,rgbImage);
        curFrame++;*/
    }
}

void PhotoTracker::getIterationCounts(int *niter) {
    if (mode == PHOTOMETRIC) memcpy(niter,&nIterations[0],sizeof(int)*4);
    if (mode == GEOMETRIC) shapeTracker->getIterationCounts(niter);
    if (mode == BIOBJECTIVE) {
        memcpy(niter,&nIterationsBiObjective[0],sizeof(int)*4);
    }
}

void PhotoTracker::estimateIncrement(int layer) {
    if (mode == GEOMETRIC) shapeTracker->copyLinearSystemTo(JtJDev,residual6Dev);
    if (mode == BIOBJECTIVE) shapeTracker->linearFuse(JtJDev,residual6Dev,photometricResidualSize, layer, lambda);
    if (estimateOnCPU) {
        cv::Mat Trel(4,4,CV_32FC1);
        cv::Mat x(6,1,CV_32FC1);
        cv::Mat b(6,1,CV_32FC1);
        cv::Mat A(6,6,CV_32FC1);
        cudaMemcpyAsync(A.ptr(),JtJDev,sizeof(float)*36,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(b.ptr(),residual6Dev,sizeof(float)*6,cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(Trel.ptr(),TrelDev,sizeof(float)*16,cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
        cv::solve(A,b,x,cv::DECOMP_CHOLESKY);
        expm((float*)x.ptr(),(float*)Trel.ptr(), optScaleOut);
        cudaMemcpyAsync(TrelDev,Trel.ptr(),sizeof(float)*16,cudaMemcpyHostToDevice,stream);
    } else {
        solveMotionCuda(JtJDev,residual6Dev,TrelDev,optScaleOut,stream);
    }
}

void PhotoTracker::gpuEstimateIncrement() {
    int mode = getMode();
    ShapeTracker *shapeTracker = getShapeTracker();

    int nIter[4]; getIterationCounts(&nIter[0]);

    vbuffer.setStream(stream);
    int firstLayer = nLayers-1;
    int finalLayer = 0;
    for (int i = firstLayer; i >= finalLayer; i--) {
        int n = nIter[nLayers-1-i];
        for (int j = 0; j < n; j++) {
            bool ok = true;
            if (mode == PHOTOMETRIC) { ok &= updateLinearSystem(i); }
            if (mode == GEOMETRIC)   { ok &= shapeTracker->updateLinearSystem(i,j);  }
            if (mode == BIOBJECTIVE) { ok &= updateLinearSystem(i); ok &= shapeTracker->updateLinearSystem(i,j);  }
            if (ok) estimateIncrement(i);
            if (j == 0 && i == firstLayer) {
                cudaStreamSynchronize(stream);
            }
        }
    }
    currentFrame++;
}

void *gpuProcessThread(void *inputParams)
{
    INPUT_PARAMS *params = (INPUT_PARAMS*)inputParams;
    selectGPU(params->deviceID);
    photoTracker = new PhotoTracker(params->deviceID,barrier,params->width,params->height,params->calib);
    ShapeTracker *shapeTracker = photoTracker->getShapeTracker();

    printf("CUDA Device #%d\n",photoTracker->getDeviceID()); fflush(stdout);

    photoTracker->allocMemory();

    photoTracker->setThreadActive(true);
    //photoTracker->isActive()
    while (!g_killThreads) {
        // wait until the current image has been updated
        photoTracker->sync();
        // now images are updated, optimize motion from reference -> current image
        if (photoTracker->readyToTrack())  {
            photoTracker->lock();
            photoTracker->gpuPreProcess();
            photoTracker->gpuEstimateIncrement();
            photoTracker->applyIncrement();
            photoTracker->unlock();            
        }
        // wait until tsdf fusion and raycasting has been finished
        photoTracker->sync();
        // now the reference may be updated
    }
    photoTracker->setThreadActive(false);
    printf("thread %d finished\n", photoTracker->getDeviceID()); fflush(stdin); fflush(stdout);
    photoTracker->freeMemory();
    delete photoTracker; photoTracker = NULL;
    delete params;

    return NULL;
  //  cudaDeviceReset();
 //   CUT_THREADEND;
}

float4 *PhotoTracker::getDebugImage() {
    return debugImage;
}

float *PhotoTracker::getDebugImage1C(int layer) {
   // return debugImage1C;
    if (mode  == PHOTOMETRIC) {
        return debugImage1C;//(float*)grayPyramidRef.getImageRef(layer).devPtr;
    } else if (mode == GEOMETRIC) {
        if (shapeTracker) return shapeTracker->getDebugImage1C(layer);
    } else if (mode == BIOBJECTIVE) {
        if (shapeTracker) return shapeTracker->getDebugImage1C(layer);
    }
//    else return NULL;
}

float *PhotoTracker::getRefImage1C(int layer) {
    if (mode  == PHOTOMETRIC) {
        return debugRefImage1C;//(float*)grayPyramidRef.getImageRef(layer).devPtr;
    } else if (mode == GEOMETRIC) {
        if (shapeTracker) return shapeTracker->getRefImage1C(layer);
    } else if (mode == BIOBJECTIVE) {
        if (shapeTracker) return shapeTracker->getRefImage1C(layer);
    }
//    else return NULL;
}

void PhotoTracker::allocMemory() {
    // allocate extra rgbdFrame for producing delay - 1 to syntetic inputs:
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);

    //    checkCudaErrors(cudaMallocArray(&this->rgbdFrame, &channelDesc, texWidth, texHeight));
    checkCudaErrors(cudaMallocArray(&this->refFrame,  &channelDesc, texWidth, texHeight));
    checkCudaErrors(cudaMalloc(&this->debugImage,      texWidth*texHeight*sizeof(float)*4));
    checkCudaErrors(cudaMalloc(&this->debugImage1C,    texWidth*texHeight*sizeof(float))); checkCudaErrors(cudaMemset(debugImage1C, 0,texWidth*texHeight*sizeof(float)));
    checkCudaErrors(cudaMalloc(&this->debugRefImage1C, texWidth*texHeight*sizeof(float))); checkCudaErrors(cudaMemset(debugRefImage1C, 0,texWidth*texHeight*sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&refPointsDev, texWidth*texHeight*sizeof(float)*4));


    float *blancoDev = NULL;
    checkCudaErrors(cudaMalloc(&blancoDev, texWidth*texHeight*4*sizeof(float)));
    checkCudaErrors(cudaMemset(blancoDev, 0,texWidth*texHeight*4*sizeof(float)));
    checkCudaErrors(cudaMemcpy2DToArray(refFrame,0,0,blancoDev,texWidth*4*sizeof(float),texWidth,texHeight,cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(blancoDev));

    if (texWidth == 640) {
        nLayers = 4;
    } else if (texWidth == 320) {
        nLayers = 3;
    } else {
        // wtf?
        nLayers = 3;
    }

    //reference = new KeyFrame(texWidth,texHeight);
    grayPyramidCur.createHdrPyramid(texWidth,texHeight,1,nLayers,false,ONLY_GPU_TEXTURE,false); grayPyramidCur.setName("grayPyramid1C-current");
    grayPyramidRef.createHdrPyramid(texWidth,texHeight,1,nLayers,false,ONLY_GPU_TEXTURE,false); grayPyramidRef.setName("grayPyramid1C-reference");

    vbuffer.init(texWidth*texHeight,false,VERTEXBUFFER_STRIDE);//COMPRESSED_STRIDE);

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

    cudaMalloc((void **)&histogramFloatDev, 256 * sizeof(float));
    cudaMalloc((void**) &d_hist, 64 * 256 * sizeof(float));
    cudaMalloc((void **)&gradientScratchDev, texWidth*texHeight * sizeof(float));
    initHistogram64();
    for (int i = 0; i < 4; i++) {
        cudaMalloc((void **)&jacobianTDev[i], 6*photometricResidualSize*sizeof(float)); cudaMemset(jacobianTDev[i],0,6*photometricResidualSize*sizeof(float));
    }

    initCudaDotProduct();

    float *zWeights = new float[photometricResidualSize];
    for (int i = 0; i < photometricResidualSize; i++) zWeights[i] = 1.0f;

    cudaMalloc((void **)&residualDev,  photometricResidualSize*sizeof(float));           cudaMemset(residualDev,0,photometricResidualSize*sizeof(float));
    cudaMalloc((void **)&weightedResidualDev,  photometricResidualSize*sizeof(float));   cudaMemset(weightedResidualDev,0,photometricResidualSize*sizeof(float));
    cudaMalloc((void **)&residual6Dev, 6*sizeof(float));                                 cudaMemset(residual6Dev,0,6*sizeof(float));
    cudaMalloc((void **)&weightsDev,  photometricResidualSize*sizeof(float));            cudaMemset(weightsDev, 0,photometricResidualSize*sizeof(float));
    cudaMalloc((void **)&zWeightsDev,  photometricResidualSize*sizeof(float));           cudaMemcpy(zWeightsDev,zWeights,photometricResidualSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&weightedJacobianTDev, 6*photometricResidualSize*sizeof(float)); cudaMemset(weightedJacobianTDev,0,6*photometricResidualSize*sizeof(float));
    cudaMalloc((void **)&JtJDev, 6*6*sizeof(float));                                     cudaMemset(JtJDev,0,6*6*sizeof(float));
    cudaMalloc((void **)&selectionPointsDev, photometricResidualSize*sizeof(float)*2);   cudaMemset(selectionPointsDev,0,photometricResidualSize*sizeof(float)*2);
    cudaMalloc((void **)&selectionColorsDev, photometricResidualSize*sizeof(float));     cudaMemset(selectionColorsDev,0,photometricResidualSize*sizeof(float));
    cudaMalloc((void **)&refSelectionPointsDev, photometricResidualSize*sizeof(float)*2);cudaMemset(refSelectionPointsDev,0,photometricResidualSize*sizeof(float)*2);
   // cudaMalloc((void **)&refSelectionColorsDev, photometricResidualSize*sizeof(float));  cudaMemset(refSelectionColorsDev,0,photometricResidualSize*sizeof(float));
    cudaMalloc((void**)&zCurrentDev,texWidth*texHeight*sizeof(float));                   cudaMemset(zCurrentDev,0,texWidth*texHeight*sizeof(float));
    cudaMalloc((void**)&zReferenceDev,texWidth*texHeight*sizeof(float));                 cudaMemset(zReferenceDev,0,texWidth*texHeight*sizeof(float));

    delete[] zWeights;

    selectionPoints    = new float[texWidth*texHeight*4];
    refSelectionPoints = new float[texWidth*texHeight*4];

    selectionColors = new float[texWidth*texHeight*2];

    nSelected = 0;

    if (shapeTracker) {
        shapeTracker->allocMemory();
//        shapeTracker->setCurrentImage(rgbdFrame);
        shapeTracker->resetTracking();
    }
}


void PhotoTracker::freeMemory() {

    // disable peer access:
    for (int i = 0; i < numCudaDevices; i++) {
        if (i != deviceID) {
            disablePeerAccess(deviceID,i);
        }
    }
    if (T) delete[] T;
    if (Tinc) delete[] Tinc;
    if (shapeTracker) { shapeTracker->release(); delete shapeTracker; }
    if (selectionPoints) delete[] selectionPoints;
    if (selectionColors) delete[] selectionColors;
    if (refSelectionPoints) delete[] refSelectionPoints;
   // if (refSelectionColors) delete[] refSelectionColors;
    //if (rgbdFrame) cudaFreeArray(rgbdFrame);
    if (refFrame) cudaFreeArray(refFrame);
    if (debugImage) cudaFree(debugImage);
    if (debugImage1C) cudaFree(debugImage1C);
    if (debugRefImage1C) cudaFree(debugRefImage1C);
    if (refPointsDev) cudaFree(refPointsDev);
    if (calibDataDev) cudaFree(calibDataDev);
    if (histogramFloatDev) cudaFree(histogramFloatDev);
    if (d_hist) cudaFree(d_hist);
    if (gradientScratchDev) cudaFree(gradientScratchDev);
    if (residualDev) cudaFree(residualDev);
    if (weightedResidualDev) cudaFree(weightedResidualDev);
    if (residual6Dev) cudaFree(residual6Dev);
    if (weightsDev) cudaFree(weightsDev);
    if (zWeightsDev) cudaFree(zWeightsDev);
    if (weightedJacobianTDev) cudaFree(weightedJacobianTDev);
    if (JtJDev) cudaFree(JtJDev);
    if (selectionPointsDev) cudaFree(selectionPointsDev);
    if (selectionColorsDev) cudaFree(selectionColorsDev);
    if (refSelectionPointsDev) cudaFree(refSelectionPointsDev);
   // if (refSelectionColorsDev) cudaFree(refSelectionColorsDev);
    if (zCurrentDev) cudaFree(zCurrentDev);
    if (zReferenceDev) cudaFree(zReferenceDev);
    for (int i = 0; i < 4; i++) if (jacobianTDev[i]) cudaFree(jacobianTDev[i]);
    if (TrelDev) cudaFree(TrelDev);
    if (TidentityDev) cudaFree(TidentityDev);
    grayPyramidRef.releaseData();
    grayPyramidCur.releaseData();
    vbuffer.release();
    releaseCudaDotProduct();
    closeHistogram64();
    //if (reference) reference->release();
}


PhotoTracker *initializePhotoTracker(int deviceID, int texWidth, int texHeight, Calibration *calib) {
    printf("running phototracker!\n");
    INPUT_PARAMS *params = new INPUT_PARAMS;
    params->width = texWidth;
    params->height = texHeight;
    params->calib = calib;
    params->deviceID = deviceID;

    pthread_create(&threadID, NULL, gpuProcessThread, (void*)params);

    while (photoTracker == NULL) { sleepMs(50); }
    while (!photoTracker->isActive()) { sleepMs(50); }
    return photoTracker;
}
/*
void releasePhotoTracker() {
    if (photoTracker) { photoTracker->release();  delete photoTracker; photoTracker = NULL; }
}*/

bool trackingThreadAlive() {
    while (photoTracker != NULL) { sleepMs(10); }
}
