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

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <multithreading.h>
#include <reconstruct/VoxelGrid.h>
#include <kernels/VoxelGrid.cuh>
#include "devices.cuh"
#include "GLWindow.h"
#include <Renderer.h>
#include <GLSLProgram.h>
#include <Shaders.h>
#include <fbo.h>
#include <SleepMs.h>
#include <performanceCounter.h>
#include <TrueTypeText.h>   // OpenGL fonts
#include <tracker/basic_math.h>
#include "screenshot.h"
#include <calib/calib.h>
#include <calib/TrackingParams.h>
#include <capture/fileSource.h>
#include <capture/VideoPreProcessorCPU.h>
#include <time.h>
#include <tracker/phototracker.h>
#include <calib/ResultTxt.h>

const int NUM_RENDER_TARGETS = 3; // 3 default views
FBO fbos(NUM_RENDER_TARGETS);
int RENDER_TARGET_WIDTH  = 320;
int RENDER_TARGET_HEIGHT = 240;
enum { RENDER_TARGET_FINAL = 0, RENDER_TARGET_MAP = 1, RENDER_TARGET_RAW = 2};
bool saveNextFrame = false;
bool running = true;
bool loading = false;
bool loadingFinished = false;
bool failureInLoading = false;
bool cudaContextExists = false;
Eigen::Vector3f mousePos(0,0,1);
Eigen::Vector3f mousePosAnchor(0,0,1);
Eigen::Vector3f mouseDx(0,0,1);
int WINDOW_RESOX = 1920*0.5;
int WINDOW_RESOY = 1080*0.75;
SDL_Window *sdlWindow = NULL;
SDL_GLContext glcontext = NULL;
SDL_Thread *loaderThreadHandle = NULL;
Renderer *renderer   = 0;
ScreenShot    *screenShot = 0;
TrueTypeText font;
bool showLoadingScreen = true;
// OpenGL sub-windows
std::vector<GLWindow *> glWindows;
Calibration calibKinect("data/calib.xml",true);
int freeMemoryKb=0, totalMemoryKb=0;
uint g_gridResolution = 256;//400;//512;//512;//256;//256; must be divisible by 8!
GLuint splashTextureID = 0;
int currentFrame = 0;
int layer = 0;
GLuint floorTex = 0;
GLuint boxTex = 0;

int g_TotalErrors = 0;

// view params
int ox, oy;
int buttonState = 0;
bool keyDown[256];

Eigen::Vector3f cameraPos(0, 0, 1000);//1000, 2000);
Eigen::Vector3f cameraRot(0, 0, 0);
Eigen::Vector3f cameraPosLag(cameraPos);
Eigen::Vector3f cameraRotLag(cameraRot);
Eigen::Vector3f cursorPos(0, 1, 0);
Eigen::Vector3f cursorPosLag(cursorPos);
float fov = 40.0f;
Eigen::Vector3f lightPos(5.0, 5.0, -5.0);

const float inertia = 0.1f;
const float translateSpeed = 100.000f;
const float cursorSpeed = 0.01f;
const float rotateSpeed = 0.2f;
const float walkSpeed = 10.00f;

enum { M_VIEW = 0, M_MOVE_CURSOR, M_MOVE_LIGHT };
int mode = 0;

bool relativeTSDF = true;//false;//true;//false;//true;//false;//true;
float planarityIndex = 0;
Eigen::Vector3f colorAttenuation(0.5f, 0.75f, 1.0f);
float blurRadius = 0.5f;
FileSource *fileSource = NULL;
bool restartTrackingFlag = true;
TrackingParams trackParams;
ResultTxt  resultTrajectory;

// fps
static int fpsCount = 0;
static int fpsLimit = 30;

float modelView[16];     // pose at frame n
float prevModelView[16]; // pose at frame n-1
float estimatedPose[16];
float tsdfPose[16];

//Keyframe keyframe;
//GLuint framebuffer = 0;     // to bind the proper targets
//GLuint depth_buffer = 0;    // for proper depth test while rendering the scene
GLuint inputTexture = 0;      // where we render the image
GLuint sensorRGBTex = 0;      // where we render the image
GLuint sensorDepthTex = 0;      // where we render the image
GLuint debugTexture = 0;
GLuint debugTexture1C[4] = {0,0,0,0};
GLuint refTexture1C[4] = {0,0,0,0};
GLuint fbo_source = 0;
GLuint outputTexture;  // where we will copy the CUDA result
struct cudaGraphicsResource *cudaInputTexture = NULL;
struct cudaGraphicsResource *cudaOutputTexture = NULL;
struct cudaGraphicsResource *cudaDebugTexture = NULL;
struct cudaGraphicsResource *cudaDebugTexture1C[4] = {NULL,NULL,NULL,NULL};
struct cudaGraphicsResource *cudaRefTexture1C[4] = {NULL,NULL,NULL,NULL};
float *blancoDev = NULL;
cudaArray *rgbdFrame = NULL;
cudaArray *rgbdFramePrev = NULL;

bool showErrorImage = true;
bool showSelectedPoints = true;//false;
bool useKinectInput = false;
VideoPreProcessorCPU *preprocessor = NULL;
PhotoTracker *photoTracker = NULL;

// CheckRender object for verification
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.40f


typedef struct {
    float *selected;
    float *refSelected;
    float *color;
    int nSelected;
} SELECTED_POINTS;
SELECTED_POINTS selectedPoints;

VoxelGrid *voxelGridArray[2] = {NULL,NULL};
VoxelGrid *voxelGrid = voxelGridArray[0];
int activeGrid = 0;

// Define the files that are to be saved and the reference images for validation
const char *projectName = "cube test";

const char *sRefBin[]  =
{
    "ref_smokePart_pos.bin",
    "ref_smokePart_vel.bin",
    NULL
};
float TtmpB[16];
float *convertT(float *Tin) {
    invertRT4(&Tin[0],&TtmpB[0]);
    return &TtmpB[0];
}

void createCudaTexture(GLuint *tex_cudaResult, bool gray, unsigned int size_x, unsigned int size_y, cudaGraphicsResource **cudaTex)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (!gray) {
        printf("Creating a Texture GL_RGBA32F_ARB\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);
    } else {
        printf("Creating a Texture GL_LUMINANCE32F_ARB\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, size_x, size_y, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    }

    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(cudaTex, *tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void createTextureSrc(GLuint *tex_screen, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_screen);
    glBindTexture(GL_TEXTURE_2D, *tex_screen);
    // buffer data
//#ifndef USE_TEXTURE_RGBA8UI
    printf("Creating a Texture render target GL_RGBA32F_ARB\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
/*#else
    printf("Creating a Texture render target GL_RGBA8UI_EXT\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
#endif*/
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);//GL_NEAREST);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaInputTexture, *tex_screen, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    // allocate extra rgbdFrame for producing delay - 1 to syntetic inputs:
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&rgbdFrame, &channelDesc, size_x, size_y));
    checkCudaErrors(cudaMallocArray(&rgbdFramePrev, &channelDesc, size_x, size_y));
    if (blancoDev == NULL) {
        checkCudaErrors(cudaMalloc(&blancoDev, size_x*size_y*4*sizeof(float)));
        checkCudaErrors(cudaMemset(blancoDev, 0,size_x*size_y*4*sizeof(float)));
    }
}

void createSensorTextures(GLuint *sensorRGBTex, GLuint *sensorDepthTex, int size_x, int size_y) {
    // create a texture
    glGenTextures(1, sensorDepthTex);
    glBindTexture(GL_TEXTURE_2D, *sensorDepthTex);
    // buffer data
    printf("Creating a Texture GL_LUMINANCE32F_ARB\n");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, size_x, size_y, 0, GL_LUMINANCE, GL_FLOAT, NULL);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    SDK_CHECK_ERROR_GL();

    // create a texture
    glGenTextures(1, sensorRGBTex);
    glBindTexture(GL_TEXTURE_2D, *sensorRGBTex);
    // buffer data
    printf("Creating a Texture GL_RGB\n");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, size_x, size_y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SDK_CHECK_ERROR_GL();
}

void setCalibration(bool kinectInput) {
    currentFrame = 0;
    resultTrajectory.reset(false);
    identity4x4(&modelView[0]);
    identity4x4(&prevModelView[0]);
    identity4x4(&TtmpB[0]);
    identity4x4(&estimatedPose[0]);
    identity4x4(&tsdfPose[0]);
    Eigen::Vector3f zeroVec(0,0,0);
    cursorPos    = zeroVec;
    cursorPosLag = zeroVec;
    cameraPos    = zeroVec;//nv::vec3f(0,0,500);
    cameraPosLag = cameraPos;
    cameraRot    = zeroVec;
    cameraRotLag = zeroVec;

    if (!photoTracker || !renderer) return;
    if (kinectInput)
    {
        calibKinect.setupCalibDataBuffer(RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
        photoTracker->signalCalibrationUpdate(calibKinect.getCalibData());
    } else {
        // initialize settings to kinect based calibration
        calibKinect.setupCalibDataBuffer(RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
        // ask renderer which are the rendering calibration parameters:
        float *renderK  = renderer->getK();
        float *zK       = &calibKinect.getCalibData()[KL_OFFSET]; float *izK = &calibKinect.getCalibData()[iKL_OFFSET];
        zK[0] = renderK[0]; zK[1] =  0;          zK[2] = renderK[2];
        zK[3] = 0;          zK[4] =  renderK[4]; zK[5] = renderK[5];
        zK[6] = 0; zK[7] = 0; zK[8] = 1;
        inverse3x3(zK,izK);
        memcpy(&calibKinect.getCalibData()[KR_OFFSET],zK,sizeof(float)*9);
        memcpy(&calibKinect.getCalibData()[iKR_OFFSET],izK,sizeof(float)*9);
        memset(&calibKinect.getCalibData()[KcR_OFFSET],0,sizeof(float)*5);
        identity4x4(&calibKinect.getCalibData()[TLR_OFFSET]);
        identity4x4(&calibKinect.getCalibData()[TRL_OFFSET]);
        photoTracker->signalCalibrationUpdate(calibKinect.getCalibData());
    }
}

void resetVariables() {
    if (voxelGrid) voxelGrid->clear();
    setCalibration(useKinectInput);
    if (photoTracker) {
        photoTracker->resetTracking();
        photoTracker->setReferenceUpdating(true);
    }
}

void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    *tex = 0;
}


void release()
{
    // signal to threads: exit
    g_killThreads = true;
    // let barriers through:
    barrier->disable();
    // cleanup
    cleanupCudaDevices();
    // wait threads to die:
    while (trackingThreadAlive()) { sleepMs(50); }

    SAFE_RELEASE(renderer);
    SAFE_RELEASE(screenShot);
    SAFE_RELEASE(fileSource);


    fbos.release();

    for (size_t i = 0; i < glWindows.size(); i++) {
        GLWindow *window = glWindows[i];
        SAFE_RELEASE(window);
    }
    glWindows.clear();
    font.clean();    

    // unregister this buffer object with CUDA
    if (cudaInputTexture != NULL)  checkCudaErrors(cudaGraphicsUnregisterResource(cudaInputTexture));
    if (cudaOutputTexture != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaOutputTexture));
    if (cudaDebugTexture != NULL)  checkCudaErrors(cudaGraphicsUnregisterResource(cudaDebugTexture));
    for (int i = 0; i < 4; i++) {
        if (cudaDebugTexture1C[i] != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaDebugTexture1C[i]));        
        if (debugTexture1C[i])  deleteTexture(&debugTexture1C[i]);
        if (cudaRefTexture1C[i] != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaRefTexture1C[i]));
        if (refTexture1C[i])  deleteTexture(&refTexture1C[i]);
    }
    if (boxTex) deleteTexture(&boxTex);
    if (floorTex) deleteTexture(&floorTex);
    if (outputTexture)  deleteTexture(&outputTexture);
    if (debugTexture)   deleteTexture(&debugTexture);
    if (inputTexture)   deleteTexture(&inputTexture);
    if (sensorRGBTex)   deleteTexture(&sensorRGBTex);
    if (sensorDepthTex) deleteTexture(&sensorDepthTex);
   // if (depth_buffer)   deleteDepthBuffer(&depth_buffer);
   // if (framebuffer)    deleteFramebuffer(&framebuffer);
    if (rgbdFramePrev)  cudaFreeArray(rgbdFramePrev);
    if (rgbdFrame)  cudaFreeArray(rgbdFrame);
    checkCudaErrors(cudaFree(blancoDev));
    if (preprocessor) { preprocessor->release(); delete preprocessor; }
    for (int i = 0; i < 2; i++) if (voxelGridArray[i] != NULL) delete voxelGridArray[i];
    resultTrajectory.save(1,0);
    cudaDeviceReset();
    SDL_GL_DeleteContext(glcontext);
    SDL_DestroyWindow(sdlWindow);
    SDL_Quit();
}

void getEstimatedPose(float *pose) {
    if (!useKinectInput) {
        //memcpy(pose,tracker->getEstimatedPose(),sizeof(float)*16);
        // ideal estimation: use the exact camera parameters
//        dumpMatrix("estimated pose",pose,4,4);
        if (!restartTrackingFlag) {
            memcpy(pose,convertT(&prevModelView[0]),sizeof(float)*16);
        } else {
            memcpy(pose,convertT(&modelView[0]),sizeof(float)*16);
        }
 //       dumpMatrix("ideal pose",pose,4,4);
    } else {
        // pose is available only for the previous frames
        float_precision4(photoTracker->getGlobalPose(),pose);
    }
}

void getTSDFPose(float *pose) {
    if (!useKinectInput) {
        //memcpy(pose,tracker->getEstimatedPose(),sizeof(float)*16);
        // ideal estimation: use the exact camera parameters
//        dumpMatrix("estimated pose",pose,4,4);
        if (!restartTrackingFlag) {
            memcpy(pose,convertT(&prevModelView[0]),sizeof(float)*16);
        } else {
            memcpy(pose,convertT(&modelView[0]),sizeof(float)*16);
        }
 //       dumpMatrix("ideal pose",pose,4,4);
    } else {
        // pose is available only for the previous frames
        if (relativeTSDF) {
            memcpy(pose,photoTracker->getRelativePose(),sizeof(float)*16);
        } else {
            float_precision4(photoTracker->getGlobalPose(),pose);
        }
    }
}




void renderDebugScene(float *pose, int offX, int offY, int viewportW, int viewportH, char *text) {

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glPushMatrix();
    float poseT[16];
    float poseTi[16];
    invertRT4(pose,poseTi);
    transpose4x4(poseTi,poseT);
    glLoadIdentity();
    glLoadMatrixf(poseT);
    renderer->render(offX,offY,viewportW,viewportH,text);

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_COLOR_MATERIAL);
    glPointSize(5.0f);
    glColor4f(0,1,0,1);
 /*   glBegin(GL_POINTS);
    glVertex3f(estimatedPose[3],estimatedPose[7],estimatedPose[11]);
    glEnd();
*/
    glPopMatrix();
}

void renderFunc1(float x0, float y0, float w, float h)
{
    fbos.blit(RENDER_TARGET_FINAL,x0,y0,w,h);
}


void updateSensorTextures() {
    static int joo = 0;
//    if (joo < 2) preprocessor->preprocess(true,0.0f,10000.0f);
    preprocessor->preprocess(true,false,false);
    Mat &depthImage = preprocessor->getDepthImageL();
    if (depthImage.cols != RENDER_TARGET_WIDTH || depthImage.rows != RENDER_TARGET_HEIGHT) { printf("updateSensorTextures: dimension mismatch!\n"); return;}
    glBindTexture(GL_TEXTURE_2D,sensorDepthTex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT,GL_LUMINANCE,GL_FLOAT,depthImage.ptr());
    glBindTexture(GL_TEXTURE_2D,sensorRGBTex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT,GL_RGB,GL_UNSIGNED_BYTE,preprocessor->getDistortedRGBImage().ptr());
    glBindTexture(GL_TEXTURE_2D,0);
}

// transfrom vector by matrix
void xform(Eigen::Vector3f &v, Eigen::Vector3f &r, float *m)
{
    r(0) = v(0)*m[0] + v(1)*m[4] + v(2)*m[8] + m[12];
    r(1) = v(0)*m[1] + v(1)*m[5] + v(2)*m[9] + m[13];
    r(2) = v(0)*m[2] + v(1)*m[6] + v(2)*m[10] + m[14];
}

// transform vector by transpose of matrix (assuming orthonormal)
void ixform(Eigen::Vector3f &v, Eigen::Vector3f &r, float *m)
{
    r(0) = v(0)*m[0] + v(1)*m[4] + v(2)*m[8];
    r(1) = v(0)*m[4] + v(1)*m[5] + v(2)*m[9];
    r(2) = v(0)*m[8] + v(1)*m[6] + v(2)*m[10];
}

void handleMouseMotion(SDL_MouseMotionEvent &motion)
{
    float x = float(motion.x);
    float y = float(motion.y);//texHeight-float(motion.y);
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    switch (mode)
    {
        case M_VIEW:
            {
                if (buttonState == 1)
                {
                    // left = rotate
                    cameraRot[0] += dy * rotateSpeed;
                    cameraRot[1] += dx * rotateSpeed;
                }

                if (buttonState == 2)
                {
                    // middle = translate
                    Eigen::Vector3f v = Eigen::Vector3f(dx*translateSpeed, -dy*translateSpeed, 0.0f);
                    Eigen::Vector3f r;
                    ixform(v, r, modelView);
                    cameraPos += r;
                }

                if (buttonState == 3)
                {
                    // left+middle = zoom
                    Eigen::Vector3f v = Eigen::Vector3f(0.0, 0.0, dy*translateSpeed);
                    Eigen::Vector3f r;
                    ixform(v, r, modelView);
                    cameraPos += r;
                }
            }
            break;

        case M_MOVE_CURSOR:
            {
                if (buttonState==1)
                {
                    Eigen::Vector3f v = Eigen::Vector3f(dx*cursorSpeed, -dy*cursorSpeed, 0.0f);
                    Eigen::Vector3f r;
                    ixform(v, r, modelView);
                    cursorPos += r;
                }
                else if (buttonState==2)
                {
                    Eigen::Vector3f v = Eigen::Vector3f(0.0f, 0.0f, dy*cursorSpeed);
                    Eigen::Vector3f r;
                    ixform(v, r, modelView);
                    cursorPos += r;
                }
            }
            break;

        case M_MOVE_LIGHT:
            if (buttonState==1)
            {
                Eigen::Vector3f v = Eigen::Vector3f(dx*cursorSpeed, -dy*cursorSpeed, 0.0f);
                Eigen::Vector3f r;
                ixform(v, r, modelView);
                lightPos += r;
            }
            else if (buttonState==2)
            {
                Eigen::Vector3f v = Eigen::Vector3f(0.0f, 0.0f, dy*cursorSpeed);
                Eigen::Vector3f r;
                ixform(v, r, modelView);
                lightPos += r;
            }

            break;

    }

    ox = x;
    oy = y;
}


// setup OpenGL state
SDL_GLContext configureGL( int width, int height )
{
    SDL_GLContext glcontext = SDL_GL_CreateContext(sdlWindow);
    SDL_GL_MakeCurrent(sdlWindow, glcontext);
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5"))
    {
        fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_VERSION_2_0\n\tGL_VERSION_1_5\n");
//        exit(EXIT_SUCCESS);
    } else {
        printf("GL_VERSION_2_0 GL_VERSION_1_5 found!\n");
    }

    if (!glewIsSupported("GL_ARB_multitexture") || !glewIsSupported("GL_ARB_vertex_buffer_object")) // "))
    {
        fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_ARB_multitexture\n\tGL_ARB_vertex_buffer_object\n\t\n");
  //      exit(EXIT_SUCCESS);
    } else {
        printf("Found extensions: GL_ARB_multitexture GL_ARB_vertex_buffer_object\n");
    }

    SDL_GL_SetSwapInterval(1); // turn on vertical retrace

    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &freeMemoryKb);
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totalMemoryKb);
    printf("OpenGL : %s, free memory: %d/%dMB\n",glGetString(GL_VERSION),freeMemoryKb/1024,totalMemoryKb/1024); fflush(stdout);

   // printf("initializing OpenGL...\n"),
    glViewport(0, 0, width, height);
    glDisable(GL_TEXTURE_2D);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(0,1.0f,0,1.0f,0.1f,100000.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_LIGHTING);
 //   glEnable(GL_COLOR_MATERIAL);
    // glEnable(GL_BLEND);
   // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    return glcontext;
}

void checkSDLError(int line = -1)
{
    const char *error = SDL_GetError();
    if (*error != '\0')
    {
        printf("SDL Error: %s\n", error);
        if (line != -1)
            printf(" + line: %i\n", line);
        SDL_ClearError();
    }
}

GLuint loadSplashScreen(const char *filename, int width, int height) {
    cv::Mat textureMap = imread(filename,-1);
    printf("loaded %s %d x %d x %d\n",filename,textureMap.cols,textureMap.rows,textureMap.channels());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GLuint splashTextureID = 0;
    glGenTextures(1, &splashTextureID);
    glBindTexture(GL_TEXTURE_2D, splashTextureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    if (textureMap.channels() == 3) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureMap.cols, textureMap.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, textureMap.ptr());
    else if (textureMap.channels() == 4) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureMap.cols, textureMap.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, textureMap.ptr());
    return  splashTextureID;
}

// create window in a given resolution, initialize OpenGL, fonts and subwindows
int initializeGraphics(const char *name, int &width, int &height) {
    if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Unable to initialize SDL: %s\n", SDL_GetError());
        return 0;
    }
    checkSDLError(__LINE__);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // setup anti aliasing
    //SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    //SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16);

/*    SDL_DisplayMode dm;
    if (SDL_GetCurrentDisplayMode(0, &dm) != 0) {
        SDL_Log("SDL_GetDesktopDisplayMode failed: %s", SDL_GetError());
        return 0;
    }
    width = dm.w - 2;
    height = dm.h - 2;
    printf("detected resolution : %d %d\n", width, height);
*/
    sdlWindow = SDL_CreateWindow(name, 1, 1, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_ALLOW_HIGHDPI);

    if (sdlWindow == NULL) {
        printf("Could not create window: %s\n", SDL_GetError());
        return 0;
    }
    checkSDLError(__LINE__);

    glcontext = configureGL(width, height);
    checkSDLError(__LINE__);
    SDL_ShowWindow(sdlWindow);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    GLWindow *window = NULL;
    window = new GLWindow(0, 0, (float)width, (float)height, renderFunc1); glWindows.push_back(window);
    calibKinect.setMinDist(400.0f);
    calibKinect.setMaxDist(4500.0f);
    calibKinect.setupCalibDataBuffer(RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT);
    renderer = new Renderer();
    renderer->setProjectionMatrix(&calibKinect.getCalibData()[KR_OFFSET], (float)RENDER_TARGET_WIDTH, (float)RENDER_TARGET_HEIGHT, 0.01f, 1e6f);
    renderer->setFOV(fov);
    renderer->resetView3d(0,0,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
    renderer->setLightTarget(Eigen::Vector3f(0.0, 1.0, 0.0));
    screenShot = new ScreenShot("scratch", 0, 0, RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT);

    printf("initializing fonts...\n");
    //   font.init("data/fonts/B52.ttf",32);
    font.init("data/fonts/blackWolf.ttf", 32);

    int ret = 0;
    for (int i = 0; i < NUM_RENDER_TARGETS; i++) {
        if (i == RENDER_TARGET_FINAL) ret = fbos.create(i,WINDOW_RESOX,WINDOW_RESOY);
        else ret = fbos.create(i,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
        if (!ret) return 0;
    }

    //char buf[512]; sprintf(buf,"data/loading.png");
    splashTextureID = loadSplashScreen("data/loading.png", width, height);
    return 1;
}


// initialize particle system
VoxelGrid *createVoxelGrid(int gridResolution, bool openglContext, int deviceID, float minDist, float maxDist)
{
    VoxelGrid *voxelGrid = new VoxelGrid(gridResolution, RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT, openglContext, deviceID, minDist,maxDist);
    voxelGrid->clear();
    return voxelGrid;
}

void writePPM(const char *fn, int dimx, int dimy, float *src) {
    float zcap = 50.0f;
    float minZ = 1e8f;
    float maxZ = 0.0f;
    FILE *fp = fopen(fn, "wb"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    unsigned char *buf = new unsigned char[dimx*dimy*3];
    int off4 = 0;
    for (int j = 0; j < dimy; j++) {
        int off3 = (dimy-1-j)*dimx*3;
        for (int i = 0; i < dimx; i++,off3+=3,off4+=4) {
            float zval = -src[off4+3];
            if (zval < minZ) minZ = zval;
            if (zval > maxZ) maxZ = zval;
            if (zval < 0.0f) zval = 0;
            if (zval > zcap) zval = zcap;
            zval = 255*zval/zcap;
            buf[off3+0] = zval;//src[off4+0]*255.0f;
            buf[off3+1] = zval;//src[off4+1]*255.0f;
            buf[off3+2] = zval;//src[off4+2]*255.0f;
        }
    }
    printf("minZ: %f maxZ: %f\n",minZ,maxZ);
    fwrite(buf,dimx*dimy*3,1,fp);
    delete[] buf;
    fclose(fp);
}


/*
void blitToPrimaryScreen() {
    //sdkStartTimer(&timer);
    glBindFramebufferEXT( GL_READ_FRAMEBUFFER_EXT, framebuffer ); // set target as primary backbuffer
    glBindFramebufferEXT( GL_DRAW_FRAMEBUFFER_EXT, 0 ); // set target as primary backbuffer
    glBlitFramebufferEXT(0, 0, texWidth,texHeight, 0, winHeight/2, winWidth/2, winHeight, GL_COLOR_BUFFER_BIT , GL_LINEAR);
}*/

void uploadCudaTexture(void *cudaDevData, int sizeTexBytes, cudaGraphicsResource *cudaTexture) {

    cudaArray *texture_ptr;
    cudaThreadSynchronize();
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexture, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaTexture, 0, 0));
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cudaDevData, sizeTexBytes, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexture, 0));
}


void renderImage(int fboIndex, GLuint textureId, bool flipY=false) {
    //if (textureIDs == NULL || textureId >= numTextures) return;
    fbos.bind(fboIndex);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    //printf("rendering!\n");
    renderer->resetView2d(0,0,fbos.width(fboIndex),fbos.height(fboIndex));
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    renderer->displayRGBDImage(textureId,flipY,false,0.0f,true);
}


std::string modeString(TrackMode mode, float lambda) {
    char buf[512]; sprintf(buf,"bi-objective-%1.2f",lambda);
    if (mode == PHOTOMETRIC) return std::string("photometric");
    if (mode == GEOMETRIC) return std::string("geometric");
    if (mode == BIOBJECTIVE) return std::string(buf);
    return std::string("unknown");
}

int skipNumber(TrackMode mode) {
    if (mode == GEOMETRIC) return 2;
    if (mode == PHOTOMETRIC) return 1;
    if (mode == BIOBJECTIVE) return 2;
    return 1;
}


void drawSelectedPoints(SELECTED_POINTS &selection, float w, float h, int skipper, bool refData=false) {
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_COLOR_MATERIAL);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glPointSize(3.0f);
    glColor3f(1,0,0);

    float *data = selection.selected;
    if (refData) data = selection.refSelected;
    if (data == NULL) return;

    glBegin(GL_POINTS);
    for (int i = 0; i < selection.nSelected; i++) {
        if (int(data[i*2+0])%skipper != 0) continue;
        if (int(data[i*2+1])%skipper != 0) continue;
        float c = selection.color[i];
        if (c == 0.5f) glColor3f(1,1,0);
        else glColor3f(1-c,c,0);
        glVertex3f(w*(data[i*2+0]/RENDER_TARGET_WIDTH),h-h*(data[i*2+1]/RENDER_TARGET_HEIGHT),-2.0f);
    }
    glEnd();

    glPopMatrix();
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glColor4f(1,1,1,1);
}


void updateFbos() {
    //printf("rendering!\n");
    if (showLoadingScreen) {
        renderImage(RENDER_TARGET_FINAL, splashTextureID);
    } else {
        fbos.bind(RENDER_TARGET_FINAL);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glEnable(GL_TEXTURE_2D);
        float w = fbos.width(RENDER_TARGET_FINAL)/2;
        float h = fbos.height(RENDER_TARGET_FINAL)/2;
        renderer->resetView2d(0,h,w,h);
        renderer->displayRGBDImage_old(fbos.getTexId(RENDER_TARGET_RAW),fbos.getDepthId(RENDER_TARGET_RAW));
        glColor4f(1,0,0,1); font.printTTF(0.05*w,0.9*h, -0.8f, 1.0f,1.0f,false,"INPUT");
        if (showSelectedPoints) {
            drawSelectedPoints(selectedPoints,w,h,skipNumber(photoTracker->getMode()));//skipNumber(prevMode));
        }
        renderer->resetView2d(w,h,w,h);
        renderer->displayRGBDImage_old(outputTexture,outputTexture,1);
        glColor4f(1,0,0,1); font.printTTF(0.05f*w,0.9f*h, -0.8f, 1.0f,1.0f,false,"TSDF");

        renderer->resetView2d(0,0,w,h);
        renderer->displayRGBDImage_old(refTexture1C[layer],refTexture1C[layer],1);
        glColor4f(1,0,0,1); font.printTTF(0.05f*w,0.9f*h, -0.8f, 1.0f,1.0f,false,"ref");
        if (showSelectedPoints) {
            drawSelectedPoints(selectedPoints,w,h,skipNumber(photoTracker->getMode()),true);//skipNumber(prevMode));
        }
        if (showErrorImage) {
            renderer->resetView2d(w,0,w,h);
            renderer->displayRGBDImage_old(debugTexture1C[layer],debugTexture1C[layer],1);
            glColor4f(1,0,0,1); font.printTTF(0.05f*w,0.9f*h, -0.8f, 1.0f,1.0f,false,"error");
        }
    }
}

void getNormalizedK(float *K) {
    memcpy(&K[0],&calibKinect.getCalibData()[KL_OFFSET],sizeof(float)*9);
    K[0] = K[0]/float(RENDER_TARGET_WIDTH); K[1] = 0;                      K[2] /= float(RENDER_TARGET_WIDTH);
    K[3] = 0;                    K[4] = K[4]/float(RENDER_TARGET_HEIGHT);  K[5] /= float(RENDER_TARGET_HEIGHT);
}

void resampleVoxelGrid(float *pose, float *normalizedK, float4 *xyzImage) {
    VoxelGrid *src = voxelGridArray[activeGrid];
    VoxelGrid *dst = voxelGridArray[!activeGrid];

    dst->setCameraParams(pose,normalizedK);
//    dst->resetToRayCastResult(xyzImage);
    dst->resample(src);
    identity4x4(&tsdfPose[0]);
    dst->setCameraParams(tsdfPose,normalizedK);
    voxelGrid = dst;
    activeGrid = !activeGrid;
}

void updateFreeGPUMemory() {
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &freeMemoryKb);
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totalMemoryKb);
}

void updateCamera() {
    if (cameraPos[1] < 0.0f) {
        cameraPos[1] = 0.0f;
    }
    cameraPosLag += (cameraPos - cameraPosLag) * inertia;
    cameraRotLag += (cameraRot - cameraRotLag) * inertia;
    cursorPosLag += (cursorPos - cursorPosLag) * inertia;

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(cameraRotLag[0], 1.0, 0.0, 0.0);
    glRotatef(cameraRotLag[1], 0.0, 1.0, 0.0);
    glTranslatef(-cameraPosLag[0], -cameraPosLag[1], -cameraPosLag[2]);

    float modelViewT[16];
    memcpy(&prevModelView,&modelView[0],sizeof(float)*16); // save previous modelview matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewT); transpose4x4(modelViewT,modelView);
    Eigen::Vector3f cameraDirLag; cameraDirLag(0) = -modelView[8]; cameraDirLag(1) = -modelView[9]; cameraDirLag(2) = -modelView[10];
}

void updateVoxelGrid() {
    if (!voxelGrid) return;
    if (!renderer) return;
    if (!barrier) return;
    if (!photoTracker) return;
    voxelGrid->preprocessMeasurementRGBD(rgbdFramePrev,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
//    voxelGrid->preprocessMeasurementRGBD(rgbdFrame,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);

    renderer->setCloudBuffer(voxelGrid->getCloudBuffer());
    renderer->setGridResolution(voxelGrid->getGridResolution());
    renderer->setPointSize(voxelGrid->getVoxelSize());
    renderer->setCube(voxelGrid->getCubeOrigin(),voxelGrid->getCubeDim());
    renderer->setCameraMatrix(&modelView[0]);
    renderer->resetView3d(0,0,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);

   // printf("updateVoxel grid!\n");

    barrier->sync();
    getEstimatedPose(&estimatedPose[0]); resultTrajectory.addPose(&estimatedPose[0]);
    getTSDFPose(&tsdfPose[0]);
    //dumpMatrix("tsdf pose",tsdfPose,4,4);
    float normalizedK[9];
    getNormalizedK(&normalizedK[0]);
    //if (photoTracker->getFilteredReference())
    {
        // update TSDF in using IR camera pose estimate:
        voxelGrid->update(&tsdfPose[0],normalizedK);
        // ray cast TSDF using IR camera pose estimate
        // NOTE: return maps are in RGB coordinate systems!
        cudaDeviceSynchronize();
        voxelGrid->rayCastImage(&tsdfPose[0],normalizedK,true);
        cudaDeviceSynchronize();
    }
    barrier->sync();

    if (showErrorImage) {
        photoTracker->updateErrorImage();
        uploadCudaTexture(photoTracker->getDebugImage1C(layer),(RENDER_TARGET_WIDTH>>layer)*(RENDER_TARGET_HEIGHT>>layer)*sizeof(float),cudaDebugTexture1C[layer]);
    }
    if (showSelectedPoints) {
        photoTracker->listSelected(layer,&selectedPoints.selected,&selectedPoints.refSelected,&selectedPoints.color,&selectedPoints.nSelected);
    }
    uploadCudaTexture(voxelGrid->getRayCastImage(),RENDER_TARGET_WIDTH*RENDER_TARGET_HEIGHT*sizeof(float4),cudaOutputTexture);
    uploadCudaTexture(photoTracker->getRefImage1C(layer),(RENDER_TARGET_WIDTH>>layer)*(RENDER_TARGET_HEIGHT>>layer)*sizeof(float),cudaRefTexture1C[layer]);

    TRACK_PARAMS *params = trackParams.getParams(currentFrame);
   // printf("found params for frame %d, seekframe: %d\n",params->frame,currentFrame); fflush(stdout);
    TrackMode mode = PHOTOMETRIC;//BIOBJECTIVE;//GEOMETRIC;//PHOTOMETRIC;//BIOBJECTIVE;//GEOMETRIC;//PHOTOMETRIC;//BIOBJECTIVE;//PHOTOMETRIC;// GEOMETRIC;//PHOTOMETRIC;//GEOMETRIC;//params->trackMode;
    float lambda = 0.75f;//params->lambda;
    // update reference frame using ray casted rgb-d
    // mikÃ¤ on referenssikuvan pose phototrackerin sisÃ¤llÃ¤?
    if (restartTrackingFlag) {
        photoTracker->setReferencePoints(voxelGrid->getRayCastXYZImage(),voxelGrid->getRayCastNormalImage(),rgbdFrame,mode,lambda);
    } else {
        if(photoTracker->setReferencePoints(voxelGrid->getRayCastXYZImage(),voxelGrid->getRayCastNormalImage(),rgbdFramePrev,mode,lambda)) {
            if (useKinectInput && relativeTSDF) {
                resampleVoxelGrid(&tsdfPose[0],normalizedK,voxelGrid->getRayCastXYZImage());
                photoTracker->resetToTransform(photoTracker->getIncrement());
            }
        }
        photoTracker->setCurrentImage(rgbdFrame);
    }

}

void updateFrame(bool newDataset) {
    if (!voxelGrid) return;
    if (!renderer) return;
    if (!fbos.count()) return;

    fbos.bind(RENDER_TARGET_RAW);

    renderer->setCloudBuffer(voxelGrid->getCloudBuffer());
    renderer->setGridResolution(voxelGrid->getGridResolution());
    renderer->setPointSize(voxelGrid->getVoxelSize());
    renderer->setCube(voxelGrid->getCubeOrigin(),voxelGrid->getCubeDim());
    renderer->setCameraMatrix(&modelView[0]);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    renderer->setFOV(fov);
    // update current frame
    if (!useKinectInput) {
        //printf("synthetic.\n");
        renderer->resetView3d(0,0,fbos.width(RENDER_TARGET_RAW),fbos.height(RENDER_TARGET_RAW));
        renderer->drawTestScene3D(boxTex,calibKinect.getMinDist(),calibKinect.getMaxDist());
    } else {
        //printf("measured.\n");
        updateSensorTextures();
        renderer->resetView2d(0,0,fbos.width(RENDER_TARGET_RAW),fbos.height(RENDER_TARGET_RAW));
        // note: this function does not correctly align the images, merely combines rgb-d data from 2 source textures
        renderer->displayRGBDImage_old(sensorRGBTex,sensorDepthTex,1);
    }
    glFlush();
    fbos.unbind();
    currentFrame++;

    cudaStreamSynchronize(0);
    cudaArray *mappedFrame = NULL;
    fbos.lock(RENDER_TARGET_RAW,&mappedFrame);
    // save the previous rgbdFrame to get frame delay - 1 for voxelupdates/raycasting
    if (!newDataset) {
        cudaMemcpy2DArrayToArray(rgbdFramePrev,0,0,rgbdFrame,0,0,RENDER_TARGET_WIDTH*sizeof(float4),RENDER_TARGET_HEIGHT,cudaMemcpyDeviceToDevice);
        // update rgbdFrame to phototracker as the "current frame"
        cudaMemcpy2DArrayToArray(rgbdFrame,0,0,mappedFrame,0,0,RENDER_TARGET_WIDTH*sizeof(float4),RENDER_TARGET_HEIGHT,cudaMemcpyDeviceToDevice);
    } else {
        // update rgbdFrame to phototracker as the "current frame"
        cudaMemcpy2DArrayToArray(rgbdFramePrev,0,0,mappedFrame,0,0,RENDER_TARGET_WIDTH*sizeof(float4),RENDER_TARGET_HEIGHT,cudaMemcpyDeviceToDevice);
        cudaMemcpy2DArrayToArray(rgbdFrame,0,0,mappedFrame,0,0,RENDER_TARGET_WIDTH*sizeof(float4),RENDER_TARGET_HEIGHT,cudaMemcpyDeviceToDevice);
    }
    cudaStreamSynchronize(0);
    fbos.unlock();
}

void updateKeys() {
    if (keyDown[SDLK_w]) {
        cameraPos[0] += -modelView[8]  * walkSpeed;
        cameraPos[1] += -modelView[9]  * walkSpeed;
        cameraPos[2] += -modelView[10] * walkSpeed;
    }

    if (keyDown[SDLK_s]) {
        cameraPos[0] -= -modelView[8] * walkSpeed;
        cameraPos[1] -= -modelView[9] * walkSpeed;
        cameraPos[2] -= -modelView[10] * walkSpeed;
    }

    if (keyDown[SDLK_a]) {
        cameraPos[0] -= modelView[0] * walkSpeed;
        cameraPos[1] -= modelView[1] * walkSpeed;
        cameraPos[2] -= modelView[2] * walkSpeed;
    }

    if (keyDown[SDLK_d]) {
        cameraPos[0] += modelView[0] * walkSpeed;
        cameraPos[1] += modelView[1] * walkSpeed;
        cameraPos[2] += modelView[2] * walkSpeed;
    }


    if (keyDown[SDLK_e]) {
        cameraPos[0] += modelView[4] * walkSpeed;
        cameraPos[1] += modelView[5] * walkSpeed;
        cameraPos[2] += modelView[6] * walkSpeed;
    }

    if (keyDown[SDLK_q]) {
        cameraPos[0] -= modelView[4] * walkSpeed;
        cameraPos[1] -= modelView[5] * walkSpeed;
        cameraPos[2] -= modelView[6] * walkSpeed;
    }
}

void update() {
    updateFreeGPUMemory();
    if (restartTrackingFlag) {
        resetVariables();
    }
    if (!showLoadingScreen) {
        updateKeys();
        updateCamera();
    }
    if (photoTracker && photoTracker->isActive()) {
        updateFrame(restartTrackingFlag);
        restartTrackingFlag = false;
    }
    updateVoxelGrid();

    updateFbos();

}

// render all subwindows
void renderFrame()
{
    fbos.unbind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (size_t i = 0; i < glWindows.size(); i++) glWindows[i]->render();
    glFlush();
    if (screenShot && saveNextFrame) {
        fbos.bind(RENDER_TARGET_FINAL);
        screenShot->save(); saveNextFrame = false;
        fbos.unbind();
    }
    SDL_GL_SwapWindow(sdlWindow);
}


// simple user interface
void handleKeyUp(int key, bool &running) {
    switch(key) {
        case SDLK_t:
            useKinectInput = !useKinectInput;
            setCalibration(useKinectInput);
            restartTrackingFlag = true;
            break;
        case SDLK_l:
            if (photoTracker && photoTracker->getLayers() > 1) layer = (layer+1)%photoTracker->getLayers();
            break;
        case SDLK_ESCAPE:
            running = false;
            break;
        case SDLK_m:
            if (photoTracker && photoTracker->isActive()) {
               /* TrackMode mode = photoTracker->getMode();
                if (mode == GEOMETRIC) mode = PHOTOMETRIC;
                else if (mode == PHOTOMETRIC) mode = BIOBJECTIVE;
                else if (mode == BIOBJECTIVE) mode = GEOMETRIC;
                photoTracker->setMode(mode);*/
            }
        break;
    }
    if (key < 256) keyDown[key] = false;
}

void handleKeyDown(int key) {
    if (key < 256) keyDown[key] = true;
}

GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(target, tex);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    return tex;
}

GLuint loadTexture(const char *filename)
{
    unsigned char *data = 0;
    unsigned int width, height;
    sdkLoadPPM4ub(filename, &data, &width, &height);
    if (!data)
    {
        printf("Error opening file '%s'\n", filename);
        return 0;
    }
    printf("Loaded '%s', %d x %d pixels\n", filename, width, height);
    return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}

int loaderThread(void * /*ptr*/) {
    // notify main thread after video has been inited (correct width&height are known)
    loading = true;

    fileSource   = new FileSource("data/sequence",true);
    preprocessor = new VideoPreProcessorCPU(fileSource,3,calibKinect,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
    resultTrajectory.init("data/cameraMatrixEst.txt",true);
    trackParams.init("data/best-method.tst");

    loading = false;
    loadingFinished = true;
    printf("loader thread finished!\n");
    return 1;
}


void loadTextures() {
    // load floor texture
   floorTex = loadTexture((const char*)"data/floortile.ppm");
   boxTex   = loadTexture((const char*)"data/skull.ppm");
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
}

int glLoader() {
    if (!loadingFinished) return 1;
    if (failureInLoading || g_killThreads) {
        running = false;
        printf("main loop: loader has failed! exiting!\n"); fflush(stdout);
        return 0;
    }

    if (!cudaContextExists) {
        if (!setupCudaDevices(2)) return 0;        
        printf("cuda contexts created!\n"); fflush(stdout);
         for (int i = 0; i < NUM_RENDER_TARGETS; i++) fbos.allowCuda(i);
        // create opengl texture that will receive the result of CUDA
        createCudaTexture(&outputTexture, false, RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT,&cudaOutputTexture);
        createCudaTexture(&debugTexture, false, RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT, &cudaDebugTexture);
        for (int i = 0; i < 4; i++) {
            createCudaTexture(&debugTexture1C[i], true, RENDER_TARGET_WIDTH>>i, RENDER_TARGET_HEIGHT>>i, &cudaDebugTexture1C[i]);
            createCudaTexture(&refTexture1C[i], true, RENDER_TARGET_WIDTH>>i, RENDER_TARGET_HEIGHT>>i, &cudaRefTexture1C[i]);
        }
        // create texture for blitting onto the screen
        createTextureSrc(&inputTexture, RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
        createSensorTextures(&sensorRGBTex, &sensorDepthTex, RENDER_TARGET_WIDTH, RENDER_TARGET_HEIGHT);
        cudaContextExists = true;
        return 1;
    }

    if (cudaContextExists && photoTracker == NULL) {
        photoTracker = initializePhotoTracker(1,RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT,&calibKinect);
        photoTracker->resetTracking();
        photoTracker->setReferenceUpdating(true);
        photoTracker->setFilteredReference(false);//true);
        photoTracker->setIncrementalMode(true);
        if (cudaContextExists && photoTracker && voxelGrid == NULL) {
            voxelGridArray[0] = createVoxelGrid(g_gridResolution,true,0,calibKinect.getMinDist(),calibKinect.getMaxDist());
            if (relativeTSDF) {
                voxelGridArray[1] = createVoxelGrid(g_gridResolution,true,0,calibKinect.getMinDist(),calibKinect.getMaxDist());
            }
            activeGrid = 0;
            voxelGrid = voxelGridArray[activeGrid];

            loadTextures();
            showLoadingScreen = false;
        }
        setCalibration(useKinectInput);
    }
    return 1;
}

int loop() {
    PerformanceCounter timer;
    loaderThreadHandle = SDL_CreateThread(loaderThread, "loaderThread", NULL);

    int frame = 0;
    float frameTime = 0.0f;
    while ( running ) {
        timer.StartCounter();
        mouseDx = Eigen::Vector3f(0,0,1);
        if(!glLoader()) return 0;

        // process input
        SDL_Event event;
        while ( SDL_PollEvent(&event)){
            if (event.type == SDL_QUIT)        running = false;
            if (!loading) {
               if (event.type == SDL_MOUSEMOTION) handleMouseMotion(event.motion);
               if (event.type == SDL_KEYUP)       handleKeyUp(event.key.keysym.sym,running);              
               if (event.type == SDL_KEYDOWN)     handleKeyDown(event.key.keysym.sym);
               if (event.type == SDL_MOUSEBUTTONDOWN) {   buttonState = 1;}
               if (event.type == SDL_MOUSEBUTTONUP)   {   buttonState = 0;}
            }
        }
        update();
        renderFrame();
        frame++;
        timer.StopCounter();
        float elapsedTime = (float)timer.GetElapsedTime()*1000.0f;
        frameTime = 0.9f*frameTime + 0.1f*elapsedTime;
        double delay = (33-elapsedTime); if (delay < 0) delay = 0;
        sleepMs(delay);
    }
    return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n\n", projectName);

    if (!initializeGraphics("rgbdscan", WINDOW_RESOX, WINDOW_RESOY)) {
          return 0;
     }
    calibKinect.init("data/calib.xml",false);
    calibKinect.setupCalibDataBuffer(RENDER_TARGET_WIDTH,RENDER_TARGET_HEIGHT);
    calibKinect.setMinDist(400.0f);
    calibKinect.setMaxDist(3500.0f);
    loop();
    release();
    return 1;
}

