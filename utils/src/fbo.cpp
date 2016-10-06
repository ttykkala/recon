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

#include <fbo.h>
#include <stdio.h>

#define SAFE_RELEASE(x) { if (x!=NULL) delete x; x = NULL; }
#define SAFE_RELEASE_ARRAY(x) { if (x!=NULL) delete[] x; x = NULL; }

FBO::FBO(int count) {
	NUM_RENDER_TARGETS   = count;
    framebuffer          = new GLuint[NUM_RENDER_TARGETS];
    depthTargetTexture   = new GLuint[NUM_RENDER_TARGETS];
    rgbTargetTexture     = new GLuint[NUM_RENDER_TARGETS];
    cudaTexture          = new cudaGraphicsResource_t[NUM_RENDER_TARGETS];
    RENDER_TARGET_WIDTH  = new int[NUM_RENDER_TARGETS];
    RENDER_TARGET_HEIGHT = new int[NUM_RENDER_TARGETS];
    prevLockedIndex      = -1;
    for (int i = 0; i < NUM_RENDER_TARGETS; i++) {
        framebuffer[i]          = (GLuint)0;
        depthTargetTexture[i]   = (GLuint)0;
        rgbTargetTexture[i]     = (GLuint)0;
        cudaTexture[i]          = NULL;
        RENDER_TARGET_WIDTH[i]  = 0;
        RENDER_TARGET_HEIGHT[i] = 0;
    }
}
void FBO::blit(int fboIndex, int x0, int y0, int rectWidth, int rectHeight) {
    glBindFramebufferEXT( GL_READ_FRAMEBUFFER_EXT, framebuffer[fboIndex] ); // set target as primary backbuffer
    glBindFramebufferEXT( GL_DRAW_FRAMEBUFFER_EXT, 0 );                     // set target as primary backbuffer
    glBlitFramebufferEXT(0, 0, RENDER_TARGET_WIDTH[fboIndex], RENDER_TARGET_HEIGHT[fboIndex], x0, y0, x0+rectWidth, y0+rectHeight, GL_COLOR_BUFFER_BIT , GL_LINEAR);
    glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, 0);
}

void FBO::blit(int fboIndex, int fboIndexDst) {
    glBindFramebufferEXT( GL_READ_FRAMEBUFFER_EXT, framebuffer[fboIndex] ); // set target as primary backbuffer
    glBindFramebufferEXT( GL_DRAW_FRAMEBUFFER_EXT, framebuffer[fboIndexDst] );                     // set target as primary backbuffer
    glBlitFramebufferEXT(0, 0, RENDER_TARGET_WIDTH[fboIndex], RENDER_TARGET_HEIGHT[fboIndex], 0, 0, RENDER_TARGET_WIDTH[fboIndexDst], RENDER_TARGET_HEIGHT[fboIndexDst], GL_COLOR_BUFFER_BIT , GL_LINEAR);
    glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, 0);
    unbind();
}

void FBO::unbind() {
    // set default frame buffer:
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

int FBO::createFramebuffer(GLuint *fbo, GLuint color, GLuint depth)
{
    printf("Creating a Framebuffer GL_RGBA32F_ARB\n"); fflush(stdout);

    // create and bind a framebuffer
    glGenFramebuffersEXT(1, fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fbo);

    // attach images
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color, 0);
    //glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, color);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth);

    GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, drawBuffers); // "1" is the size of drawBuffers

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("render target creation failed!.. exiting!\n"); fflush(stdout);
        return 0;
    }
    // clean up
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    return 1;
}

int FBO::count() {
    return NUM_RENDER_TARGETS;
}

void FBO::createRenderTargetTexture(GLuint *tex_screen, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_screen);
    glBindTexture(GL_TEXTURE_2D, *tex_screen);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
//#ifndef USE_TEXTURE_RGBA8UI
    printf("Creating a Texture render target GL_RGBA32F_ARB\n"); fflush(stdout);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);       

    //printf("Creating a Texture render target GL_RGBA16F_ARB\n");
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
/*#else
    printf("Creating a Texture render target GL_RGBA8UI_EXT\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
#endif
*/
    //SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    //checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *tex_screen, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}

void FBO::createDepthBuffer(GLuint *depth, unsigned int size_x, unsigned int size_y)
{
    printf("Creating a Depth Texture render target GL_RGBA32F_ARB\n"); fflush(stdout);

    // create a renderbuffer
    glGenRenderbuffersEXT(1, depth);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);
    // allocate storage
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);
    // clean up
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

int FBO::create(unsigned int index, unsigned int resox, unsigned int resoy) {
	if (index >= NUM_RENDER_TARGETS) return 0;
    createRenderTargetTexture(&rgbTargetTexture[index], resox, resoy);
    createDepthBuffer(&depthTargetTexture[index], resox, resoy);
    if (!createFramebuffer(&framebuffer[index],rgbTargetTexture[index], depthTargetTexture[index])) {
		return 0;
    }
    printf("Setting target sizes\n"); fflush(stdout);

	RENDER_TARGET_WIDTH[index]  = resox;
	RENDER_TARGET_HEIGHT[index] = resoy;	
    printf("Done\n"); fflush(stdout);
	return 1;
}


void FBO::allowCuda(int fboIndex) {
    if (fboIndex >= NUM_RENDER_TARGETS) return;
    // register this texture with CUDA
    glBindTexture(GL_TEXTURE_2D, rgbTargetTexture[fboIndex]);
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTexture[fboIndex], rgbTargetTexture[fboIndex], GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}


GLuint FBO::getTexId(int fboIndex) {
    return rgbTargetTexture[fboIndex];
}

GLuint FBO::getDepthId(int fboIndex) {
    return depthTargetTexture[fboIndex];
}


void FBO::deleteTexture(GLuint *tex, cudaGraphicsResource_t &cudaTex)
{
    if (cudaTex != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaTex));
    glDeleteTextures(1, tex);
    *tex = 0;
}

void FBO::deleteDepthBuffer(GLuint *depth)
{
    glDeleteRenderbuffersEXT(1, depth);
    *depth = 0;
}

void FBO::deleteFramebuffer(GLuint *fbo)
{
    glDeleteFramebuffersEXT(1, fbo);
    *fbo = 0;
}

int FBO::width(int fboIndex) {
    if (fboIndex < 0 || fboIndex >= NUM_RENDER_TARGETS) return 0;
    return RENDER_TARGET_WIDTH[fboIndex];
}

int FBO::height(int fboIndex) {
    if (fboIndex < 0 || fboIndex >= NUM_RENDER_TARGETS) return 0;
    return RENDER_TARGET_HEIGHT[fboIndex];
}


int FBO::release() {
    for (int i = 0; i < NUM_RENDER_TARGETS; i++) {
        if (rgbTargetTexture[i])     deleteTexture(&rgbTargetTexture[i],cudaTexture[i]);
        if (depthTargetTexture[i])   deleteDepthBuffer(&depthTargetTexture[i]);
        if (framebuffer[i])			 deleteFramebuffer(&framebuffer[i]);
    }
    return 1;
}

FBO::~FBO() {
	SAFE_RELEASE_ARRAY(framebuffer);
	SAFE_RELEASE_ARRAY(depthTargetTexture);
	SAFE_RELEASE_ARRAY(rgbTargetTexture);
    SAFE_RELEASE_ARRAY(cudaTexture);
	SAFE_RELEASE_ARRAY(RENDER_TARGET_WIDTH);
	SAFE_RELEASE_ARRAY(RENDER_TARGET_HEIGHT);	
}

void FBO::bind(int fboIndex) {
 	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer[fboIndex]);
}

void FBO::lock(int fboIndex,cudaArray **mappedFrame) {
    // map buffer objects to get CUDA device pointers
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexture[fboIndex], 0));
    //printf("Mapping tex_in\n");
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(mappedFrame, cudaTexture[fboIndex], 0, 0));
    prevLockedIndex = fboIndex;
}

void FBO::unlock() {
    if (prevLockedIndex < 0 || prevLockedIndex >= NUM_RENDER_TARGETS) return;
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexture[prevLockedIndex], 0));
}

