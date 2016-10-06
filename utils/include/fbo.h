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
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
class FBO {
public:
    FBO(int numfbos);
    ~FBO();
    int  create(unsigned int index, unsigned int resox, unsigned int resoy);
    void bind(int fboIndex);
    void unbind();
    int  release();
    void blit(int fboIndex, int x0, int y0, int rectWidth, int rectHeight);
    void blit(int fboIndex, int fboIndexDst);
    int width(int fboIndex);
    int height(int fboIndex);
    GLuint getTexId(int fboIndex);
    GLuint getDepthId(int fboIndex);
    void lock(int fboIndex,cudaArray **mappedFrame);
    void unlock();
    int count();
    void allowCuda(int fboIndex);
private:
    GLuint *framebuffer;           // to bind the proper targets
    GLuint *depthTargetTexture;    // for proper depth test while rendering the scene
    GLuint *rgbTargetTexture;      // where we render the image
    int *RENDER_TARGET_WIDTH;
    int *RENDER_TARGET_HEIGHT;
    int NUM_RENDER_TARGETS;        // 3 default views
    int  createFramebuffer(GLuint *fbo, GLuint color, GLuint depth);
    void deleteFramebuffer(GLuint *fbo);
    void createRenderTargetTexture(GLuint *tex_screen, unsigned int size_x, unsigned int size_y);
    void deleteTexture(GLuint *tex, cudaGraphicsResource_t &cudaTex);
    void createDepthBuffer(GLuint *depth, unsigned int size_x, unsigned int size_y);
    void deleteDepthBuffer(GLuint *depth);
    cudaGraphicsResource_t *cudaTexture;
    int prevLockedIndex;
};
