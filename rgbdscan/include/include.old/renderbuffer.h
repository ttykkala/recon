/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/


#ifndef UCDAVIS_RENDER_BUFFER_H
#define UCDAVIS_RENDER_BUFFER_H

#include "framebufferObject.h"

/*!
Renderbuffer Class. This class encapsulates the Renderbuffer OpenGL
object described in the FramebufferObject (FBO) OpenGL spec.
See the official spec at:
    http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt
for complete details.

A "Renderbuffer" is a chunk of GPU memory used by FramebufferObjects to
represent "traditional" framebuffer memory (depth, stencil, and color buffers).
By "traditional," we mean that the memory cannot be bound as a texture.
With respect to GPU shaders, Renderbuffer memory is "write-only." Framebuffer
operations such as alpha blending, depth test, alpha test, stencil test, etc.
read from this memory in post-fragement-shader (ROP) operations.

The most common use of Renderbuffers is to create depth and stencil buffers.
Note that as of 7/1/05, NVIDIA drivers to do not support stencil Renderbuffers.

Usage Notes:
  1) "internalFormat" can be any of the following:
      Valid OpenGL internal formats beginning with:
        RGB, RGBA, DEPTH_COMPONENT

      or a stencil buffer format (not currently supported
      in NVIDIA drivers as of 7/1/05).
        STENCIL_INDEX1_EXT
        STENCIL_INDEX4_EXT
        STENCIL_INDEX8_EXT
        STENCIL_INDEX16_EXT
*/
class Renderbuffer
{
    public:
        /// Ctors/Dtors
        Renderbuffer();
        Renderbuffer(GLenum internalFormat, int width, int height);
        ~Renderbuffer();

        void   Bind();
        void   Unbind();
        void   Set(GLenum internalFormat, int width, int height);
        GLuint GetId() const;

        static GLint GetMaxSize();

    private:
        GLuint m_bufId;
        static GLuint _CreateBufferId();
};

#endif

