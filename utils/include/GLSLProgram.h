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

// Simple class to contain GLSL shaders/programs
#pragma once

#include <GL/glew.h>
#include <stdio.h>

class GLSLProgram
{
    public:
        // construct program from strings
        GLSLProgram(const char *vsource, const char *fsource);
        GLSLProgram(const char *vsource, const char *gsource, const char *fsource,
                    GLenum gsInput = GL_POINTS, GLenum gsOutput = GL_TRIANGLE_STRIP, int nVertices=4);
        ~GLSLProgram();

        void enable();
        void disable();

        void setUniform1f(const GLchar *name, GLfloat x);
        void setUniform2f(const GLchar *name, GLfloat x, GLfloat y);
        void setUniform3f(const char *name, float x, float y, float z);
        void setUniform4f(const char *name, float x, float y, float z, float w);
        void setUniformfv(const GLchar *name, GLfloat *v, int elementSize, int count=1);
        void setUniformMatrix4fv(const GLchar *name, GLfloat *m, bool transpose);

        void bindTexture(const char *name, GLuint tex, GLenum target, GLint unit);

        inline GLuint getProgId()
        {
            return mProg;
        }

    private:
        GLuint checkCompileStatus(GLuint shader, GLint *status);
        GLuint compileProgram(const char *vsource, const char *gsource, const char *fsource,
                              GLenum gsInput = GL_POINTS, GLenum gsOutput = GL_TRIANGLE_STRIP, int nVertices=4);
        GLuint mProg;
};
