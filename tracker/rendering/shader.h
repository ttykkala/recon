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


#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library
#include <fstream>

using namespace std;

class Shader {
public:
    Shader(const char *vsFilename, const char *fsFilename);
    void bind();
    void unbind();
    void release();
    int getAttrib(const char *name);
    void setUniformVec4(const char *name, const float *vec4);
    ~Shader();
private:
    GLhandleARB vs, fs, program; // handles to objects
    char *loadSource(const char* filename);
    int unloadSource(GLcharARB** ShaderSource);
};


#endif // SHADER_H
