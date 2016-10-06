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


#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library

class Sphere
{
protected:
    GLUquadricObj *quadric;
    float radius;
    int slices,stacks;
public:
    Sphere(float radius, unsigned int slices, unsigned int stacks);
    ~Sphere();
    void draw(GLfloat x, GLfloat y, GLfloat z, float r, float g, float b);
};
