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

#include "Sphere.h"
#include <math.h>

//#define M_PI (3.141592653f)
//#define M_PI_2 (3.141592653f*2.0f)

Sphere::Sphere(float radius, unsigned int slices, unsigned int stacks)
{
    quadric = gluNewQuadric();
    gluQuadricNormals(quadric, GLU_SMOOTH);   // Create Smooth Normals ( NEW )
    gluQuadricTexture(quadric, GL_TRUE);      // Create Texture Coords ( NEW )

    this->radius = radius;
    this->stacks = stacks;
    this->slices = slices;
/*    float const R = 1./(float)(rings-1);
    float const S = 1./(float)(sectors-1);
    int r, s;

    sphere_vertices.resize(rings * sectors * 3);
    sphere_normals.resize(rings * sectors * 3);
    sphere_texcoords.resize(rings * sectors * 2);
    std::vector<GLfloat>::iterator v = sphere_vertices.begin();
    std::vector<GLfloat>::iterator n = sphere_normals.begin();
    std::vector<GLfloat>::iterator t = sphere_texcoords.begin();
    for(r = 0; r < rings; r++) for(s = 0; s < sectors; s++) {
            float const y = sin( -M_PI_2 + M_PI * r * R );
            float const x = cos(2*M_PI * s * S) * sin( M_PI * r * R );
            float const z = sin(2*M_PI * s * S) * sin( M_PI * r * R );

            *t++ = s*S;
            *t++ = r*R;

            *v++ = x * radius;
            *v++ = y * radius;
            *v++ = z * radius;

            *n++ = x;
            *n++ = y;
            *n++ = z;
    }

    sphere_indices.resize(rings * sectors * 4);
    std:vector<GLushort>::iterator i = sphere_indices.begin();
    for(r = 0; r < rings; r++) for(s = 0; s < sectors; s++) {
            *i++ = r * sectors + s;
            *i++ = r * sectors + (s+1);
            *i++ = (r+1) * sectors + (s+1);
            *i++ = (r+1) * sectors + s;
    }*/
}

Sphere::~Sphere() {
    gluDeleteQuadric(quadric);
}

void Sphere::draw(GLfloat x, GLfloat y, GLfloat z, float r, float g, float b) {
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT);

    glPushMatrix();
    glTranslatef(x,y,z);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColor3f(r,g,b);
    gluSphere(quadric, radius, slices, stacks);
    glPopMatrix();

    glPopAttrib();
}
