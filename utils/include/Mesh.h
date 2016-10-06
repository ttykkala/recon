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


#ifndef MESH_H
#define MESH_H

#include <Eigen/Geometry>

class Mesh {
public:
    Mesh();
    ~Mesh();
    int init(const char *filename);
    unsigned int vbo;
    unsigned int ibo;
    int nfaces,npoints;
    bool glInited;
    bool isInited();
    int initGL();
private:
    int *faces3;
    float *vertices;
    int readLine(FILE *f, char *linebuf, int bufsize);
    int parsePLY(const char *filename, float **vertexOut, int &nverts, int **faceOut, int &nfaces);
    int allocateAndUpdateVbo();
    int allocateAndUpdateIbo();
    void releaseVbo();
    void transformMesh(float *vertices, int nverts, Eigen::Matrix4f &T, float dataScale=1.0f);
};


#endif // MESH_H
