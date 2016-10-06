#include <GL/glew.h>
#include <stdio.h>
#include <string.h>
#include <Mesh.h>
#include <Eigen/Geometry>

// two helper functions for clean release code:
#define SAFE_RELEASE(x) { if (x != NULL) {delete x; x = NULL;} }
#define SAFE_RELEASE_ARRAY(x) { if (x != NULL) {delete[] x; x = NULL;} }

Mesh::Mesh() {
    faces3 = NULL;
    vertices = NULL;
    vbo = 0;
    ibo = 0;
    npoints = 0;
    nfaces  = 0;
    glInited = false;
}

Mesh::~Mesh() {
    releaseVbo();
    SAFE_RELEASE_ARRAY(faces3);
    SAFE_RELEASE_ARRAY(vertices);
}

void Mesh::releaseVbo() {
    if (vbo > 0) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    if (ibo > 0) {
        glDeleteBuffers(1, &ibo);
        ibo = 0;
    }
}

int Mesh::allocateAndUpdateVbo() {
    if (vertices == NULL || npoints < 1) return 0;
    int requestedBytes = npoints * sizeof(float) * 6;
    glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, requestedBytes, vertices, GL_STATIC_DRAW);//GL_STREAM_COPY);
    GLenum errorCode = glGetError();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (vbo == 0 || (errorCode != GL_NO_ERROR)) {
        printf("vbo allocation failed! (requested for %dMB)\n", (requestedBytes)/(1024*1024));//,allocatedBytes/(1024*1024));
        if (errorCode == GL_OUT_OF_MEMORY) printf("out of memory!\n");
        fflush(stdout);
        return 0;
    } else {
        //allocatedBytes += requestedBytes;
        //printf("vbos allocate: %d\n",allocatedBytes/(1024*1024)); fflush(stdout);
    }
    return 1;
}

int Mesh::allocateAndUpdateIbo() {
    if (faces3 == NULL || nfaces < 1) return 0;
    int requestedBytes = nfaces*3*sizeof(unsigned int);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, requestedBytes, &faces3[0], GL_STATIC_DRAW);
    GLenum errorCode = glGetError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    if (ibo == 0 || (errorCode != GL_NO_ERROR)) {
        printf("ibo allocation failed! (requested for %dMB)\n", (requestedBytes)/(1024*1024));//,allocatedBytes/(1024*1024));
        if (errorCode == GL_OUT_OF_MEMORY) printf("out of memory!\n");
        fflush(stdout);
        return 0;
    } else {
        //allocatedBytes += requestedBytes;
        //printf("vbos allocate: %d\n",allocatedBytes/(1024*1024)); fflush(stdout);
    }
    return 1;
}

void Mesh::transformMesh(float *pointData, int nverts, Eigen::Matrix4f &R, float dataScale) {
    // a hack to transform current ply format to OpenGL coordinate system
    for (int i = 0; i < npoints; i++) {
        float *p = &pointData[i*6];
        Eigen::Vector4f v(p[0],p[1],p[2],1);
        Eigen::Vector3f n(p[3],p[4],p[5]);
        Eigen::Vector4f w = dataScale*R*v;
        Eigen::Vector3f wn = dataScale*R.block(0,0,3,3)*n;
        p[0] =   w(0);
        p[1] =   w(1);
        p[2] =   w(2);
        p[3] =   wn(0);
        p[4] =   wn(1);
        p[5] =   wn(2);
    }

}

int Mesh::init(const char *filename) {
    if (!parsePLY(filename,&vertices,npoints,&faces3,nfaces)) return 0;
    Eigen::Matrix4f R;
    R(0,0) = 1; R(0,1) =       0; R(0,2) =      0; R(0,3) =  0;
    R(1,0) = 0; R(1,1) =       0; R(1,2) =      1; R(1,3) =  0;
    R(2,0) = 0; R(2,1) =      -1; R(2,2) =      0; R(2,3) =  0;
    R(3,0) = 0; R(3,1) =       0; R(3,2) =      0; R(3,3) =  1;
    transformMesh(vertices,npoints,R,1.0f);
    return 1;
}


bool Mesh::isInited() {
    return glInited;
}

int Mesh::initGL() {
    if (glInited) return 0;
    if (!allocateAndUpdateVbo()) return 0;
    if (!allocateAndUpdateIbo()) return 0;
    glInited = true;
    return 1;
}

int Mesh::readLine(FILE *f, char *linebuf, int bufsize) {
    int off = 0;
    memset(linebuf,0,bufsize);
    while (1) {
        int c = fgetc(f);
        if (c == EOF) break;
        if (c == '\n') break;
        linebuf[off] = char(c); off++;
    }
    return off > 0;
}

int Mesh::parsePLY(const char *filename, float **vertexOut, int &nverts, int **faceOut, int &nfaces) {
    *vertexOut = 0; *faceOut = 0; nverts = 0; nfaces = 0;
    FILE *f = fopen(filename,"rb");
    if (f == NULL) return 0;

    char linebuf[2048],temp1[2048],temp2[2048];
    int xyzflag[3]    = {0,0,0};
    int normalflag[3] = {0,0,0};
    int colorflag[3]  = {0,0,0};
    int nattrib = 0;

    // parse header
    while (readLine(f,&linebuf[0],2048)) {
        if (strncmp(linebuf,"ply",3)==0) continue;
        if (strncmp(linebuf,"format",6)==0) continue;
        if (strncmp(linebuf,"comment",7)==0) continue;
        if (strncmp(linebuf,"obj_info",8)==0) continue;
        if (strncmp(linebuf,"property float x",16)==0) { xyzflag[0] = true; nattrib++; continue; }
        if (strncmp(linebuf,"property float y",16)==0) { xyzflag[1] = true; nattrib++; continue; }
        if (strncmp(linebuf,"property float z",16)==0) { xyzflag[2] = true; nattrib++; continue; }
        if (strncmp(linebuf,"property float nx",17)==0) { normalflag[0] = true; nattrib++; continue; }
        if (strncmp(linebuf,"property float ny",17)==0) { normalflag[1] = true; nattrib++; continue; }
        if (strncmp(linebuf,"property float nz",17)==0) { normalflag[2] = true; nattrib++; continue; }
        if (strncmp(linebuf,"element vertex ",15)==0) {
            sscanf(linebuf,"%s %s %d\n",&temp1[0],&temp2[0],&nverts);
            continue;
        }
        if (strncmp(linebuf,"element face ",13)==0) {
            sscanf(linebuf,"%s %s %d\n",&temp1[0],&temp2[0],&nfaces);
            continue;
        }
        if (strncmp(linebuf,"property list ",14)==0) continue;
        if (strncmp(linebuf,"end_header",10)==0) break;
    }
    printf("num vertices: %d, num faces: %d, nattrib: %d\n",nverts,nfaces,nattrib); fflush(stdout);
    if (nverts == 0 || nfaces == 0 || nattrib != 6) {
        printf("no vertices or faces or wrong attribute count (should be %d)!\n",nattrib); fflush(stdout); fclose(f);
        return 0;
    }

    float *pointData = new float[nattrib*nverts];
    int   *faceData  = new int[3*nfaces];
    int error = 0;
    for (int i = 0; i < nverts; i++) {
        if (!readLine(f,&linebuf[0],2048)) {
            printf("corrupted PLY file!\n"); fflush(stdout);
            error = 1;
            break;
        }
        float *p = &pointData[i*nattrib];
        if (nattrib == 3) {
            sscanf(&linebuf[0],"%f %f %f",&p[0],&p[1],&p[2]);
        } else if (nattrib == 6) {
            sscanf(&linebuf[0],"%f %f %f %f %f %f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5]);
        } else if (nattrib == 9) {
            sscanf(&linebuf[0],"%f %f %f %f %f %f %f %f %f",&p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6],&p[7],&p[8]);
        }
    }
    for (int i = 0; i < nfaces; i++) {
        if (!readLine(f,&linebuf[0],2048)) {
            printf("corrupted PLY file!\n"); fflush(stdout);
            break;
        }
        int *f = &faceData[i*3];
        int numCorners = 0;
        sscanf(&linebuf[0],"%d %d %d %d",&numCorners,&f[0],&f[1],&f[2]);
        if (numCorners != 3) {
            printf("PLY reader only supports triangles!\n"); fflush(stdout);
            error = 1;
            break;
        }
    }
    fclose(f);

    if (error) {
        delete[] pointData;
        delete[] faceData;
        return 0;
    }
    *vertexOut = pointData;
    *faceOut   = faceData;
    //printf("x:%d y:%d z:%d nx:%d ny:%d nz:%d\n",xyzflag[0],xyzflag[1],xyzflag[2],normalflag[0],normalflag[1],normalflag[2]);
    return 1;
}
