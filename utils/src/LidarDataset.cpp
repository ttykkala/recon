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

#include <liblas/liblas.hpp>
#include <fstream>  // std::ifstream
#include <iostream> // std::cout
#include <stdlib.h>
#include <LidarDataset.h>
#include <GL/glew.h>
#include <basic_math.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "eigenMath.h"
using namespace Eigen;
using namespace cv;
using namespace std;

void LidarDataset::resetVariables() {
    vertexBufferHost    = NULL;
    projectedPointsHost = NULL;
    indexBufferHost     = NULL;
    numElements         = 0;
    projectedCount      = 0;
    pointCount          = 0;
    id                  = -1;
    vbo                 = -1;
    ibo                 = -1;
    dataScale           = 1.0f;
    visible             = true;
    poseInitialized     = false;
    resetPose();
    center[0]  = 0; center[1]  = 0; center[2]  = 0;
    extents[0] = 0; extents[1] = 0; extents[2] = 0;
}

void genCacheFilename(const char *inputFilename, char *outputFilename, int dstlen) {
    strncpy(outputFilename,inputFilename,dstlen);
    char *dot = strrchr(outputFilename,'.');
    int offset = (dot-outputFilename)+1;
    outputFilename[offset+0] = 'u';
    outputFilename[offset+1] = 'c';
    outputFilename[offset+2] = 'p';
}

int LidarDataset::loadCacheFile(const char *filename) {
    FILE *f = fopen(filename,"rb");
    if (f == NULL) return 0;
    fread(&this->pointCount,sizeof(int),1,f);

    vertexBufferHost    = new float[pointCount*6];
    projectedPointsHost = new float[pointCount*6]; // (pixelX,pixelY,depth in original scale)
    indexBufferHost     = NULL;//new int[(width-1)*(height-1)*4];
    allocateVbo();

    fread(&this->vertexBufferHost[0],sizeof(float),pointCount*6,f);

    extents[0] = 0; extents[1] = 0; extents[2] = 0;
    center[0]  = 0; center[1]  = 0; center[2]  = 0;
    for (int off = 0; off < pointCount; off++) {
        float x = vertexBufferHost[off*6+0];
        float y = vertexBufferHost[off*6+1];
        float z = vertexBufferHost[off*6+2];
        center[0] += x;
        center[1] += y;
        center[2] += z;
    }
    center[0] /= float(pointCount);
    center[1] /= float(pointCount);
    center[2] /= float(pointCount);

    for (int off = 0; off < pointCount; off++) {
        float x = vertexBufferHost[off*6+0];
        float y = vertexBufferHost[off*6+1];
        float z = vertexBufferHost[off*6+2];
        float xdiff = fabs(x-center[0]);
        float ydiff = fabs(y-center[1]);
        float zdiff = fabs(z-center[2]);
        if (xdiff > extents[0]) extents[0] = xdiff;
        if (ydiff > extents[1]) extents[1] = ydiff;
        if (zdiff > extents[2]) extents[2] = zdiff;
    }
    printf("center: %f %f %f, extents: %f %f %f\n",center[0],center[1],center[2],extents[0],extents[1],extents[2]);
    updateVbo();
    fclose(f);
    printf("point count: %d\n",pointCount);
    return 1;
}

void LidarDataset::saveCacheFile(const char *filename) {
    FILE *f = fopen(filename,"wb");
    if (f == NULL) return;
    fwrite(&this->pointCount,sizeof(int),1,f);
    fwrite(&vertexBufferHost[0],sizeof(float),this->pointCount*6,f);
    fclose(f);
}


LidarDataset::LidarDataset(char *path, int id, float scale) {
    resetVariables();
    this->id = id;
    this->dataScale = scale;

    char buf[512];
    genCacheFilename(path,buf,512);
    printf("cache filename: %s\n",buf);

    if (loadCacheFile(buf)) return;

    std::ifstream ifs;
    ifs.open(path, std::ios::in | std::ios::binary);
    liblas::ReaderFactory f;
    liblas::Reader reader = f.CreateWithStream(ifs);
    liblas::Header const& header = reader.GetHeader();

    this->pointCount = header.GetPointRecordsCount();

    vertexBufferHost    = new float[pointCount*6];
    projectedPointsHost = new float[pointCount*6]; // (pixelX,pixelY,depth in original scale)
    indexBufferHost     = NULL;//new int[(width-1)*(height-1)*4];
    allocateVbo();

    printf("Compressed: %s\n",(header.Compressed() == true) ? "true":"false");
    std::cout << "Signature: " << header.GetFileSignature() << '\n';
    std::cout << "Points count: " << this->pointCount << '\n';

    extents[0] = 0; extents[1] = 0; extents[2] = 0;
    center[0]  = 0; center[1]  = 0; center[2]  = 0;
    int off = 0;

    Eigen::Matrix4f R;
    R(0,0) = 1; R(0,1) =       0; R(0,2) =      0; R(0,3) =  0;
    R(1,0) = 0; R(1,1) =       0; R(1,2) =      1; R(1,3) =  0;
    R(2,0) = 0; R(2,1) =      -1; R(2,2) =      0; R(2,3) =  0;
    R(3,0) = 0; R(3,1) =       0; R(3,2) =      0; R(3,3) =  1;

    while (reader.ReadNextPoint())
    {
        liblas::Point const& p = reader.GetPoint();
        liblas::Color const& c = p.GetColor();
        Eigen::Vector4f v(p.GetX(),p.GetY(),p.GetZ(),1);
        Eigen::Vector4f w = dataScale*R*v;
        float x =   w(0);
        float y =   w(1);
        float z =   w(2);
        vertexBufferHost[off*6+0] = x;
        vertexBufferHost[off*6+1] = y;
        vertexBufferHost[off*6+2] = z;
        vertexBufferHost[off*6+3] = float(c.GetRed())/65535.0f;
        vertexBufferHost[off*6+4] = float(c.GetGreen())/65535.0f;
        vertexBufferHost[off*6+5] = float(c.GetBlue())/65535.0f;
        center[0] += x;
        center[1] += y;
        center[2] += z;
        off++;
    }
    center[0] /= float(pointCount);
    center[1] /= float(pointCount);
    center[2] /= float(pointCount);
    off = 0;
    for (int i = 0; i < pointCount; i++) {
        float x = vertexBufferHost[off*6+0];
        float y = vertexBufferHost[off*6+1];
        float z = vertexBufferHost[off*6+2];
        float xdiff = fabs(x-center[0]);
        float ydiff = fabs(y-center[1]);
        float zdiff = fabs(z-center[2]);
        if (xdiff > extents[0]) extents[0] = xdiff;
        if (ydiff > extents[1]) extents[1] = ydiff;
        if (zdiff > extents[2]) extents[2] = zdiff;
        off++;
    }

    printf("center: %f %f %f, extents: %f %f %f\n",center[0],center[1],center[2],extents[0],extents[1],extents[2]);
    updateVbo();
    saveCacheFile(buf);
}

Eigen::Matrix4f LidarDataset::poseGL() {
    Eigen::Matrix4f bose2 = pose;
    /* flip Y and Z for OpenGL visualizations
    C1 <- C2 <- W
    [1 0  0] [x1 x2 x3 t1] [X] = [ x1  x2  x3  t1][X]
    [0 -1 0] [y1 y2 y3 t2] [Y]   [-y1 -y2 -y3 -t2][Y]
    [0 0 -1] [z1 z2 z3 t3] [Z]   [-z1 -z2 -z3 -t3][Z]
                           [1]                    [1]
    */
	bose2.row(1) = -bose2.row(1);
	bose2.row(2) = -bose2.row(2);
	return bose2;
}

LidarDataset::~LidarDataset() {
    release();
}

void LidarDataset::release() {
    if (vertexBufferHost) delete[] vertexBufferHost; vertexBufferHost = NULL;
    if (projectedPointsHost) delete[] projectedPointsHost; projectedPointsHost = NULL;
    if (indexBufferHost) delete[] indexBufferHost; indexBufferHost = NULL;
    releaseVbo();
}

void LidarDataset::releaseVbo() {
    if (vbo != -1) {
        glDeleteBuffers(1, &vbo);
        vbo = -1;
    }
    if (ibo != -1) {
        glDeleteBuffers(1, &ibo);
        ibo = -1;
    }
}

void LidarDataset::resetPose() {
    pose = Eigen::Matrix4f::Identity();
}

void LidarDataset::updateVbo() {
    if (pointCount <= 0) return;
    unsigned int size = pointCount*6;
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size * sizeof(float), vertexBufferHost);
}

void LidarDataset::allocateVbo() {
    glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, pointCount * sizeof(float) * 6, NULL, GL_STREAM_COPY);
    if (vbo == 0) {
        printf("vbo allocation failed!\n");
        fflush(stdin); fflush(stdout); fflush(stderr);
    }
    assert(vbo != 0);
}

void LidarDataset::savePLY(const char *filename, int numberOfTris, int *triIndex, int numberOfVerts, float *vertices) {
    printf("numberOfTris: %d\n",numberOfTris);

    FILE *f = fopen(filename,"wb");
    fprintf(f,"ply\n");
    fprintf(f,"format ascii 1.0\n");
    fprintf(f,"element vertex %d\n",numberOfVerts);
    fprintf(f,"property float x\n");
    fprintf(f,"property float y\n");
    fprintf(f,"property float z\n");
    fprintf(f,"element face %d\n",numberOfTris);
    fprintf(f,"property list uchar int vertex_index\n");
    fprintf(f,"end_header\n");
    for (int i = 0; i < numberOfVerts; i++) {
        fprintf(f,"%f %f %f\n",vertices[i*3+0],vertices[i*3+1],-vertices[i*3+2]*100.0f);
    }
    for (int i = 0; i < numberOfTris; i++) {
        fprintf(f,"3 %d %d %d\n",triIndex[i*3+0],triIndex[i*3+1],triIndex[i*3+2]);
    }
    fclose(f);
}

// tri class for estimating z for each zmap grid point
class Tri2D {
public:
    Tri2D(int i0, int i1, int i2) {
        index[0] = i0;
        index[1] = i1;
        index[2] = i2;
    }
    int index[3];
};

void genAABB(Tri2D &tri, float *gridPoints3, Eigen::Vector2f &boundsMin, Eigen::Vector2f &boundsMax) {
    // initialize bounds to a single point:
    boundsMin(0) = gridPoints3[tri.index[0]*3+0]; boundsMax(0) = boundsMin(0);
    boundsMin(1) = gridPoints3[tri.index[0]*3+1]; boundsMax(1) = boundsMin(1);
    // extend bounds using the rest 2 points:
    for (int i = 1; i <= 2; i++) {
        float x = gridPoints3[tri.index[i]*3+0];
        float y = gridPoints3[tri.index[i]*3+1];
        if (x < boundsMin(0)) boundsMin(0) = x;
        if (y < boundsMin(1)) boundsMin(1) = y;
        if (x > boundsMax(0)) boundsMax(0) = x;
        if (y > boundsMax(1)) boundsMax(1) = y;
    }
}

void addTriToBuckets(Tri2D &tri,float *vertices3,std::map<int,std::vector<Tri2D> > &buckets, float xBuckets, float yBuckets, float xBucketSize, float yBucketSize)
{
    Eigen::Vector2f boundsMin,boundsMax;
    genAABB(tri,vertices3,boundsMin,boundsMax);

    int bucketWidth  = int(xBuckets);
    int bucketHeight = int(yBuckets);

    // convert pixel coordinates into bucket coordinates:
    int border=0;
    int bucketX0 = int(boundsMin(0) / xBucketSize)-border; if (bucketX0 < 0) bucketX0 = 0;
    int bucketY0 = int(boundsMin(1) / yBucketSize)-border; if (bucketY0 < 0) bucketY0 = 0;
    int bucketX1 = int(boundsMax(0) / xBucketSize)+border; if (bucketX1 > bucketWidth-1)  bucketX1 = bucketWidth-1;
    int bucketY1 = int(boundsMax(1) / yBucketSize)+border; if (bucketY1 > bucketHeight-1) bucketY1 = bucketHeight-1;

    // fill overlapping buckets:
    for (int by = bucketY0; by <= bucketY1; by++) {
        for (int bx = bucketX0; bx <= bucketX1; bx++) {
            int bOffset = bx+by*bucketWidth;
            buckets[bOffset].push_back(tri);
        }
    }
}

bool testTri(Tri2D &tri, float *vertices3, float minZ, float zDeviationMax) {
    float z0 = fabs(vertices3[tri.index[0]*3+0]);
    float z1 = fabs(vertices3[tri.index[1]*3+1]);
    float z2 = fabs(vertices3[tri.index[2]*3+2]);
    if (z0 < minZ) return false;
    if (z1 < minZ) return false;
    if (z2 < minZ) return false;
  /*
    float dev0 = fabs(z0-z1);
    float dev1 = fabs(z1-z2);
    float dev2 = fabs(z0-z2);
    float dev = MAX(MAX(dev0,dev1),dev2);
    if (dev > zDeviationMax) return false;
    */
    return true;
}

void storeTrianglesToBuckets(int numberOfTris, int *triIndex, float *vertices3, int width, int height, float xBucketSize, float yBucketSize, std::map<int,std::vector<Tri2D> > &buckets) {
    float xBuckets = width /xBucketSize;
    float yBuckets = height/yBucketSize;
    for (int by = 0; by < yBuckets; by++) {
        for (int bx = 0; bx < xBuckets; bx++) {
            std::vector<Tri2D> &bucket = buckets[bx+by*int(xBuckets)]; bucket.reserve(100);
        }
    }
    //cv::Mat testImg(height,width,CV_8UC3);
    //unsigned char *ptr = testImg.ptr(); memset(ptr,0,width*height*3);

//    float minAngle = deg2rad(1.0f);
    float minAngle = 400.0f;
    for (int tri = 0; tri < numberOfTris; tri++) {
        // store triangle to buckets:
        Tri2D triA(triIndex[tri*3+0], triIndex[tri*3+1], triIndex[tri*3+2]);
        if (testTri(triA,vertices3,1e-7f,minAngle)) {
            addTriToBuckets(triA,vertices3,buckets,xBuckets,yBuckets,xBucketSize,yBucketSize);
        }
    }
}

bool hitFound(float px, float py, Tri2D &tri, float *vertices3, Eigen::Vector4f &hitPoint) {
    float *pA = &vertices3[tri.index[0]*3+0];
    float *pB = &vertices3[tri.index[1]*3+0];
    float *pC = &vertices3[tri.index[2]*3+0];

    Eigen::Vector2d A(pA[0],pA[1]);
    Eigen::Vector2d B(pB[0],pB[1]);
    Eigen::Vector2d C(pC[0],pC[1]);
    Eigen::Vector2d P(px,py);

    Eigen::Vector2d v0 = C - A;
    Eigen::Vector2d v1 = B - A;
    Eigen::Vector2d v2 = P - A;

    // Compute dot products
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

    // Compute barycentric coordinates
    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    bool triHit = (u >= 0) && (v >= 0) && (u+v<1);
    if (triHit) {
        // inverse depths are linear in pixel coordinates:
        double izA = 1.0f / pA[2];
        double izB = 1.0f / pB[2];
        double izC = 1.0f / pC[2];
        hitPoint(0) = float(pA[0] + u*(pC[0] - pA[0]) + v*(pB[0] - pA[0]));
        hitPoint(1) = float(pA[1] + u*(pC[1] - pA[1]) + v*(pB[1] - pA[1]));
        hitPoint(2) = float(1.0 / (izA + u*(izC - izA) + v*(izB - izA)));
        hitPoint(3) = 1;
        return true;
    }
    // triangle not hit:
    return false;
}

void fillDepthmapHoles(cv::Mat &depthMap, std::map<int,std::vector<Tri2D> > &buckets, int xBucketSize, int yBucketSize, float *vertices3) {
    float xBuckets = depthMap.cols/xBucketSize;
//    float yBuckets = depthMap.rows/yBucketSize;
    float *zmap = (float*)depthMap.ptr();
    int offset = 0;
    for (int j = 0; j < depthMap.rows; j++) {
        for (int i = 0; i < depthMap.cols; i++,offset++) {
            if (zmap[offset] < 1e-5f) {
                bool hit = false;
                int bx = int(i/xBucketSize);
                int by = int(j/yBucketSize);
                Eigen::Vector4f hitPoint;
                Eigen::Vector4f nearestHitPoint(0,0,FLT_MAX,1);
                std::vector<Tri2D> &bucket = buckets[bx+by*int(xBuckets)];
                for (size_t bi = 0; bi < bucket.size(); bi++) {
                    Tri2D &tri = bucket[bi];
                    if (hitFound(i,j,tri,vertices3,hitPoint)) {
                        if (fabs(hitPoint(2)) < fabs(nearestHitPoint(2))) {
                            nearestHitPoint = hitPoint;
                            hit = true;
                        }
                    }
                }
                if (hit) zmap[offset] = nearestHitPoint(2);
                else zmap[offset] = 0.0f;
            }
        }
    }
}

void LidarDataset::triangulateMap(cv::Mat &depthMapInput, const char *ptsfile, const char *trifile) {
    int width   = depthMapInput.cols;
    int height  = depthMapInput.rows;
    float *zmap = (float*)depthMapInput.ptr();

    // count the number of valid depths:
    int zcnt = 4;
    for (int i = 0; i < width*height; i++) { if (zmap[i] > 0.0f)  { zcnt++; } }

    // store semi-dense depth maps to ascii file:
    FILE *f = fopen(ptsfile,"wb");
    fprintf(f,"%d \n",zcnt);
    fprintf(f,"2 \n");
    fprintf(f,"%lf %lf \n",0.0,0.0);
    fprintf(f,"%lf %lf \n",double(width-1),0.0);
    fprintf(f,"%lf %lf \n",double(width-1),double(height-1));
    fprintf(f,"%lf %lf \n",0.0,double(height-1));

    float *vertices = new float[zcnt*3];
    int vindex = 0;
    // note: triangles with borderline rectangle will be contaminated with invalid/unknown depths!
    vertices[vindex*3+0] = 0.0f;            vertices[vindex*3+1] = 0.0f;            vertices[vindex*3+2] = 0.0f; vindex++;
    vertices[vindex*3+0] = float(width-1);  vertices[vindex*3+1] = 0.0f;            vertices[vindex*3+2] = 0.0f; vindex++;
    vertices[vindex*3+0] = float(width-1);  vertices[vindex*3+1] = float(height-1); vertices[vindex*3+2] = 0.0f; vindex++;
    vertices[vindex*3+0] = 0.0f;            vertices[vindex*3+1] = float(height-1); vertices[vindex*3+2] = 0.0f; vindex++;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = x+y*width;
            if (zmap[i] > 0.0f) {
                fprintf(f,"%lf %lf \n",double(x),double(y));
                vertices[vindex*3+0] = float(x);
                vertices[vindex*3+1] = float(y);
                vertices[vindex*3+2] = zmap[i];
                vindex++;
            }

        }
    }
    fclose(f);

    // execute delaunay triangulation:
    char buf[512];
    sprintf(buf,"cat %s | qdelaunay Qt s i > %s",ptsfile,trifile);
    system(buf);

    // read triangle indexing:
    f = fopen(trifile,"rb");
    int numberOfTris = 0;
    fscanf(f,"%d\n",&numberOfTris);
    int *triIndex = new int[numberOfTris*3];
    for (int i = 0; i < numberOfTris; i++) {
        int i0=0, i1=0, i2=0;
        fscanf(f,"%d %d %d\n",&i0,&i1,&i2);
        triIndex[i*3+0] = i0;
        triIndex[i*3+1] = i1;
        triIndex[i*3+2] = i2;
    }
    fclose(f);

    std::map<int,std::vector<Tri2D> > buckets;
    storeTrianglesToBuckets(numberOfTris,triIndex,vertices,width,height,4,4,buckets);
    fillDepthmapHoles(depthMapInput,buckets,4,4,vertices);

//    char *plyfile = "test.ply";
//    savePLY(plyfile,numberOfTris,triIndex,zcnt,vertices);

    delete[] triIndex;
    delete[] vertices;
}

void smartMinimumFilter(cv::Mat &src, cv::Mat &dst, int kernelSize, float zThreshold) {
    int ksize = kernelSize/2;
    int w = src.cols;
    int h = src.rows;
    float *srcZ = (float*)src.ptr();
    float *dstZ = (float*)dst.ptr();
    int dstOff = 0;
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++,dstOff++) {
            float currentZ = fabs(srcZ[dstOff]);
            if (currentZ == 0.0f) continue; // skip holes, delaunay will take care of this later on
            float minZ = FLT_MAX, maxZ = 0;
            int zSampleCount=0;
            for (int v = -ksize; v <= ksize; v++) {
                for (int u = -ksize; u <= ksize; u++) {
                    int x = i+u;
                    int y = j+v;
                    if (x < 0) x = 0; if (x >= w) x = w-1;
                    if (y < 0) y = 0; if (y >= h) y = h-1;
                    float z = fabs(srcZ[x+y*w]);
                    if (z > 0.0f) {
                        if (z < minZ) minZ = z;
                        if (z > maxZ) maxZ = z;
                        zSampleCount++;
                    }
                }
            }
            if (((currentZ - minZ) > zThreshold) && zSampleCount > 0) {
                dstZ[dstOff] = minZ;
            } else {
                dstZ[dstOff] = currentZ;
            }
        }
    }
}

void LidarDataset::updateDepthmap(int width, int height, cv::Mat &colorMap, cv::Mat &depthMapFiltered)  {
    // allocate and resize depth map
    if (depthMap.cols != width || depthMap.rows != height) {
        if (depthMap.cols*depthMap.rows > 0) depthMap.release();
        depthMap = cv::Mat(height,width,CV_32FC1);
    }
    //imwrite("test.ppm",colorMap);
    // reset depth map
    depthMap = 0.0f;

    //cv::Mat testColorMap = cv::Mat(depthMap.rows,depthMap.cols,CV_8UC3); memset(testColorMap.ptr(),0,depthMap.rows*depthMap.cols*3);
    //testColorMap = 0;
    // pick nearest z's for semi-dense depthmap
    float *zmap = (float*)depthMap.ptr();
    unsigned char *cmap = (unsigned char*)colorMap.ptr();
    //unsigned char *tmap = (unsigned char*)testColorMap.ptr();
    for (int i = 0; i < projectedCount; i++) {
        int xi  = int(projectedPointsHost[i*6+0]);
        int yi  = int(projectedPointsHost[i*6+1]);
        float z = projectedPointsHost[i*6+2];
        int off = xi+yi*width;
        float diffR = fabs(float(cmap[off*3+0])/255.0f - projectedPointsHost[i*6+5]);
        float diffG = fabs(float(cmap[off*3+1])/255.0f - projectedPointsHost[i*6+4]);
        float diffB = fabs(float(cmap[off*3+2])/255.0f - projectedPointsHost[i*6+3]);
        float colorDist = MAX(MAX(diffR,diffG),diffB);

        float currentDepth = zmap[off];
        if (z > 0.0f && (currentDepth == 0.0f || z < currentDepth) && colorDist < 0.5f) {
            zmap[off] = z;
        }
    }
    //imwrite("test2.ppm",testColorMap);

    depthMapFiltered = 0.0f;
    zmap = (float*)depthMapFiltered.ptr();
    smartMinimumFilter(depthMap,depthMapFiltered,5,0.05f);
//     cv::Mat strElem = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
//     cv::erode(depthMap,depthMapFiltered,strElem);

    char ptsfile[512]; memset(ptsfile,0,512); sprintf(ptsfile,"pts.ascii");
    char trifile[512]; memset(trifile,0,512); sprintf(trifile,"tri.ascii");
    triangulateMap(depthMapFiltered,ptsfile,trifile);
}

void LidarDataset::calcDepthMap(const Eigen::Matrix4f &pose, Calibration &calib, int width, int height, float r, float g, float b, cv::Mat &colorMap, cv::Mat &depthMap)
{
    float fovXDeg  = calib.getFovX_R();
    float fovYDeg  = calib.getFovY_R();
    float maxDepth = calib.getMaxDist();

    Eigen::Matrix3f intrinsic,intrinsicDepth;
    float kcR[8];
    setupIntrinsics(calib,width,width,intrinsic,intrinsicDepth,&kcR[0]);
    // NOTE: distortion is currently disabled!
    for (int i = 0; i < 8; i++) kcR[i] = 0.0f;

    float dirX = tan(3.141592653f*fovXDeg/360.0f);
    float dirY = tan(3.141592653f*fovYDeg/360.0f);

    float minX=width,minY=width,maxX=0,maxY=0;
    projectedCount = 0;
    bool updateFlag = false;
    int off = 0;
    for (int i = 0; i < pointCount; i++) {
        float x = vertexBufferHost[off*6+0];
        float y = vertexBufferHost[off*6+1];
        float z = vertexBufferHost[off*6+2];
        Eigen::Vector4f v(x,y,z,1);
        Eigen::Vector4f w = pose*v;
        if (w(2) < 0 && w(2) >= -maxDepth) {
            float limitX = fabs(dirX*w(2));
            float limitY = fabs(dirY*w(2));
            if (fabs(w(0)) < limitX && fabs(w(1)) < limitY) {
                Eigen::Vector2f pu(-w(0)/w(2),w(1)/w(2));
                Eigen::Vector2f pd;
                radialDistort(pu,kcR,intrinsic,pd);
                //vertexBufferHost[off*6+3] = r;
                //vertexBufferHost[off*6+4] = g;
                //vertexBufferHost[off*6+5] = b;
                if (pd(0) >= 0 && pd(1) >= 0 && pd(0) < width && pd(1) < height) {
                    if (pd(0) < minX) minX = pd(0);
                    if (pd(1) < minY) minY = pd(1);
                    if (pd(0) > maxX) maxX = pd(0);
                    if (pd(1) > maxY) maxY = pd(1);
                    projectedPointsHost[projectedCount*6+0] = pd(0);
                    projectedPointsHost[projectedCount*6+1] = pd(1);
                    projectedPointsHost[projectedCount*6+2] = fabs(w(2))/dataScale;
                    projectedPointsHost[projectedCount*6+3] = vertexBufferHost[off*6+3];
                    projectedPointsHost[projectedCount*6+4] = vertexBufferHost[off*6+4];
                    projectedPointsHost[projectedCount*6+5] = vertexBufferHost[off*6+5];

                    projectedCount++;
                }
                updateFlag = true;
            }
        }
        off++;
    }
    printf("pixel extents: %f %f %f %f\n",minX,minY,maxX,maxY);

    updateDepthmap(width,height,colorMap,depthMap);
  //  updateVbo();
}
