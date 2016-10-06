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

#include <iostream>
#include <Eigen/Geometry>
#include <vector>
#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <calib/calib.h>
#include <multicore/multicore.h>
//#include <5point-wrapper/RobustFivePoint.h>

const unsigned int maxNeighborAmount = 256;
const unsigned int maxLayers = 5;

typedef struct {
    int id;
    int frame; // frame where the landmark was extracted from
    Eigen::Vector3f p;
} LANDMARK;

// tri class for describing half-quads when estimating z for each feature point
class Tri2D {
public:
    Tri2D(int i0, int i1, int i2) {
        index[0] = i0;
        index[1] = i1;
        index[2] = i2;
    }
    int index[3];
};


// active pose determines which pose matrix is used: bundle adjusted or the original
extern int activePose;

enum FeatureCostMetric { HAMMING, EUCLIDIAN };
enum Feature3DPointStatus {FEATURE3D_POINT_INVALID=-2, FEATURE3D_POINT_UNASSIGNED=-1};

class Keyframe {    
private:
    void resetVariables();
public:
    unsigned int width,height,imageSize;
    FeatureCostMetric costMetric;
    cv::Mat depthMap[maxLayers],tmpDepthMap[maxLayers],depthTextureHost[maxLayers];
    cv::Mat data1C[maxLayers];
    cv::Mat rgbImage,rgbImagePacked;
    ProjectData *pointGrid;
    float *vertexBufferHost,*normalBufferHost,*tangentBufferHost,*uvBufferHost;
    int   *colorBufferHost;
    int *indexBufferHost; int numElements;
    float extents[3];
    float center[3];
    Eigen::MatrixXf features;
    unsigned int rgbTex;
    unsigned int grayTex[maxLayers];
    unsigned int depthTex[maxLayers];    
    unsigned int vbo,ibo;

    // keyframes are allowed to have pose alternatives, where active pose is the one used:
 //   Eigen::Matrix4f pose[2];
    Eigen::Matrix4f pose[2];
    bool poseInitialized;
    Eigen::Matrix3f intrinsic;
    Eigen::Matrix3f intrinsicDepth;

    Eigen::Matrix4f baseline;

    float distcoeff[5];
    Keyframe *neighbors[maxNeighborAmount];
    unsigned int numNeighbors;
    // distorted and undistorted measurement points
    Eigen::MatrixXf measurements;
    Eigen::MatrixXf featurePoints3D;
    Eigen::MatrixXf statusPoint3D;
    int featureLength;
    bool visible;
    int id;

    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> modelCorners;
    std::vector<cv::Point3f> pattern;
    cv::Size pattern_size;
    float csize;

    Keyframe(char *path, int id, Calibration &calib, Eigen::Matrix4f &depthPose, Eigen::Matrix4f &baseline, cv::Ptr<cv::DescriptorExtractor> &extractor, FeatureCostMetric metric, bool *useLayer, float depthScale=1.0f);
    ~Keyframe();
	void release();	
    void generateTexture(cv::Mat &colorMap);
    void generateDepthTextures();
    bool loadPose(char *posefile);
    bool loadCalib(char *calibfile);
    bool loadMeasurements(char *measurementsfile);
    void getFov(float *fovX, float *fovY);
    void generatePyramid1C();
    void addNeighbor(Keyframe *kf);
    bool isNeighbor(Keyframe *kf);
    Eigen::Matrix4f poseGL(int slot);
    Eigen::Matrix3f intrinsicGL();
    void poseDistance(Eigen::Matrix4f &poseGL, float *transDist, float *rotDist);
    void updateDepthMap(float *costVolume, int numDepthSamples, int layer);
    Eigen::Matrix3f scaleIntrinsic(unsigned int layer); 
    Eigen::Matrix3f scaleIntrinsicDepth(unsigned int layer);
    static void loadLandmarks(char *path, std::vector<LANDMARK> &landmarks);
    void distort(Eigen::Vector3f &p, Eigen::Vector3f &r);
    void undistortPixels(Eigen::Vector3f &p, Eigen::Vector3f &r);
    void undistort(Eigen::Vector3f &p, Eigen::Vector3f &r);
    void extractFeatures(cv::Ptr<cv::DescriptorExtractor> extractor,bool *useLayer);
    void generate3DFeatures();
    bool hitFound(float px, float py, Tri2D &tri, ProjectData *grid, Eigen::Vector4f &hitPoint);
    bool testTri(Tri2D &tri, ProjectData *grid, float minZ);
    bool testQuad(int *index, ProjectData *grid, float minZ);
    void genAABB(Tri2D &tri, ProjectData *gridData, Eigen::Vector2f &boundsMin, Eigen::Vector2f &boundsMax);
    void storeTriangleToBuckets(Tri2D &tri,ProjectData *gridData,std::map<int,std::vector<Tri2D> > &buckets, float xBuckets, float yBuckets, float xBucketSize, float yBucketSize);
    void promoteFeaturesToMeasurements();
    void setupCalib(Calibration &calib);
    void updateColorMap(cv::Mat &colorMap, int rIndex, int gIndex, int bIndex, bool findCheckerBoard=false, bool flipBoard=false);
    void updateDisparityMap(cv::Mat &colorMap,cv::Mat &disparityMap,Calibration &calib);
    bool disparityToDepth(cv::Mat &disparityMap, Calibration &calib, cv::Mat &depthMap);
    void processDepthmaps(cv::Mat &colorMap, Calibration &calib);
    void getDepthPattern(std::vector<cv::Point3f> &camPattern);
    float *getDepthData();
    void estimateRelativePose(std::vector<cv::Point3f> &camPattern);
    void allocateVbo();
    void releaseVbo();
    void updateVbo();
    void estimatePairwisePose(std::vector<cv::Point3f> &camPattern);
    void writeMesh(const char *fn);
    void resetPose();
    void generateNormals(float minDist);
    void filterZMap(cv::Mat &depthMapSrc,cv::Mat &depthMapDst, float zThreshold);
    void correctZMap(cv::Mat &depthMapSrc, cv::Mat &depthMapDst, int numPolyCoeffs, float *polyCoeffs);
    void allocateMaps(int width, int height);
};

// extracts matching points into modular format
void enumerateCorrespondencies(Keyframe *kA, Keyframe *kB, std::vector<Eigen::Vector2i> &matchingIndex);
// triangulate feature points
void triangulateFeatures(Eigen::MatrixXf &fA, Eigen::Matrix4f &poseA, Eigen::MatrixXf &fB, Eigen::Matrix4f &poseB, std::vector<Eigen::Vector2i> &matchingIndex, std::vector<Eigen::Vector3f> &points3d);
//void triangulateFeatures(Keyframe *kA, Keyframe *kB, std::vector<Eigen::Vector2i> &matchingIndex, std::vector<Eigen::Vector3f> &points3d);
