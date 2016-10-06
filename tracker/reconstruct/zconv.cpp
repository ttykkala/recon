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


#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include  <calib/calib.h>
#include <tracker/basic_math.h>
#include <multicore/multicore.h>
#include <reconstruct/zconv.h>
#include <map>
#include <Eigen/Geometry>
#include <tracker/eigenMath.h>

using namespace cv;

// tri class for describing half-quads when estimating z for each feature point
class Triangle2D {
public:
    Triangle2D(int i0, int i1, int i2) {
        index[0] = i0;
        index[1] = i1;
        index[2] = i2;
    }
    int index[3];
};

ZConv::ZConv() {

}

ZConv::~ZConv() {

}

void ZConv::mapDisparityRange(unsigned short* ptr, int w, int h, int minD2,int maxD2) {
    int len = w*h;
    unsigned short minD = 65535;
    unsigned short maxD = 0;
    int cnt = 0;
    for (int i = 0; i < len; i++) {
        if (ptr[i] > 0) {
            ptr[i] = 65535-ptr[i];
            if (ptr[i] < minD) minD = ptr[i];
            if (ptr[i] > maxD) maxD = ptr[i];
        }
    }

/*    unsigned short trans = minD2-minD;
    float scale = float(maxD2-minD2+1)/float(maxD-minD+1);
    for (int i = 0; i < len; i++) {
        if (ptr[i]>0)
            ptr[i] = (unsigned short)(float(ptr[i]+trans)*scale);
    }*/
}

void ZConv::dumpDepthRange(float *depthMap, int width, int height) {
    float *zptr = depthMap;
    float minZ = FLT_MAX;
    float maxZ = FLT_MIN;
    int len = width*height;
    int cnt = 0;
    float *tmp = new float[len];
    for (int i = 0; i < len; i++) {
        if (zptr[i] < minZ) minZ = zptr[i];
        if (zptr[i] > maxZ) maxZ = zptr[i];
        if (zptr[i] != 0.0f) { tmp[cnt] = zptr[i] ; cnt++; }
    }
    float medianZ = quickMedian(tmp,cnt);
    printf("minz: %f maxz:%f median:%f cnt: %d\n",minZ,maxZ,medianZ,cnt); fflush(stdin); fflush(stdout);
    delete[] tmp;
}

void ZConv::baselineTransform(float *zptrSrc, float *zptrDst, int width, int height, Calibration *calib)
{
    memset(zptrDst,0,sizeof(float)*width*height);

    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float *KL = &calib->getCalibData()[KL_OFFSET];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, zptrSrc);
    Mat depthMapR(height, width, CV_32FC1, zptrDst);

    multicore->baselineTransform(depthMapL,depthMapR,KL, TLR, KR);

    calib->setupCalibDataBuffer(prevW,prevH);
}


void ZConv::baselineWarp(float *depthImageL,unsigned char *grayDataR, ProjectData *fullPointSet, int width, int height, Calibration *calib) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float *KL = &calib->getCalibData()[KL_OFFSET];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];
    float *kc = &calib->getCalibData()[KcR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, depthImageL);
    Mat grayImageR(height, width, CV_8UC1, grayDataR);
    multicore->baselineWarp(depthMapL,grayImageR,KL, TLR, KR, kc, fullPointSet);
    calib->setupCalibDataBuffer(prevW,prevH);
}

void ZConv::baselineWarpRGB(float *depthImageL,unsigned char *rgbDataR, ProjectData *fullPointSet, int width, int height, Calibration &calib) {

    //TODO: this routine does not yet show up in rendering?
    //what is the difference to the lower routine?
    Eigen::Matrix3f KL,KR;
    float distRGB[8];
    setupIntrinsics(calib,width,width,KR,KL,&distRGB[0]);
    Eigen::Matrix4f baselineLR;
    setupBaseline(calib,baselineLR);

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, depthImageL);
    Mat rgbImageR(height, width, CV_8UC3, rgbDataR);
    multicore->baselineWarpRGB(depthMapL,rgbImageR,KL, baselineLR, KR, &distRGB[0], fullPointSet);
/*
    cv::Mat rgbImg(height,width,CV_8UC3,rgbDataR);
    cv::Mat grayImageR(height,width,CV_8UC1);
    cv::cvtColor(rgbImg,grayImageR,CV_RGB2GRAY);
    int prevW = calib.getWidth(); int prevH = calib.getHeight();
    calib.setupCalibDataBuffer(width,height);

    float *KL  = &calib.getCalibData()[KL_OFFSET];
    float *KR  = &calib.getCalibData()[KR_OFFSET];
    float *TLR = &calib.getCalibData()[TLR_OFFSET];
    float *kc  = &calib.getCalibData()[KcR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, depthImageL);
   // Mat grayImageR(height, width, CV_8UC1, grayDataR);
    multicore->baselineWarp(depthMapL,grayImageR,KL, TLR, KR, kc, fullPointSet);
    calib.setupCalibDataBuffer(prevW,prevH);
    */
}


void ZConv::undistortDisparityMap(unsigned short* disp16, float *udisp, int width, int height, Calibration* /*calib*/) {

    // this method is currently deprecated.
    // TODO: use polyCoeffs for undistortion!
    for (int i = 0; i < width*height; i++) { udisp[i] = float(disp16[i]);}
    /*    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);
    float alpha0 = calib->getCalibData()[ALPHA0_OFFSET];
    float alpha1 = calib->getCalibData()[ALPHA1_OFFSET];
    float *beta = &calib->getCalibData()[BETA_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat dispImage(height, width, CV_16UC1, disp16);
    Mat uDispImage(height, width, CV_32FC1, udisp);
    multicore->undistortDisparityMap(dispImage,uDispImage, alpha0, alpha1, beta);
    calib->setupCalibDataBuffer(prevW,prevH);*/
}



void genAABB(Triangle2D &tri, ProjectData *gridData, Eigen::Vector2f &boundsMin, Eigen::Vector2f &boundsMax) {
    // initialize bounds to a single point:
    boundsMin(0) = gridData[tri.index[0]].rx2; boundsMax(0) = boundsMin(0);
    boundsMin(1) = gridData[tri.index[0]].ry2; boundsMax(1) = boundsMin(1);
    // extend bounds using the rest 2 points:
    for (int i = 1; i <= 2; i++) {
        float x = gridData[tri.index[i]].rx2;
        float y = gridData[tri.index[i]].ry2;
        if (x < boundsMin(0)) boundsMin(0) = x;
        if (y < boundsMin(1)) boundsMin(1) = y;
        if (x > boundsMax(0)) boundsMax(0) = x;
        if (y > boundsMax(1)) boundsMax(1) = y;
    }
}

void storeTriangleToBuckets(Triangle2D &tri,ProjectData *gridData,std::map<int,std::vector<Triangle2D> > &buckets, float xBuckets, float yBuckets, float xBucketSize, float yBucketSize)
{
    Eigen::Vector2f boundsMin,boundsMax;
    genAABB(tri,gridData,boundsMin,boundsMax);

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

bool testQuad(int *index, ProjectData *grid, float minZ) {
    for (int k = 0; k < 4; k++) {
        ProjectData &p = grid[index[k]];
        if (fabs(p.pz) < minZ) return false;
    }
    return true;
}

bool testTri(Triangle2D &tri, ProjectData *grid, float minZ) {
    for (int k = 0; k < 3; k++) {
        ProjectData &p = grid[tri.index[k]];
        if (fabs(p.pz) < minZ) return false;
    }
    return true;
}


bool hitFound(float px, float py, Triangle2D &tri, ProjectData *grid, Eigen::Vector4f &hitPoint) {
    ProjectData &pA = grid[tri.index[0]];
    ProjectData &pB = grid[tri.index[1]];
    ProjectData &pC = grid[tri.index[2]];

    Eigen::Vector2d A(pA.rx2,pA.ry2);
    Eigen::Vector2d B(pB.rx2,pB.ry2);
    Eigen::Vector2d C(pC.rx2,pC.ry2);
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
        double izA = 1.0f / pA.rz;
        double izB = 1.0f / pB.rz;
        double izC = 1.0f / pC.rz;
        hitPoint(0) = float(pA.rx + u*(pC.rx - pA.rx) + v*(pB.rx - pA.rx));
        hitPoint(1) = float(pA.ry + u*(pC.ry - pA.ry) + v*(pB.ry - pA.ry));
        hitPoint(2) = float(1.0 / (izA + u*(izC - izA) + v*(izB - izA)));
        hitPoint(3) = 1;
        return true;
    }
    // triangle not hit:
    return false;
}



void ZConv::mapFromRGBToDepth(ProjectData *pointGrid, Calibration &calib, std::vector<cv::Point2f> &rgbPoints, std::vector<cv::Point3f> &depthPoints3D, std::vector<cv::Point2f> &depthPoints2D, std::vector<int> &mask)
{
    // generate 3d feature points:
    // 1) warp each triangle T_i into 2d rgb imaging plane
    // 2) fetch related features using bucketing structure
    // 3) do triangle test for each related feature f_i
    // 3) if f_i is within T_i -> solve depth using baryocentric coordinates
    // 4)     warp 3d point in RGB view back to IR view -> store to an array
    //statusPoint3D    = Eigen::MatrixXf(3,rgbPoints.size());
    Eigen::Matrix3f KL,KR;
    float distRGB[8],distDepth[5]={0,0,0,0,0};
    int width = calib.getWidth();
    int height = calib.getHeight();
    setupIntrinsics(calib,width,width,KR,KL,&distRGB[0]);

    Eigen::Matrix4f baseline;
    setupBaseline(calib,baseline);


    float xBucketSize = 4;
    float yBucketSize = 4;
    float xBuckets = width /xBucketSize;
    float yBuckets = height/yBucketSize;
    std::map<int,std::vector<Triangle2D> > buckets;
    for (int by = 0; by < yBuckets; by++) {
        for (int bx = 0; bx < xBuckets; bx++) {
            std::vector<Triangle2D> &bucket = buckets[bx+by*int(xBuckets)];
            bucket.reserve(100);
        }
    }

//    cv::Mat testImg(height,width,CV_8UC3);
//    unsigned char *ptr = testImg.ptr(); memset(ptr,0,width*height*3);

    for (int j = 0; j < height-1; j++) {
        for (int i = 0; i < width-1; i++) {
            int index[4];
            int offset = i+j*width;
            index[0] = offset;
            index[1] = offset+1;
            index[2] = offset+width+1;
            index[3] = offset+width;
            // store triangles to buckets:
            Triangle2D triA(index[1], index[2], index[0]);
            // are there holes in depth map?
            if (testTri(triA,pointGrid,100.0f)) {
                storeTriangleToBuckets(triA,pointGrid,buckets,xBuckets,yBuckets,xBucketSize,yBucketSize);
                //cv::line(testImg,cv::Point(pointGrid[triA.index[0]].rx2,pointGrid[triA.index[0]].ry2),cv::Point(pointGrid[triA.index[1]].rx2,pointGrid[triA.index[1]].ry2),cv::Scalar(100,100,100),1);
                //cv::line(testImg,cv::Point(pointGrid[triA.index[1]].rx2,pointGrid[triA.index[1]].ry2),cv::Point(pointGrid[triA.index[2]].rx2,pointGrid[triA.index[2]].ry2),cv::Scalar(100,100,100),1);
                //cv::line(testImg,cv::Point(pointGrid[triA.index[2]].rx2,pointGrid[triA.index[2]].ry2),cv::Point(pointGrid[triA.index[0]].rx2,pointGrid[triA.index[0]].ry2),cv::Scalar(100,100,100),1);
            }
            Triangle2D triB(index[3], index[0], index[2]);
            // are there holes in depth map?
            if (testTri(triB,pointGrid,100.0f)) {
                storeTriangleToBuckets(triB,pointGrid,buckets,xBuckets,yBuckets,xBucketSize,yBucketSize);
                //cv::line(testImg,cv::Point(pointGrid[triB.index[0]].rx2,pointGrid[triB.index[0]].ry2),cv::Point(pointGrid[triB.index[1]].rx2,pointGrid[triB.index[1]].ry2),cv::Scalar(100,100,100),1);
                //cv::line(testImg,cv::Point(pointGrid[triB.index[1]].rx2,pointGrid[triB.index[1]].ry2),cv::Point(pointGrid[triB.index[2]].rx2,pointGrid[triB.index[2]].ry2),cv::Scalar(100,100,100),1);
                //cv::line(testImg,cv::Point(pointGrid[triB.index[2]].rx2,pointGrid[triB.index[2]].ry2),cv::Point(pointGrid[triB.index[0]].rx2,pointGrid[triB.index[0]].ry2),cv::Scalar(100,100,100),1);
            }
        }
    }
    depthPoints2D.reserve(rgbPoints.size()); depthPoints2D.clear();
    depthPoints3D.reserve(rgbPoints.size()); depthPoints3D.clear();
    mask.reserve(rgbPoints.size());          mask.clear();


    float hitCnt = 0;
    float totalCnt = 0;
    for (size_t fi = 0; fi < rgbPoints.size(); fi++) {
        float px = rgbPoints[fi].x;
        float py = rgbPoints[fi].y;
        int bx = int(px/xBucketSize);
        int by = int(py/yBucketSize);
        if (bx < 0 || by < 0) continue;
        if (bx >= xBuckets || by >= yBuckets) continue;

        //  cv::circle(testImg,cv::Point(px,py),3,cv::Scalar(0,0,255),1,-1);

        bool hit = false;
        Eigen::Vector4f hitPoint;
        Eigen::Vector4f nearestHitPoint(0,0,FLT_MAX,1);
        std::vector<Triangle2D> &bucket = buckets[bx+by*int(xBuckets)];
        for (size_t bi = 0; bi < bucket.size(); bi++) {
            Triangle2D &tri = bucket[bi];
            if (hitFound(px,py,tri,pointGrid,hitPoint)) {
                if (fabs(hitPoint(2)) < fabs(nearestHitPoint(2))) {
                    nearestHitPoint = hitPoint;
                    hit = true;
                    //float tx = pointGrid[tri.index[0]].rx2+pointGrid[tri.index[1]].rx2+pointGrid[tri.index[2]].rx2; tx /= 3;
                    //float ty = pointGrid[tri.index[0]].ry2+pointGrid[tri.index[1]].ry2+pointGrid[tri.index[2]].ry2; ty /= 3;
                    //cv::line(testImg,cv::Point(tx,ty+2),cv::Point(tx,ty-2),cv::Scalar(0,255,0));
                    //cv::line(testImg,cv::Point(tx-2,ty),cv::Point(tx+2,ty),cv::Scalar(0,255,0));
                }
                hitCnt++;
            }
            totalCnt++;
        }
        if (hit) {
            // transform points back to depth coordinate system:
            Eigen::Vector4f p = baseline.inverse() * nearestHitPoint;
            Eigen::Vector2f pu(p(0)/p(2),p(1)/p(2));
            Eigen::Vector2f pd;
            radialDistort(pu,&distDepth[0],KL,pd);
            depthPoints2D.push_back(cv::Point2f(pd(0),pd(1)));
            depthPoints3D.push_back(cv::Point3f(p(0),p(1),p(2)));
            mask.push_back(1);
        } else {
            depthPoints2D.push_back(cv::Point2f(0,0));
            depthPoints3D.push_back(cv::Point3f(0,0,0));
            mask.push_back(0);
        }
    }
        //    imwrite("scratch/testimg.ppm",testImg);
        //    printf("hit percentage: %f\n",hitCnt/totalCnt);
}

int ZConv::d2z(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    Mat dispImage(height, width, CV_16UC1, dptr);
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

//    maxDist = 4000;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    if (bilateralFiltering) {
        Mat dispImageHdr(height, width, CV_32FC1);
        Mat dispImageHdr2(zheight, zwidth, CV_32FC1);
        multicore->replaceUShortRange(dispImage, 2047, 0xffff, 0xffff);
        multicore->convert2Float(dispImage,dispImageHdr);
		cv::bilateralFilter(dispImageHdr, dispImageHdr2, -1, 10.0f, 3.0f);
        if (width == zwidth) {
			multicore->d2ZHdr(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        } else {
            multicore->d2ZLowHdr(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        }
    } else {
        if (width == zwidth) {
//            printf("1converting input disparity map from %d x %d -> %d x %d!\n",width,height,zwidth,zheight);
            multicore->d2Z(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        } else {
    //        printf("2converting input disparity map from %d x %d -> %d x %d!\n",width,height,zwidth,zheight);
            multicore->d2ZLow(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        }
    }
    calib->setupCalibDataBuffer(prevW,prevH);
	return 1;
}

void ZConv::setRange(float*ptr, int len, float minZ, float maxZ, float z) {
    for (int i = 0; i < len; i++) if (ptr[i] >= minZ && ptr[i] <= maxZ) ptr[i] = z;
}

int ZConv::d2zHdr(float *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    Mat dispImage(height, width, CV_32FC1, dptr);
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    if (bilateralFiltering) {
        Mat dispImageHdr2(zheight, zwidth, CV_32FC1);
        cv::bilateralFilter(dispImage, dispImageHdr2, -1, 3.0, 3.0f);
        multicore->d2ZLowHdr(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
    } else {
        multicore->d2ZLowHdr(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
    }
    calib->setupCalibDataBuffer(prevW,prevH);
	return 1;
}

//NOTE: this routine normalizes depth map (gpu compatible form)!
int ZConv::d2zGPU(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib) {
    Mat dispImage(height, width, CV_16UC1, dptr);
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }


    multicore->d2ZLowGPU(dispImage,depthImage,c0,c1,minDist,maxDist, xOff, yOff);
    calib->setupCalibDataBuffer(prevW,prevH);
	return 1;
}


//NOTE: this routine will not normalize depth map!
int ZConv::convert(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib) {
    assert(width == 640 && height == 480);
    assert(zwidth == 320 && zheight == 240);
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *kcR = &calib->getCalibData()[KcR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];

    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat dispImage(height, width, CV_16UC1, dptr);
    Mat depthImageSmall(zheight,    zwidth, CV_32FC1,zptr);

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    OMPFunctions *multicore = getMultiCoreDevice();   
    multicore->d2ZLow(dispImage, depthImageSmall, c0,c1,minDist,maxDist, xOff, yOff);
    calib->setupCalibDataBuffer(prevW,prevH);
	return 1;
}

