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

#include <Keyframe.h>
#include <GL/glew.h>
#include <basic_math.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <omp.h>
#include "zconv.h"
#include "png_loader.cpp"
#include "eigenMath.h"
using namespace Eigen;
using namespace cv;
using namespace std;

// active pose determines which pose matrix is used: bundle adjusted or the original
int activePose = 0;

void genPixelCoord(Eigen::Vector3f &s, float *kc, Eigen::Matrix3f &K) {
    float dx  = s(0)*s(0);
    float dy  = s(1)*s(1);
    // generate distorted coordinates
    float r2 = dx+dy, r4 = r2*r2, r6 = r4 * r2;
    float radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
    s(0) *= radialDist;
    s(1) *= radialDist;
    s = K*s;
}

void Keyframe::resetVariables() {
	rgbTex = -1;
    for (int i = 0; i < maxLayers; i++) {
		depthTex[i] = -1;
		grayTex[i]  = -1;
    }
    pointGrid = NULL; vertexBufferHost = NULL; indexBufferHost = NULL;
    normalBufferHost = NULL; tangentBufferHost=NULL; uvBufferHost=NULL; colorBufferHost = NULL;
    numElements = 0;
    id = -1;
    width = 0;
    height = 0;
    vbo = -1;
    ibo = -1;
	featureLength = 0;
    numNeighbors=0;
    visible=true;
	costMetric = HAMMING;
    poseInitialized = false;
    resetPose();
    intrinsic = Eigen::Matrix3f::Identity();
    memset(distcoeff,0,sizeof(distcoeff));
    center[0]  = 0; center[1]  = 0; center[2]  = 0;
    extents[0] = 0; extents[1] = 0; extents[2] = 0;
    // construct the planar pattern
    pattern_size = cv::Size(0,0);
    pattern_size.height = 6;
    pattern_size.width = 8;
    csize = 150.0f;//30.0f;
    pattern.clear();
    corners.clear();
    modelCorners.clear();
    for (float j=0; j<pattern_size.height; j++)
        for (float i=0; i<pattern_size.width; i++)
            pattern.push_back(cv::Point3f(i*csize,j*csize,0));
}

bool loadDepthmap(const char *depthfile, float *depthMapRaw, int depthMapSize, float dataScale) {
    bool loadOk = false;
    FILE *f = fopen(depthfile,"rb");
    if (f != NULL) {
        loadOk = true;
        fread(depthMapRaw,sizeof(float),depthMapSize,f);
        fclose(f);
        for (int i = 0; i < depthMapSize; i++) {
            depthMapRaw[i] *= dataScale;
        }
    }
    return loadOk;
}

void Keyframe::allocateMaps(int width, int height) {
    pointGrid = new ProjectData[width*height];
    for (int i = 0; i < maxLayers; i++) {
        depthMap[i] = cv::Mat(height>>i,width>>i,CV_32FC1);
        tmpDepthMap[i] = cv::Mat(height>>i,width>>i,CV_32FC1);
        depthTextureHost[i] = cv::Mat(height>>i,width>>i,CV_32FC1);
        memset(depthMap[i].ptr(),0, sizeof(float)*(width>>i)*(height>>i));
        memset(tmpDepthMap[i].ptr(),0, sizeof(float)*(width>>i)*(height>>i));
        memset(depthTextureHost[i].ptr(),0, sizeof(float)*(width>>i)*(height>>i));
    }
}

Keyframe::Keyframe(char *path, int id, Calibration &calib, Eigen::Matrix4f &depthPose, Eigen::Matrix4f &baselineExt, cv::Ptr<cv::DescriptorExtractor> &extractor, FeatureCostMetric metric, bool *useLayer, float depthScale) {
    resetVariables();
    this->id = id;
    this->costMetric = metric;
    char imagefile[512];
	sprintf(imagefile,"%s/%04d.ppm",path,id);

    setupNormalizedIntrinsics(calib, intrinsic, intrinsicDepth, distcoeff);
    pose[activePose]  = depthPose;
    pose[!activePose] = depthPose;
    baseline = baselineExt;

    bool depthOk = false;
    bool colorOk = false;

	cv::Mat inputImage = imread(imagefile,-1);
    if (inputImage.rows > 0 && inputImage.cols > 0 && inputImage.ptr() && inputImage.channels() == 3) {
        rgbImage = cv::Mat(inputImage.rows,inputImage.cols,CV_32FC3);
        rgbImagePacked = cv::Mat(inputImage.rows,inputImage.cols,CV_8UC3);
        printf("loaded %s (%dx%dx%d)\n",imagefile, rgbImage.cols,rgbImage.rows,3);
        imageSize = rgbImage.cols*rgbImage.rows;
        for (int i = 0; i < maxLayers; i++) {
            data1C[i]   = cv::Mat(rgbImage.rows>>i,rgbImage.cols>>i,CV_32FC1);
            memset(data1C[i].ptr(),  0, sizeof(float)*(rgbImage.cols>>i)*(rgbImage.rows>>i));
        }
        generateTexture(inputImage);
        colorOk = true;
    } else {
        printf("%s could not be found!\n",imagefile); fflush(stdout);
        return;
    }

    char depthfile[512];
    sprintf(depthfile,"%s/rawdepth%04d.ppm",path,id);
    cv::Mat dispHeader = imread(depthfile,-1);
    if (dispHeader.rows > 0 && dispHeader.cols > 0 && dispHeader.ptr() && dispHeader.channels() == 1) {
        width = dispHeader.cols; height = dispHeader.rows;
        printf("loaded %s (%dx%dx%d)\n",depthfile, width,height,2);
        allocateMaps(width,height);
        if (!disparityToDepth(dispHeader,calib,depthMap[0])) {
            printf("disparity to depth conversion failed!!\n");
            fflush(stdout);
            return;
        }
        generateDepthTextures();
        depthOk = true;
    } else {
        sprintf(depthfile,"%s/depthmap-%04d.dat",path,id);
        width = inputImage.cols; height = inputImage.rows;
        allocateMaps(width,height);
        if (!loadDepthmap(depthfile, (float*)depthMap[0].ptr(),width*height,depthScale)) {
            printf("%s not found!\n",depthfile);
            return;
        }
        printf("%s found!\n",depthfile);
        generateDepthTextures();
        depthOk = true;
    }

    if (depthOk && colorOk) {
        processDepthmaps(inputImage,calib);
        //extractFeatures(extractor,useLayer);
    }
}

void Keyframe::extractFeatures(cv::Ptr<cv::DescriptorExtractor> extractor,bool *useLayer) {
    // extract harris corners:
    //default: maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=false, double k=0.04
	GoodFeaturesToTrackDetector harrisDetector(1000.0f,0.025f,10,3,true);
	Eigen::MatrixXf featureData[maxLayers]; int nFeaturePoints = 0; int rows=0;
    for (int layer = 0; layer < maxLayers; layer++) {
        if (!useLayer[layer]) continue;
		cv::Mat grayImageHDR(height>>layer,width>>layer,CV_32FC1,data1C[layer].ptr());
        cv::Mat grayImage(height>>layer,width>>layer,CV_8UC1);
        grayImageHDR.convertTo(grayImage,CV_8UC1,255.0,0);
        vector<KeyPoint> points; Mat descriptors;
		//FAST(grayImage,points,50,true);
		harrisDetector.detect(grayImage,points);
       // printf("%d feature points detected!\n",int(points.size()));
        extractor->compute(grayImage, points, descriptors);
       // printf("descriptors found for %d points ( size : %d )!\n",descriptors.rows,descriptors.cols);
        if (descriptors.rows > 0) {
            if (descriptors.type() == CV_8UC1) printf("descriptors are bytes!\n");
            // convert to eigen format:
            featureData[layer] = Eigen::MatrixXf(descriptors.cols+8,descriptors.rows);
            for (size_t pi = 0; pi < points.size(); pi++) {
                featureData[layer](0,pi) = float(points[pi].pt.x)*float(1<<layer);
                featureData[layer](1,pi) = float(points[pi].pt.y)*float(1<<layer);
                featureData[layer](2,pi) = 1.0f;
				featureData[layer](3,pi) = featureData[layer](0,pi);
				featureData[layer](4,pi) = featureData[layer](1,pi);
                featureData[layer](5,pi) = featureData[layer](2,pi);
                featureData[layer](6,pi) = layer;
                featureData[layer](7,pi) = -1; // id is invalid for now, since no 3d points associated
            }
			rows = descriptors.cols+8;
            nFeaturePoints += points.size();
			printf("%d feature points added, total found:%d!\n",int(points.size()),nFeaturePoints);
            for (size_t j = 0; j < descriptors.rows; j++) {
                for (size_t i = 0; i < descriptors.cols; i++) {
                    featureData[layer](i+8,j) = descriptors.at<unsigned char>(j,i);
                }
            }
        }
    }
	if (rows == 0 || nFeaturePoints == 0) {
        return;
    }
    // concat matrices:
	features = Eigen::MatrixXf(rows,nFeaturePoints);
    int startCol = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        if (featureData[layer].rows() > 0 && featureData[layer].cols() > 0) {
            features.block(0,startCol,rows,featureData[layer].cols()) = featureData[layer];
            startCol += featureData[layer].cols();
        }
    }
	featureLength = features.rows()-8;
    // produce undistorted points:
    for (int i = 0; i < nFeaturePoints; i++) {
        Eigen::Vector3f p = features.block(0,i,3,1),r;
        undistort(p,r);
        features.block(0,i,3,1) = r;
	}
	generate3DFeatures();
}

void Keyframe::genAABB(Tri2D &tri, ProjectData *gridData, Eigen::Vector2f &boundsMin, Eigen::Vector2f &boundsMax) {
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

void Keyframe::storeTriangleToBuckets(Tri2D &tri,ProjectData *gridData,std::map<int,std::vector<Tri2D> > &buckets, float xBuckets, float yBuckets, float xBucketSize, float yBucketSize)
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

bool Keyframe::testQuad(int *index, ProjectData *grid, float minZ) {
	for (int k = 0; k < 4; k++) {
		ProjectData &p = grid[index[k]];
		if (fabs(p.pz) < minZ) return false;
	}
	return true;
}

bool Keyframe::testTri(Tri2D &tri, ProjectData *grid, float minZ) {
	for (int k = 0; k < 3; k++) {
		ProjectData &p = grid[tri.index[k]];
		if (fabs(p.pz) < minZ) return false;
	}
	return true;
}


bool Keyframe::hitFound(float px, float py, Tri2D &tri, ProjectData *grid, Eigen::Vector4f &hitPoint) {
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

void Keyframe::generate3DFeatures() {
	// generate 3d feature points:
	// 1) warp each triangle T_i into 2d rgb imaging plane
	// 2) fetch related features using bucketing structure
	// 3) do triangle test for each related feature f_i
	// 3) if f_i is within T_i -> solve depth using baryocentric coordinates
	// 4)     warp 3d point in RGB view back to IR view -> store to an array
	featurePoints3D  = Eigen::MatrixXf(4,features.cols());
	statusPoint3D    = Eigen::MatrixXf(3,features.cols());

	float xBucketSize = 4;
	float yBucketSize = 4;
	float xBuckets = width /xBucketSize;
	float yBuckets = height/yBucketSize;
	std::map<int,std::vector<Tri2D> > buckets;
	for (int by = 0; by < yBuckets; by++) {
		for (int bx = 0; bx < xBuckets; bx++) {
			std::vector<Tri2D> &bucket = buckets[bx+by*int(xBuckets)];
			bucket.reserve(100);
		}
	}

	cv::Mat testImg(height,width,CV_8UC3);
	unsigned char *ptr = testImg.ptr(); memset(ptr,0,width*height*3);

	for (int j = 0; j < height-1; j++) {
		for (int i = 0; i < width-1; i++) {
			int index[4];
			int offset = i+j*width;
			index[0] = offset;
			index[1] = offset+1;
			index[2] = offset+width+1;
			index[3] = offset+width;
			// store triangles to buckets:
			Tri2D triA(index[1], index[2], index[0]);
			if (testTri(triA,pointGrid,100.0f)) {
				storeTriangleToBuckets(triA,pointGrid,buckets,xBuckets,yBuckets,xBucketSize,yBucketSize);
				cv::line(testImg,cv::Point(pointGrid[triA.index[0]].rx2,pointGrid[triA.index[0]].ry2),cv::Point(pointGrid[triA.index[1]].rx2,pointGrid[triA.index[1]].ry2),cv::Scalar(100,100,100),1);
				cv::line(testImg,cv::Point(pointGrid[triA.index[1]].rx2,pointGrid[triA.index[1]].ry2),cv::Point(pointGrid[triA.index[2]].rx2,pointGrid[triA.index[2]].ry2),cv::Scalar(100,100,100),1);
				cv::line(testImg,cv::Point(pointGrid[triA.index[2]].rx2,pointGrid[triA.index[2]].ry2),cv::Point(pointGrid[triA.index[0]].rx2,pointGrid[triA.index[0]].ry2),cv::Scalar(100,100,100),1);
			}
			Tri2D triB(index[3], index[0], index[2]);
			if (testTri(triB,pointGrid,100.0f)) {
				storeTriangleToBuckets(triB,pointGrid,buckets,xBuckets,yBuckets,xBucketSize,yBucketSize);
				cv::line(testImg,cv::Point(pointGrid[triB.index[0]].rx2,pointGrid[triB.index[0]].ry2),cv::Point(pointGrid[triB.index[1]].rx2,pointGrid[triB.index[1]].ry2),cv::Scalar(100,100,100),1);
				cv::line(testImg,cv::Point(pointGrid[triB.index[1]].rx2,pointGrid[triB.index[1]].ry2),cv::Point(pointGrid[triB.index[2]].rx2,pointGrid[triB.index[2]].ry2),cv::Scalar(100,100,100),1);
				cv::line(testImg,cv::Point(pointGrid[triB.index[2]].rx2,pointGrid[triB.index[2]].ry2),cv::Point(pointGrid[triB.index[0]].rx2,pointGrid[triB.index[0]].ry2),cv::Scalar(100,100,100),1);
			}
		}
	}

	float hitCnt = 0;
	float totalCnt = 0;
	for (size_t fi = 0; fi < features.cols(); fi++) {
		float px = features(3,fi);
		float py = features(4,fi);
		int bx = int(px/xBucketSize);
		int by = int(py/yBucketSize);
		if (bx < 0 || by < 0) continue;
		if (bx >= xBuckets || by >= yBuckets) continue;

		cv::circle(testImg,cv::Point(px,py),3,cv::Scalar(0,0,255),1,-1);

		bool hit = false;
		Eigen::Vector4f hitPoint;
		Eigen::Vector4f nearestHitPoint(0,0,FLT_MAX,1);
		std::vector<Tri2D> &bucket = buckets[bx+by*int(xBuckets)];
		for (size_t bi = 0; bi < bucket.size(); bi++) {
			Tri2D &tri = bucket[bi];
			if (hitFound(px,py,tri,pointGrid,hitPoint)) {
				if (fabs(hitPoint(2)) < fabs(nearestHitPoint(2))) {
					nearestHitPoint = hitPoint;
					hit = true;
					float tx = pointGrid[tri.index[0]].rx2+pointGrid[tri.index[1]].rx2+pointGrid[tri.index[2]].rx2; tx /= 3;
					float ty = pointGrid[tri.index[0]].ry2+pointGrid[tri.index[1]].ry2+pointGrid[tri.index[2]].ry2; ty /= 3;
					cv::line(testImg,cv::Point(tx,ty+2),cv::Point(tx,ty-2),cv::Scalar(0,255,0));
					cv::line(testImg,cv::Point(tx-2,ty),cv::Point(tx+2,ty),cv::Scalar(0,255,0));
				}
				hitCnt++;
			}
			totalCnt++;
		}
		if (hit) {
			// transform points back to depth coordinate system:
			Eigen::Vector4f p = baseline.inverse() * nearestHitPoint;
			featurePoints3D(0,fi)  = float(p(0));
			featurePoints3D(1,fi)  = float(p(1));
			featurePoints3D(2,fi)  = float(p(2));
			featurePoints3D(3,fi)  = 1.0f;
			statusPoint3D(0,fi)    = FEATURE3D_POINT_UNASSIGNED;
			statusPoint3D(1,fi)    = 0.0f;
			statusPoint3D(2,fi)    = 0.0f;
		} else {
			featurePoints3D(0,fi)  = 0;
			featurePoints3D(1,fi)  = 0;
			featurePoints3D(2,fi)  = 0;
			featurePoints3D(3,fi)  = 1.0f;
			statusPoint3D(0,fi)    = FEATURE3D_POINT_INVALID;
			statusPoint3D(1,fi)    = 0.0f;
			statusPoint3D(2,fi)    = 0.0f;
		}
	}
	imwrite("scratch/testimg.ppm",testImg);

	printf("hit percentage: %f\n",hitCnt/totalCnt);
}

void Keyframe::addNeighbor(Keyframe *kf) {
    if (numNeighbors>=maxNeighborAmount-1) return;
    if (isNeighbor(kf)) return;
    neighbors[numNeighbors] = kf; numNeighbors++;
}

bool Keyframe::isNeighbor(Keyframe *kf) {
    if (kf == NULL) return false;
    for (int i = 0; i < numNeighbors; i++) {
        if (kf == neighbors[i]) return true;
    }
    return false;
}

void Keyframe::poseDistance(Eigen::Matrix4f &testPose, float *transDist, float *rotDist) {
	Eigen::Matrix4f dT =  pose[activePose] * testPose.inverse();
    Matrix3f mat = dT.block(0,0,3,3);
    AngleAxisf aa(mat);
    *rotDist = fabs(aa.angle());
    *transDist = (float)sqrt(dT(0,3)*dT(0,3)+dT(1,3)*dT(1,3)+dT(2,3)*dT(2,3)+1e-8f);
}

int readLine(char *filebuf, char *buf) {
    int off = 0;
    while (filebuf[off] != '\n') { buf[off] = filebuf[off]; off++; }
    buf[off] = '\0';
    return off+1;
}

bool isCommentLine(char *linebuf, int limit) {
    int i = 0;
    for (i = 0; i < limit; i++) {
        if (linebuf[i] != ' ') break;
    }
    return (linebuf[i] == '#');
}

char *loadBufferFromFile(char *filename, int *sz) {
    FILE *f = fopen(filename,"rb");
    if (f == 0) return NULL;
    fseek(f,0,SEEK_END);
    *sz = ftell(f);
    fseek(f,0,SEEK_SET);
    char *buf = new char[*sz];
    fread(buf,1,*sz,f);
    fclose(f);
    return buf;
}


Eigen::Matrix4f Keyframe::poseGL(int slot) {
    Eigen::Matrix4f bose2 = pose[slot];
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

Eigen::Matrix3f Keyframe::intrinsicGL() {
    Eigen::Matrix3f intrinsic2 = intrinsic;
    intrinsic2(0,0) *= -1.0f;
    return intrinsic2;
}

void Keyframe::undistortPixels(Eigen::Vector3f &p, Eigen::Vector3f &r) {
    Eigen::Matrix3f intrinsic  = scaleIntrinsic(0);
    Eigen::Matrix3f intrinsicT = scaleIntrinsic(0).transpose();

    // collect distorted measurements into opencv matrix
    cv::Mat pointsA(1,1,CV_32FC2);  float *ptrA  = (float*)pointsA.ptr(); ptrA[0] = p(0); ptrA[1] = p(1);
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, intrinsicT.data());
    cv::Mat distCoeffs   = cv::Mat(5, 1, CV_32FC1, distcoeff);

    undistortPoints(pointsA, pointsA, cameraMatrix, distCoeffs);

    // write undistorted values:
    Eigen::Vector3f pnorm(ptrA[0],ptrA[1],1);
    Eigen::Vector3f pu = intrinsic * pnorm;
    r(0) = pu(0);
    r(1) = pu(1);
    r(2) = pu(2);
}

void Keyframe::undistort(Eigen::Vector3f &p, Eigen::Vector3f &r) {
	Eigen::Matrix3f intrinsicT = scaleIntrinsic(0).transpose();

    // collect distorted measurements into opencv matrix
    cv::Mat pointsA(1,1,CV_32FC2);  float *ptrA  = (float*)pointsA.ptr(); ptrA[0] = p(0); ptrA[1] = p(1);
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, intrinsicT.data());
    cv::Mat distCoeffs   = cv::Mat(5, 1, CV_32FC1, distcoeff);

    undistortPoints(pointsA, pointsA, cameraMatrix, distCoeffs);

    // write undistorted values:
    Eigen::Vector3f pnorm(ptrA[0],ptrA[1],1);
    r = pnorm;
 }


void Keyframe::distort(Eigen::Vector3f &p, Eigen::Vector3f &r) {
    Eigen::Matrix3f intrinsic  = scaleIntrinsic(0);
    Eigen::Vector3f ptmp = intrinsic.inverse() * p;
//    float dcoeff[5]; for (int i = 0; i < 5; i++) dcoeff[i] = distcoeff[i]*100.0f;
    genPixelCoord(ptmp,distcoeff,intrinsic);
    r(0) = ptmp(0);
    r(1) = ptmp(1);
    r(2) = ptmp(2);
}

typedef struct {
    int id;
    float x,y;
} MEASUREMENT;

bool Keyframe::loadMeasurements(char *measurementsfile) {
    int sz = 0;
    char *buf = loadBufferFromFile(measurementsfile,&sz);
    if (buf == NULL) return false;

    std::vector<MEASUREMENT> measurementVector;
    int off = 0;
    char linebuf[512];
    while (off < sz) {
        int linecnt = readLine(&buf[off],linebuf);
        off+=linecnt;
        if (isCommentLine(linebuf,linecnt)) {
            continue;
        }
        MEASUREMENT m;
        sscanf(linebuf,"%d %f %f",&m.id,&m.x,&m.y);
        measurementVector.push_back(m);
    }
    delete[] buf;

    measurements = Eigen::MatrixXf(8,measurementVector.size());

    for (int i = 0; i < measurements.cols(); i++) {
        measurements(0,i)  = measurementVector[i].x;
        measurements(1,i)  = measurementVector[i].y;
        measurements(2,i)  = 1.0;
        measurements(3,i)  = measurementVector[i].x;
        measurements(4,i)  = measurementVector[i].y;
        measurements(5,i)  = 1.0;
        measurements(6,i)  = 0;
        measurements(7,i)  = measurementVector[i].id;
    }

    Eigen::Matrix3f intrinsicT = scaleIntrinsic(0).transpose();

    // collect distorted measurements into opencv matrix
    cv::Mat pointsA(1,measurements.cols(),CV_32FC2);  float *ptrA  = (float*)pointsA.ptr(); for (int i = 0; i < measurements.cols(); i++) { ptrA[i*2+0] = measurements(0,i); ptrA[i*2+1] = measurements(1,i); }
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, intrinsicT.data());
    cv::Mat distCoeffs   = cv::Mat(5, 1, CV_32FC1, distcoeff);

    undistortPoints(pointsA, pointsA, cameraMatrix, distCoeffs);
    // write undistorted values:
    for (int i = 0; i < measurements.cols(); i++) {
        Eigen::Vector3f pnorm(ptrA[i*2+0],ptrA[i*2+1],1);
        measurements(0,i) = pnorm(0);
        measurements(1,i) = pnorm(1);
    }

    return true;
}

void Keyframe::loadLandmarks(char *landmarkfile, std::vector<LANDMARK> &landmarks) {
    landmarks.clear();
    int sz = 0;
    char *buf = loadBufferFromFile(landmarkfile,&sz);
    if (buf == NULL) return;

    int off = 0;
    char linebuf[512];
    while (off < sz) {
        int linecnt = readLine(&buf[off],linebuf);
        off+=linecnt;
        if (isCommentLine(linebuf,linecnt)) {
            continue;
        }
        int id; float p[3];
        sscanf(linebuf,"%d %f %f %f",&id,&p[0],&p[1],&p[2]);
        LANDMARK m;
        m.id = id; m.p(0) = float(p[0]); m.p(1) = float(p[1]); m.p(2) = float(p[2]);
        landmarks.push_back(m);
    }
    delete[] buf;
    return;
}



bool Keyframe::loadCalib(char *calibfile) {
    int sz = 0;
    char *buf = loadBufferFromFile(calibfile,&sz);
    if (buf == NULL) return false;
    float cdata[6]; float *dst = &cdata[0];
    int off = 0;
    char linebuf[512];
    while (1) {
        int linecnt = readLine(&buf[off],linebuf);
        off+=linecnt;
        if (isCommentLine(linebuf,linecnt)) {
            //printf("found comment line\n");
            continue;
        }
        sscanf(linebuf,"%f %f %f %f %f %f",&dst[0],&dst[1],&dst[2],&dst[3],&dst[4],&dst[5]);
        break;
    }
    delete[] buf;

    float wf = float(width);    
    intrinsic(0,0) = cdata[2]/wf;  intrinsic(0,1) = 0.0f;        intrinsic(0,2) = cdata[0]/(2.0f*wf);
    intrinsic(1,0) = 0.0f;         intrinsic(1,1) = cdata[3]/wf; intrinsic(1,2) = cdata[1]/(2.0f*wf);
    intrinsic(2,0) = 0.0f;         intrinsic(2,1) = 0.0f;        intrinsic(2,2) = 1;

    distcoeff[0] = cdata[4]; distcoeff[1] = cdata[5]; distcoeff[2] = 0.0f; distcoeff[3] = 0.0f; distcoeff[4] = 0.0f;
    return true;
}

void Keyframe::getFov(float *fovX, float *fovY) {
    // note: principal point is assumed to be in the middle of the screen!
    float normalizedK11 = 2*intrinsic(0,0);
    *fovX = 180.0f*2.0f*atan(1.0f/fabs(normalizedK11))/3.141592653f;
    float normalizedK22 = 2*intrinsic(1,1);
    *fovY = 180.0f*2.0f*atan((3.0f/4.0f)/fabs(normalizedK22))/3.141592653f;
}

Keyframe::~Keyframe() {
   // release();
}

void Keyframe::release() {
	rgbImage.release();
    rgbImagePacked.release();
	if (rgbTex>=0) {
        glDeleteTextures(1, &rgbTex);
    }
	if (pointGrid!=NULL) delete[] pointGrid;
    for (int i = 0; i < maxLayers; i++) {
		data1C[i].release();
		if (grayTex[i]>=0) {
            glDeleteTextures(1, &grayTex[i]);
        }
		if (depthTex[i]>=0) {
			depthMap[i].release();
            depthTextureHost[i].release();
            glDeleteTextures(1, &depthTex[i]);
        }
    }    
    releaseVbo();
}

void Keyframe::releaseVbo() {
    if (vbo != -1) {
        glDeleteBuffers(1, &vbo);
        vbo = -1;
        if (vertexBufferHost) delete[] vertexBufferHost; vertexBufferHost = NULL;
 //       if (normalBufferHost) delete[] normalBufferHost; normalBufferHost = NULL;
        if (tangentBufferHost) delete[] tangentBufferHost; tangentBufferHost = NULL;
        if (uvBufferHost) delete[] uvBufferHost; uvBufferHost = NULL;
        if (colorBufferHost) delete[] colorBufferHost; colorBufferHost = NULL;
    }
    if (ibo != -1) {
        glDeleteBuffers(1, &ibo);
        ibo = -1;
        if (indexBufferHost) delete[] indexBufferHost; indexBufferHost = NULL;
    }
}

void Keyframe::generatePyramid1C() {
    int w = data1C[0].cols;
    int h = data1C[0].rows;
    if (w > 0 && h > 0) {
        for (int i = 1; i < maxLayers; i++) {
            cv::Mat hires(h>>(i-1),w>>(i-1),CV_32FC1,data1C[i-1].ptr());
            cv::Mat hires2(h>>(i-1),w>>(i-1),CV_32FC1);
            cv::Mat lowres(h>>i,w>>i,CV_32FC1,data1C[i].ptr());
            // note: pyrdown introduces shifting
            cv::GaussianBlur(hires,hires2,cv::Size2f(5,5),0,0);
            cv::resize(hires2,lowres,cv::Size(w>>i,h>>i));
        }
    }
}

void Keyframe::resetPose() {
    pose[1] = pose[0] = Eigen::Matrix4f::Identity();
}

void Keyframe::getDepthPattern(std::vector<cv::Point3f> &camPattern) {
    camPattern.resize(pattern.size());
    Eigen::Matrix4f mtx = pose[0];
    for (size_t i = 0; i < pattern.size(); i++) {
        cv::Point3f &p  = pattern[i];
        cv::Point3f &cp = camPattern[i];
        cp.x = mtx(0,0)*p.x + mtx(0,1)*p.y + mtx(0,2)*p.z + mtx(0,3);
        cp.y = mtx(1,0)*p.x + mtx(1,1)*p.y + mtx(1,2)*p.z + mtx(1,3);
        cp.z = mtx(2,0)*p.x + mtx(2,1)*p.y + mtx(2,2)*p.z + mtx(2,3);
    }
/*
    cv::Mat rvec =  cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec =  cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs   = cv::Mat(1, 5, CV_64FC1);
    Eigen::Matrix3f intrinsic = scaleIntrinsic(0);
    for (int i = 0; i < 5; i++) distCoeffs.at<double>(0,i) = distcoeff[i];
    for (int r = 0; r < 3; r++)
        for (int p = 0; p < 3; p++)
            cameraMatrix.at<double>(r,p) = intrinsic(r,p);
    modelCorners.clear();
    cv::projectPoints(camPattern,rvec,tvec,cameraMatrix,distCoeffs,modelCorners);*/
}


void Keyframe::estimatePairwisePose(std::vector<cv::Point3f> &camPattern) {
    // Currently assuming zero distortion
    Eigen::Matrix3f intrinsic = scaleIntrinsic(0);//.transpose();
    // collect distorted measurements into opencv matrix
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64FC1);
    cv::Mat distCoeffs   = cv::Mat(1, 5, CV_64FC1);

    for (int i = 0; i < 5; i++) distCoeffs.at<double>(0,i) = distcoeff[i];
    for (int r = 0; r < 3; r++)
        for (int p = 0; p < 3; p++)
            cameraMatrix.at<double>(r,p) = intrinsic(r,p);

    cv::Mat rvec,tvec;
    cv::solvePnPRansac(camPattern, corners, cameraMatrix, distCoeffs, rvec, tvec,false,100,5.0f,corners.size());

    Eigen::Vector3d t(tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0));
    printf("tvec size: %d x %d\n",tvec.cols,tvec.rows);
    printf("%f %f %f\n",t(0),t(1),t(2));

    cv::Mat rot3x3(3,3,CV_64FC1);
    cv::Rodrigues(rvec, rot3x3);
    Eigen::Matrix3f K = scaleIntrinsic(0);
    double *R = (double*)rot3x3.ptr();
    double *T = (double*)tvec.ptr();

    bool toggle = true;
    modelCorners.clear();
    if (toggle) {
        cv::projectPoints(camPattern,rot3x3,tvec,cameraMatrix,distCoeffs,modelCorners);
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; r++)
            for (int p = 0; p < 3; p++)
                m(r,p) = rot3x3.at<double>(r,p);
        m(0,3) = t(0);
        m(1,3) = t(1);
        m(2,3) = t(2);
        pose[0] = baseline.inverse()*m;
        pose[1] = pose[0];
    } else {
        printf("\nCalibration results:\n"); fflush(stdout);
        for (size_t i = 0; i < camPattern.size(); i++) {
            cv::Point3f p = camPattern[i];
            double X = p.x, Y = p.y, Z = p.z;
            // Eigen::Vector3d p3d(p.x,p.y,p.z);
            double x = rot3x3.at<double>(0,0)*X + rot3x3.at<double>(0,1)*Y + rot3x3.at<double>(0,2)*Z + tvec.at<double>(0,0);
            double y = rot3x3.at<double>(1,0)*X + rot3x3.at<double>(1,1)*Y + rot3x3.at<double>(1,2)*Z + tvec.at<double>(1,0);
            double z = rot3x3.at<double>(2,0)*X + rot3x3.at<double>(2,1)*Y + rot3x3.at<double>(2,2)*Z + tvec.at<double>(2,0);
            double r2, r4, r6, a1, a2, a3, cdist, icdist2;
            double xd, yd;
            z = z ? 1./z : 1;
            x *= z; y *= z;
            r2 = x*x + y*y;
            r4 = r2*r2;
           /* r6 = r4*r2;
            a1 = 0;//2*x*y;
            a2 = 0;//r2 + 2*x*x;
            a3 = 0;//r2 + 2*y*y;*/
            cdist = 1 + distcoeff[0]*r2 + distcoeff[1]*r4;// + distcoeff[4]*r6;
            icdist2 = 1;
            xd = x*cdist*icdist2;// + distcoeff[2]*a1 + distcoeff[3]*a2;
            yd = y*cdist*icdist2;// + distcoeff[2]*a3 + distcoeff[3]*a1;
            cv::Point2f mc(xd*K(0,0) + K(0,2),yd*K(1,1) + K(1,2));
            ///Eigen::Vector3d cr = rot*p3d + t;
            ///Eigen::Vector2f pu(cr(0)/cr(2),cr(1)/cr(2));
            ///Eigen::Vector2f pd;
            ///radialDistort(pu,distcoeff,K,pd);
            ///cv::Point2f mc(pd(0),pd(1));
            modelCorners.push_back(mc);
            //printf("\nReprojection error = %f\n\n", err); fflush(stdout);
        }

    }
}

void Keyframe::updateColorMap(cv::Mat &colorMap, int rIndex, int gIndex, int bIndex, bool findCheckerBoard, bool flipBoard) {
    if (colorMap.rows > 0 && colorMap.cols>0) {
        // convert color data into internal variables:
        unsigned char *data3C  = colorMap.ptr();
        unsigned char *dst3C   = rgbImagePacked.ptr();
        float *fdata3C         = (float*)rgbImage.ptr();
        float *mono            = (float*)data1C[0].ptr();
        for (int i = 0; i < imageSize; i++) {
            dst3C[i*3+0] = data3C[i*3+rIndex];
            dst3C[i*3+1] = data3C[i*3+gIndex];
            dst3C[i*3+2] = data3C[i*3+bIndex];
            fdata3C[i*3+0] = float(data3C[i*3+rIndex])/255.0f;
            fdata3C[i*3+1] = float(data3C[i*3+gIndex])/255.0f;
            fdata3C[i*3+2] = float(data3C[i*3+bIndex])/255.0f;
            mono[i]        = 0.21f*fdata3C[i*3+0] + 0.72f*fdata3C[i*3+1] + 0.07f*fdata3C[i*3+2];
        }
        glBindTexture(GL_TEXTURE_2D, rgbTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, colorMap.cols, colorMap.rows, GL_RGB, GL_UNSIGNED_BYTE, rgbImagePacked.ptr());//fdata3C);
        generatePyramid1C();

        if (findCheckerBoard) {
            corners.clear();
            std::vector<cv::Point2f> rawCorners;
            printf("finding checkerboard (%dx%d)\n",pattern_size.width,pattern_size.height);
            cv::findChessboardCorners(colorMap, pattern_size, rawCorners);
            if (rawCorners.size() == pattern_size.width * pattern_size.height) {
                for (int i = 0; i < pattern_size.width * pattern_size.height; i++)
                {
//                    corners.push_back(rawCorners[i]);
                    if (!flipBoard) corners.push_back(rawCorners[i]);
                    else corners.push_back(rawCorners[pattern_size.width * pattern_size.height-1-i]);
                }
                cv::Mat grayMap(colorMap.rows,colorMap.cols,CV_8UC1);
                cvtColor(colorMap, grayMap, CV_RGB2GRAY);
                cv::cornerSubPix(grayMap, corners, cv::Size(5,5), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 30, 0.1));
//                drawChessboardCorners(imgc, pattern_size, cv::Mat(corners), ret);
                printf("checkerboard found!\n"); fflush(stdout);
            }
        }

    }
}


void Keyframe::generateNormals(float minDist)
{
    int fw = width;
    int fh = height;
    int step = 5;
    int offset=0;
    for (int yi = 0; yi < (fh-step); yi++) {
        for (int xi = 0; xi < (fw-step); xi++) {
            offset = xi + yi*fw;
            float p[3],u[3],v[3];

            p[0] = pointGrid[offset].px;
            p[1] = pointGrid[offset].py;
            p[2] = pointGrid[offset].pz;
            u[0] = pointGrid[offset+step].px;
            u[1] = pointGrid[offset+step].py;
            u[2] = pointGrid[offset+step].pz;
            v[0] = pointGrid[offset+width*step].px;
            v[1] = pointGrid[offset+width*step].py;
            v[2] = pointGrid[offset+width*step].pz;

            float nu[3],nv[3],n[3]={0,0,0};
            if (fabs(p[2]) > minDist && fabs(u[2]) > minDist && fabs(v[2]) > minDist) {
                nu[0] = u[0] - p[0]; nu[1] = u[1] - p[1]; nu[2] = u[2] - p[2];
                nv[0] = v[0] - p[0]; nv[1] = v[1] - p[1]; nv[2] = v[2] - p[2];
                // compute normal as crossproduct
                n[0] =  nu[1] * nv[2] - nu[2] * nv[1];
                n[1] =-(nu[0] * nv[2] - nu[2] * nv[0]);
                n[2] =  nu[0] * nv[1] - nu[1] * nv[0];
                // normal to unit length
                float len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]+1e-5f);
                // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
                n[0] /= len; n[1] /= len; n[2] /= len;
            }
            vertexBufferHost[offset*6+3] = -n[0];
            vertexBufferHost[offset*6+4] = -n[1];
            vertexBufferHost[offset*6+5] = -n[2];
        }
    }
}

void Keyframe::filterZMap(cv::Mat &depthMapSrc, cv::Mat &depthMapDst, float zThreshold) {
    int h = 2;
    float gaussianKernel[5*5] = { 1,  4,  7,  4, 1,
                                  4, 16, 26, 16, 4,
                                  7, 26, 41, 26, 7,
                                  4, 16, 26, 16, 4,
                                  1,  4,  7,  4, 1};
    for (int i = 0; i < 25; i++) gaussianKernel[i] *= (1.0f/273.0f);
    float *src = (float*)depthMapSrc.ptr();
    float *dst = (float*)depthMapDst.ptr();
    for (int j = h; j < depthMapSrc.rows-h; j++) {
        for (int i = h; i < depthMapSrc.cols-h; i++) {
            int off = i+j*depthMapSrc.cols;
            float zref = src[off];
            if (zref == 0.0f) {
                // do not do any holefilling for now:
                dst[off] = 0.0f; continue;
            }
            float zstar = 0; float wsum = 0;
            float wz,ws,w;
            int yStart = -h*depthMapSrc.cols;
            for (int p = -h; p <= h; p++,yStart+=depthMapSrc.cols) {
                for (int r = -h; r <= h; r++) {
                    float z = src[off+r+yStart];
                    if (z == 0) continue;
                    // do not sample beyond depth discontinuities:
                    wz = 1.0f; if (fabs(z-zref) > zThreshold) wz = 0;
                    // spatially gaussian weights:
                    ws = gaussianKernel[(p+h)*5 + r + h];
                    w  = ws*wz;
                    zstar    += z*w;
                    wsum     += w;
                }
            }
            if (wsum > 0) dst[off] = zstar/wsum; else dst[off] = 0.0f;
        }
    }
}

void Keyframe::correctZMap(cv::Mat &depthMapSrc, cv::Mat &depthMapDst, int numPolyCoeffs, float *polyCoeffs) {
    if (numPolyCoeffs <= 0) { depthMapSrc.copyTo(depthMapDst); return; }
    float *src = (float*)depthMapSrc.ptr();
    float *dst = (float*)depthMapDst.ptr();
    for (int j = 0; j < depthMapSrc.rows; j++) {
        for (int i = 0; i < depthMapSrc.cols; i++) {
            int off = i+j*depthMapSrc.cols;
            float zref = src[off];
            if (zref == 0.0f) {
                // do not do any holefilling for now:
                dst[off] = 0.0f; continue;
            }
            float zterm = 1.0f;
            float polySum = 0.0f;
            for (int ci = 0; ci < numPolyCoeffs; ci++) {
                polySum += polyCoeffs[ci]*zterm;
                zterm *= zref;
            }
            dst[off] = polySum;
        }
    }
}

void Keyframe::updateDisparityMap(cv::Mat &colorMap, cv::Mat &disparityMap,Calibration &calib) {
    if (!disparityToDepth(disparityMap,calib,depthMap[0])) {
        printf("disparity to depth conversion failed!!\n");
        fflush(stdout);
        return;
    }
    processDepthmaps(colorMap,calib);
}

void Keyframe::processDepthmaps(cv::Mat &colorMap, Calibration &calib) {
    ZConv zconv;
    zconv.baselineWarpRGB((float*)depthMap[0].ptr(),colorMap.ptr(),pointGrid,width,height,calib);
    updateVbo();
    float *depth  = (float*)depthMap[0].ptr();
    float *depthGPU = (float*)depthTextureHost[0].ptr();
    for (int i = 0; i < depthMap[0].cols*depthMap[0].rows; i++) depthGPU[i] = (depth[i]-calib.getMinDist())/(calib.getMaxDist()-calib.getMinDist());
    glBindTexture(GL_TEXTURE_2D, depthTex[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_FLOAT, depthGPU);
}

bool Keyframe::disparityToDepth(cv::Mat &disparityMap, Calibration &calib, cv::Mat &depthMapOut) {
    if (disparityMap.rows > 0 && disparityMap.cols>0) {
        ZConv zconv;
        zconv.d2z((unsigned short*)disparityMap.ptr(),disparityMap.cols,disparityMap.rows,(float*)depthMapOut.ptr(),disparityMap.cols,disparityMap.rows,&calib,false);
        correctZMap(depthMapOut,tmpDepthMap[0],calib.getNumPolyCoeffs(),calib.getPolyCoeffs());
        tmpDepthMap[0].copyTo(depthMapOut);
        return true;
    }
    return false;
}

float *Keyframe::getDepthData() {
    return (float*)depthMap[0].ptr();
}

void Keyframe::updateVbo() {
    if (pointGrid == NULL || imageSize < 1) return;

    generateNormals(300.0f);

    center[0]  = 0; center[1]  = 0; center[2]  = 0;
    for (int i = 0; i < imageSize; i++) {
        float x = pointGrid[i].px;
        float y = pointGrid[i].py;
        float z = pointGrid[i].pz;
        vertexBufferHost[i*6+0] = x;
        vertexBufferHost[i*6+1] = y;
        vertexBufferHost[i*6+2] = z;
//        normalBufferHost[i*3+0] = 0;
//        normalBufferHost[i*3+1] = 0;
//        normalBufferHost[i*3+2] = 1;
        tangentBufferHost[i*4+0] = 1;
        tangentBufferHost[i*4+1] = 0;
        tangentBufferHost[i*4+2] = 0;
        tangentBufferHost[i*4+3] = 1;
        uvBufferHost[i*2+0] = pointGrid[i].rx2;
        uvBufferHost[i*2+1] = pointGrid[i].ry2;
        colorBufferHost[i*4+0] = int(pointGrid[i].colorR*255);
        colorBufferHost[i*4+1] = int(pointGrid[i].colorG*255);
        colorBufferHost[i*4+2] = int(pointGrid[i].colorB*255);
        colorBufferHost[i*4+3] = 1;
        center[0] += x;
        center[1] += y;
        center[2] += z;
    }
    center[0] /= float(imageSize);
    center[1] /= float(imageSize);
    center[2] /= float(imageSize);

    // compute extents:
    extents[0] = 0; extents[1] = 0; extents[2] = 0;
    for (int i = 0; i < imageSize; i++) {
        float extX = fabs(pointGrid[i].px-center[0]); if (extX > extents[0]) extents[0] = extX;
        float extY = fabs(pointGrid[i].py-center[1]); if (extY > extents[1]) extents[1] = extY;
        float extZ = fabs(pointGrid[i].pz-center[2]); if (extZ > extents[2]) extents[2] = extZ;
    }


    unsigned int size = imageSize*6;
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size * sizeof(float), vertexBufferHost);

    int dstOffset=0;
    for (int j = 1; j < (height-1); j++) {
        for (int i = 1; i < (width-1); i++) {
            int offset = i+j*width;
            ProjectData &p0 = pointGrid[offset]; //float n0 = randGrid[offset];
            ProjectData &p1 = pointGrid[offset+width]; //float n1 = randGrid[offset+texWidth];
            ProjectData &p2 = pointGrid[offset+1+width]; //float n2 = randGrid[offset+1+texWidth];
            ProjectData &p3 = pointGrid[offset+1]; //float n3 = randGrid[offset+1];
            if (p0.magGrad < 0) continue;
            if (p1.magGrad < 0) continue;
            if (p2.magGrad < 0) continue;
            if (p3.magGrad < 0) continue;
            float limit0 = fabs(MIN(MIN(MIN(p0.pz,p1.pz),p2.pz),p3.pz));
            float limit1 = fabs(MAX(MAX(MAX(p0.pz,p1.pz),p2.pz),p3.pz));
            if (limit0 < 0.0f || limit1 < 0.0f || fabs(limit0-limit1) > 7.0f) continue; // || fabs(limit0-limit1) > 5.0f) continue;
            indexBufferHost[dstOffset+0] = offset;
            indexBufferHost[dstOffset+1] = offset+width;
            indexBufferHost[dstOffset+2] = offset+1+width;
            indexBufferHost[dstOffset+3] = offset+1;
            dstOffset+=4;
        }
    }
    numElements = dstOffset;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,numElements*sizeof(int), indexBufferHost);
}

void Keyframe::writeMesh(const char *fn) {
    if (pointGrid == NULL || imageSize < 1) return;

    FILE *f = fopen(fn,"wb");
    if (f == NULL) { printf("unable to write file : %s\n",fn); fflush(stdout); return; }

    // Triangles, C# : WriteInt32Array(mesh.triangles);
    int triIndexCount = 6*numElements/4;
    fwrite(&triIndexCount,sizeof(int),1,f);
    for (int i = 0; i < numElements; i+=4) {
        // tri0
        fwrite(&indexBufferHost[i+0],sizeof(int),1,f);
        fwrite(&indexBufferHost[i+1],sizeof(int),1,f);
        fwrite(&indexBufferHost[i+2],sizeof(int),1,f);
        // tri1
        fwrite(&indexBufferHost[i+0],sizeof(int),1,f);
        fwrite(&indexBufferHost[i+2],sizeof(int),1,f);
        fwrite(&indexBufferHost[i+3],sizeof(int),1,f);
    }

    /* Vertices
    w.Write(mesh.vertexCount);
    WriteVector3Array(mesh.vertices);
    WriteVector3Array(mesh.normals);
    WriteVector4Array(mesh.tangents);
    WriteVector2Array(mesh.uv);
    WriteColor32Array(mesh.colors32);
    */

    int numCoords2 = imageSize*2;
    int numCoords3 = imageSize*3;
    int numCoords4 = imageSize*4;
    int numCoords6 = imageSize*6;
    fwrite(&imageSize,sizeof(int),1,f);
    fwrite(&numCoords6,sizeof(int),1,f); fwrite(&vertexBufferHost[0], sizeof(float),numCoords6,f);
    //fwrite(&numCoords3,sizeof(int),1,f); fwrite(&normalBufferHost[0], sizeof(float),numCoords3,f);
    fwrite(&numCoords4,sizeof(int),1,f); fwrite(&tangentBufferHost[0],sizeof(float),numCoords4,f);
    fwrite(&numCoords2,sizeof(int),1,f); fwrite(&uvBufferHost[0],     sizeof(float),numCoords2,f);
    fwrite(&numCoords4,sizeof(int),1,f); fwrite(&colorBufferHost[0],  sizeof(int),  numCoords4,f);

    /*
    // Bounds
    WriteVector3(mesh.bounds.center);
    WriteVector3(mesh.bounds.extents);
    */
    fwrite(&center[0],sizeof(float),3,f);
    fwrite(&extents[0],sizeof(float),3,f);
    fclose(f);
}

void Keyframe::allocateVbo() {
    glGenBuffers( 1, &vbo);
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, width*height * sizeof(float) * 6, NULL, GL_STREAM_COPY);
    vertexBufferHost  = new float[width*height*6];
    //normalBufferHost  = new float[width*height*3];
    tangentBufferHost = new float[width*height*4];
    uvBufferHost      = new float[width*height*2];
    colorBufferHost   = new int[width*height*4];

    if (vbo == 0) {
        printf("vbo allocation failed!\n");
        fflush(stdin); fflush(stdout); fflush(stderr);
    }
    assert(vbo != 0);


    indexBufferHost = new int[(width-1)*(height-1)*4];
    int dstOffset=0;
    for (int j = 1; j < (height-1); j++) {
        for (int i = 1; i < (width-1); i++) {
            int offset = i+j*width;
            indexBufferHost[dstOffset+0] = offset;
            indexBufferHost[dstOffset+1] = offset+width;
            indexBufferHost[dstOffset+2] = offset+1+width;
            indexBufferHost[dstOffset+3] = offset+1;
            dstOffset+=4;
        }
    }
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*(width-1)*(height-1)*sizeof(int), indexBufferHost, GL_STREAM_COPY);
    if (ibo == 0) {
        printf("ibo allocation failed!\n");
        fflush(stdin); fflush(stdout); fflush(stderr);
    }
}

void Keyframe::generateDepthTextures() {
    allocateVbo();
    int i = 0;
    if (depthMap[i].rows > 0 && depthMap[i].cols > 0) {
        glGenTextures(1, &depthTex[i]);
        glBindTexture(GL_TEXTURE_2D, depthTex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width>>i, height>>i, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    }
    /*
    for (int i = 0; i < maxLayers; i++) {
        if (data1C[i].cols > 0 && data1C[i].rows > 0) {
            glGenTextures(1, &grayTex[i]);
            glBindTexture(GL_TEXTURE_2D, grayTex[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width>>i, height>>i, 0, GL_LUMINANCE, GL_FLOAT, (float*)data1C[i].ptr());

            glGenTextures(1, &depthTex[i]);
            glBindTexture(GL_TEXTURE_2D, depthTex[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width>>i, height>>i, 0, GL_LUMINANCE, GL_FLOAT, (float*)depthMap[i].ptr());
        }
    }*/
}

void Keyframe::generateTexture(cv::Mat &colorMap) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &rgbTex);
    glBindTexture(GL_TEXTURE_2D, rgbTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 //   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, colorMap.cols, colorMap.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, colorMap.cols, colorMap.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    updateColorMap(colorMap,2,1,0);
}

Eigen::Matrix3f Keyframe::scaleIntrinsic(unsigned int layer) {
    Eigen::Matrix3f K = intrinsic;
    float w = float(width>>layer);
    K(0,0) *= w; K(0,1) *= w; K(0,2) *= w;
    K(1,0) *= w; K(1,1) *= w; K(1,2) *= w;
	K(2,0)  = 0; K(2,1)  = 0; K(2,2)  = 1;
    return K;
}


Eigen::Matrix3f Keyframe::scaleIntrinsicDepth(unsigned int layer) {
    Eigen::Matrix3f K = intrinsicDepth;
    float w = float(width>>layer);
    K(0,0) *= w; K(0,1) *= w; K(0,2) *= w;
    K(1,0) *= w; K(1,1) *= w; K(1,2) *= w;
    K(2,0)  = 0; K(2,1)  = 0; K(2,2)  = 1;
    return K;
}

float filter(float *ptr, Eigen::Vector3f &p, int w) {
    int xi = int(p(0));
    int yi = int(p(1));
    float fx = p(0)-xi;
    float fy = p(1)-yi;
    float *ptr0 = &ptr[xi+yi*w];
    return (1-fx)*(1-fy)*ptr0[0]+ fx*(1-fy)*ptr0[1]+(1-fx)*fy*ptr0[w]+ fx*fy*ptr0[w+1];
}

void normalizeCost(float *ptr, float *samples, int c, float gain) {
    for (int i = 0; i < c; i++) {
        if (samples[i]) {
            ptr[i] = MIN(gain*ptr[i]/samples[i],1.0f);
        }
    }
}

// bilinear filtering requires size-1
bool inBounds(Eigen::Vector3f &p, float w, float h) {
    return (p(0) >= 0.0f && p(0) < (w-2) && p(1) >= 0.0f && p(1) < (h-2));
}

// extracts matching points into modular format
void enumerateCorrespondencies(Keyframe *kA, Keyframe *kB, std::vector<Eigen::Vector2i> &matchingIndex) {
    if (kA == NULL || kB == NULL) return;

    Eigen::MatrixXf &mA        = kA->measurements;
    int cntA                   = kA->measurements.cols();
    Eigen::MatrixXf &mB        = kB->measurements;
    int cntB                   = kB->measurements.cols();

    if (cntA <= 0 || cntB <= 0) return;
    for (int i = 0; i < cntA; i++) {
       for (int j = 0; j < cntB; j++) {
           if (mA(7,i) == mB(7,j)) {
               Eigen::Vector2i correspondency(i,j);
               matchingIndex.push_back(correspondency);
               break;
           }
       }
   }
}
/*
void triangulateFeatures(Keyframe *kA, Keyframe *kB, std::vector<Eigen::Vector2i> &matchingIndex, std::vector<Eigen::Vector3f> &points3d)
{
    if (kA == NULL || kB == NULL) return;
    Eigen::MatrixXf &fA        = kA->features;
    Eigen::MatrixXf &fB        = kB->features;

    points3d.clear(); points3d.reserve(matchingIndex.size());

    Matrix4f A;
    for (size_t i = 0; i < matchingIndex.size(); i++)
    {
        int index0 = matchingIndex[i](0);
        int index1 = matchingIndex[i](1);
        float x1 = fA(0,index0);
        float y1 = fA(1,index0);
        float x2 = fB(0,index1);
        float y2 = fB(1,index1);
        Matrix4f &P1 = kA->pose[0];
        Matrix4f &P2 = kB->pose[0];

        A(0,0) = x1*P1(2,0) - P1(0,0);
        A(0,1) = x1*P1(2,1) - P1(0,1);
        A(0,2) = x1*P1(2,2) - P1(0,2);
        A(0,3) = x1*P1(2,3) - P1(0,3);

        A(1,0) = y1*P1(2,0) - P1(1,0);
        A(1,1) = y1*P1(2,1) - P1(1,1);
        A(1,2) = y1*P1(2,2) - P1(1,2);
        A(1,3) = y1*P1(2,3) - P1(1,3);

        A(2,0) = x2*P2(2,0) - P2(0,0);
        A(2,1) = x2*P2(2,1) - P2(0,1);
        A(2,2) = x2*P2(2,2) - P2(0,2);
        A(2,3) = x2*P2(2,3) - P2(0,3);

        A(3,0) = y2*P2(2,0) - P2(1,0);
        A(3,1) = y2*P2(2,1) - P2(1,1);
        A(3,2) = y2*P2(2,2) - P2(1,2);
        A(3,3) = y2*P2(2,3) - P2(1,3);

        JacobiSVD<Matrix4f> svd(A, ComputeFullV);
        Eigen::Vector4f p4 = svd.matrixV().col(3);
        Eigen::Vector3f p3(p4(0)/p4(3),p4(1)/p4(3),p4(2)/p4(3));
        points3d.push_back(p3);
    }
}
*/
void triangulateFeatures(Eigen::MatrixXf &fA, Eigen::Matrix4f &poseA, Eigen::MatrixXf &fB, Eigen::Matrix4f &poseB, std::vector<Eigen::Vector2i> &matchingIndex, std::vector<Eigen::Vector3f> &points3d)
{
    points3d.clear(); points3d.reserve(matchingIndex.size());

    Matrix4f A;
    for (size_t i = 0; i < matchingIndex.size(); i++)
    {
        int index0 = matchingIndex[i](0);
        int index1 = matchingIndex[i](1);
        float x1 = fA(0,index0);
        float y1 = fA(1,index0);
        float x2 = fB(0,index1);
        float y2 = fB(1,index1);
        Matrix4f &P1 = poseA;
        Matrix4f &P2 = poseB;

        A(0,0) = x1*P1(2,0) - P1(0,0);
        A(0,1) = x1*P1(2,1) - P1(0,1);
        A(0,2) = x1*P1(2,2) - P1(0,2);
        A(0,3) = x1*P1(2,3) - P1(0,3);

        A(1,0) = y1*P1(2,0) - P1(1,0);
        A(1,1) = y1*P1(2,1) - P1(1,1);
        A(1,2) = y1*P1(2,2) - P1(1,2);
        A(1,3) = y1*P1(2,3) - P1(1,3);

        A(2,0) = x2*P2(2,0) - P2(0,0);
        A(2,1) = x2*P2(2,1) - P2(0,1);
        A(2,2) = x2*P2(2,2) - P2(0,2);
        A(2,3) = x2*P2(2,3) - P2(0,3);

        A(3,0) = y2*P2(2,0) - P2(1,0);
        A(3,1) = y2*P2(2,1) - P2(1,1);
        A(3,2) = y2*P2(2,2) - P2(1,2);
        A(3,3) = y2*P2(2,3) - P2(1,3);

        JacobiSVD<Matrix4f> svd(A, ComputeFullV);
        Eigen::Vector4f p4 = svd.matrixV().col(3);
        Eigen::Vector3f p3(p4(0)/p4(3),p4(1)/p4(3),p4(2)/p4(3));
        points3d.push_back(p3);
    }
}

void Keyframe::promoteFeaturesToMeasurements() {
    // count valid indices:
    int validIndices = 0;
    for (size_t i = 0; i < features.cols(); i++) {
        if (features(7,i) >= 0) validIndices++;
    }
    // (re-)allocate measurements:
    measurements = Eigen::MatrixXf(8,validIndices);
    // pick valid measurements:
    int outIndex=0;
    for (size_t i = 0; i < features.cols(); i++) {
        if (features(7,i) < 0) continue;
        for (int j = 0; j < 8; j++) {
            measurements(j,outIndex) = features(j,i);
        }
        outIndex++;
    }
}
