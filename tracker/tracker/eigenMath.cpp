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


#include <tracker/eigenMath.h>
#include <calib/calib.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void setupBaseline(Calibration &calib,Eigen::Matrix4f &baseline) {
    double *TLR = calib.getTLR();
    baseline(0,0) = TLR[0];  baseline(0,1) = TLR[1];  baseline(0,2) = TLR[2];  baseline(0,3) = TLR[3];
    baseline(1,0) = TLR[4];  baseline(1,1) = TLR[5];  baseline(1,2) = TLR[6];  baseline(1,3) = TLR[7];
    baseline(2,0) = TLR[8];  baseline(2,1) = TLR[9];  baseline(2,2) = TLR[10]; baseline(2,3) = TLR[11];
    baseline(3,0) = TLR[12]; baseline(3,1) = TLR[13]; baseline(3,2) = TLR[14]; baseline(3,3) = TLR[15];
}

void setupNormalizedIntrinsics(Calibration &calib, Eigen::Matrix3f &intrinsic, Eigen::Matrix3f &intrinsicDepth, float *distcoeffRGB) {
    double *K = calib.getKR();
    double *KIR = calib.getKL();

    intrinsic(0,0) = fabs(K[0]);  intrinsic(0,1) = fabs(K[1]); intrinsic(0,2) = fabs(K[2]);
    intrinsic(1,0) = fabs(K[3]);  intrinsic(1,1) = fabs(K[4]); intrinsic(1,2) = fabs(K[5]);
    intrinsic(2,0) = fabs(K[6]);  intrinsic(2,1) = fabs(K[7]); intrinsic(2,2) = fabs(K[8]);

    intrinsicDepth(0,0) = fabs(KIR[0]);  intrinsicDepth(0,1) = fabs(KIR[1]); intrinsicDepth(0,2) = fabs(KIR[2]);
    intrinsicDepth(1,0) = fabs(KIR[3]);  intrinsicDepth(1,1) = fabs(KIR[4]); intrinsicDepth(1,2) = fabs(KIR[5]);
    intrinsicDepth(2,0) = fabs(KIR[6]);  intrinsicDepth(2,1) = fabs(KIR[7]); intrinsicDepth(2,2) = fabs(KIR[8]);

    double *kc = calib.getKcR();
    for (int i = 0; i < 5; i++) distcoeffRGB[i] = kc[i];
}


void setupIntrinsics(Calibration &calib, int width, int depthWidth, Eigen::Matrix3f &intrinsic, Eigen::Matrix3f &intrinsicDepth, float *distcoeffRGB) {
    double *K = calib.getKR();
    double *KIR = calib.getKL();

    intrinsic(0,0) = fabs(K[0])*width;  intrinsic(0,1) = fabs(K[1])*width; intrinsic(0,2) = fabs(K[2])*width;
    intrinsic(1,0) = fabs(K[3])*width;  intrinsic(1,1) = fabs(K[4])*width; intrinsic(1,2) = fabs(K[5])*width;
    intrinsic(2,0) = 0;  intrinsic(2,1) = 0; intrinsic(2,2) = 1;

    intrinsicDepth(0,0) = fabs(KIR[0])*depthWidth;  intrinsicDepth(0,1) = fabs(KIR[1])*depthWidth; intrinsicDepth(0,2) = fabs(KIR[2])*depthWidth;
    intrinsicDepth(1,0) = fabs(KIR[3])*depthWidth;  intrinsicDepth(1,1) = fabs(KIR[4])*depthWidth; intrinsicDepth(1,2) = fabs(KIR[5])*depthWidth;
    intrinsicDepth(2,0) = 0;  intrinsicDepth(2,1) = 0; intrinsicDepth(2,2) = 1;

    double *kc = calib.getKcR();
    for (int i = 0; i < 8; i++) distcoeffRGB[i] = kc[i];
}


void radialDistort(Eigen::Vector2f &pu, float *kc, Eigen::Matrix3f &K, Eigen::Vector2f &pd)
{
    // distort point
    float r2,r4,r6;
    float radialDist;
    float dx;
    float dy;

    // generate r2 components
	dx  = pu(0)*pu(0); dy  = pu(1)*pu(1);
    // generate distorted coordinates
    r2 = dx+dy; r4 = r2*r2; r6 = r4 * r2;
	radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
	pd(0) = K(0,0)*pu(0)*radialDist+K(0,2);
	pd(1) = K(1,1)*pu(1)*radialDist+K(1,2);
}

void planeRegression(std::vector<Eigen::Vector3f> &points, Eigen::Vector3f &mean, Eigen::Vector3f &uVec, Eigen::Vector3f &vVec)
{
    mean(0) = 0;
    mean(1) = 0;
    mean(2) = 0;
    for (int i = 0; i < points.size(); i++) {
        mean(0) += points[i](0);
        mean(1) += points[i](1);
        mean(2) += points[i](2);
    }
    mean(0) /= float(points.size());
    mean(1) /= float(points.size());
    mean(2) /= float(points.size());

    float mtx[9] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < points.size(); i++) {
        float n[3] = {0,0,0};
        n[0] = points[i](0)-mean(0);
        n[1] = points[i](1)-mean(1);
        n[2] = points[i](2)-mean(2);
        mtx[0] += n[0]*n[0]; mtx[1] += n[0]*n[1]; mtx[2] += n[0]*n[2];
        mtx[3] += n[0]*n[1]; mtx[4] += n[1]*n[1]; mtx[5] += n[1]*n[2];
        mtx[6] += n[0]*n[2]; mtx[7] += n[1]*n[2]; mtx[8] += n[2]*n[2];
    }
    cv::Mat E, V;
    cv::Mat M(3,3,CV_32FC1,mtx);
    cv::eigen(M,E,V);
    uVec(0) = V.at<float>(0,0);
    uVec(1) = V.at<float>(0,1);
    uVec(2) = V.at<float>(0,2);
    vVec(0) = V.at<float>(1,0);
    vVec(1) = V.at<float>(1,1);
    vVec(2) = V.at<float>(1,2);
    uVec.normalize();
    vVec.normalize();
}
