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
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <calib/calib.h>

// todo: support for LIDAR RGB distortion model! (6-parameters)
// todo: support for tangential distortion!
void radialDistort(Eigen::Vector2f &pu, float *kc, Eigen::Matrix3f &K, Eigen::Vector2f &pd);
void planeRegression(std::vector<Eigen::Vector3f> &points, Eigen::Vector3f &mean, Eigen::Vector3f &uVec, Eigen::Vector3f &vVec);
void setupBaseline(Calibration &calib,Eigen::Matrix4f &baseline);
void setupNormalizedIntrinsics(Calibration &calib, Eigen::Matrix3f &intrinsic, Eigen::Matrix3f &intrinsicDepth, float *distcoeffRGB);
void setupIntrinsics(Calibration &calib, int width, int depthWidth, Eigen::Matrix3f &intrinsic, Eigen::Matrix3f &intrinsicDepth, float *distcoeffRGB);

