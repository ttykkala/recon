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

#include <opencv2/opencv.hpp>

const int NUM_SCREEN_POLY_PTS = 16;

class ScreenPoly {
public:
    ScreenPoly(int width, int height, int xtile, int ytile);
    ~ScreenPoly();
    void reset();
    void addPoint(float x, float y);
    void draw();
    void finish();
    cv::Mat &getMask() { return mask; }
    bool finished;
private:
    float pts[NUM_SCREEN_POLY_PTS*2];
    int pointIndex;
    int xsize;
    int ysize;
    int xtile;
    int ytile;
    cv::Mat mask;

};
