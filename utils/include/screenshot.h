/*
Copyright 2016 Tommi M. Tykk�l�

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

class ScreenShot {
public:
    ScreenShot(const char *basepath, int x0, int y0, int width, int height);
    ~ScreenShot();
    void save();
private:
    int shotIndex;
    char basePath[512];
    unsigned char *screenShot;
    unsigned char *flippedImage;
    int x0,y0;
    int width,height;
    void flip(unsigned char *srcData, unsigned char *dstData);
    void writePPM(const char *fn, int dimx, int dimy, unsigned char *src);
//    cv::Mat flippedImage;
//    cv::Mat screenShot;
};
