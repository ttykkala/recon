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

#include <GL/glew.h>
//#include <opencv2/opencv.hpp>
#include <screenpoly.h>
#include <string.h>
#include <stdio.h>

ScreenPoly::ScreenPoly(int width, int height, int tileX, int tileY) {
    xsize = width;;
    ysize = height;
    xtile = tileX;
    ytile = tileY;
    mask = cv::Mat(height,width,CV_8UC1); memset(mask.ptr(),0,width*height);
    reset();
}

void ScreenPoly::reset() {
    pointIndex = 0;
    finished = false;

}
ScreenPoly::~ScreenPoly() {
}

void ScreenPoly::addPoint(float x, float y) {
    if (pointIndex < NUM_SCREEN_POLY_PTS && !finished) {
        //printf("tile : %d %d\n",int(x/xsize),int(y/ysize)); fflush(stdout);
        if (int(x/xsize)==xtile && int(y/ysize)==ytile) {
            int wx = int(x)%xsize;
            int wy = int(y)%ysize;
            pts[pointIndex*2+0] = wx;
            pts[pointIndex*2+1] = wy;//ysize-1-wy;
            pointIndex++;
        }
    }
}

void ScreenPoly::finish() {
    finished = true;

    // zero out mask:
    memset(mask.ptr(),0,xsize*ysize);

    // fill mask:
    cv::Point2i parr[NUM_SCREEN_POLY_PTS];
    for (int i = 0; i < pointIndex; i++) {
        parr[i].x = pts[i*2+0];
        parr[i].y = pts[i*2+1];
    }
    cv::fillConvexPoly(mask, &parr[0], pointIndex, cv::Scalar(255));
}

void ScreenPoly::draw() {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_LIGHTING);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glColor4f(1,0,0,0.5f);
    if (!finished) {
        if (pointIndex >= 4) {
            glBegin(GL_POLYGON);
            for (int i = 0; i < pointIndex; i++) glVertex3f(pts[i*2+0],ysize-1-pts[i*2+1],-1);
            glEnd();
        } else {
            glPointSize(5.0f);
            glBegin(GL_POINTS);
            for (int i = 0; i < pointIndex; i++) glVertex3f(pts[i*2+0],ysize-1-pts[i*2+1],-1);
            glEnd();
        }
    } else {
        unsigned char *ptr = (unsigned char*)mask.ptr();
        glPointSize(1.0f);
        glBegin(GL_POINTS);
        for (int j = 0; j < ysize; j++) {
            for (int i = 0; i < xsize; i++) {
                if (ptr[i+j*xsize] != 0) glVertex3f(i,ysize-1-j,-1);
            }
        }
        glEnd();
    }
    glDisable(GL_BLEND);
}

