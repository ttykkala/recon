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
#include <screenshot.h>
#include <string.h>
#include <stdio.h>

ScreenShot::ScreenShot(const char *basepath, int x0, int y0, int width, int height) {
    shotIndex = 0;
    strncpy(&basePath[0],basepath,512);
    screenShot = new unsigned char[width*height*4];
    flippedImage = new unsigned char[width*height*3];
    this->x0 = x0; this->y0 = y0;
    this->width = width; this->height = height;
}
ScreenShot::~ScreenShot() {
    delete[] screenShot;
    delete[] flippedImage;
}

void ScreenShot::writePPM(const char *fn, int dimx, int dimy, unsigned char *src) {
    FILE *fp = fopen(fn, "wb"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    fwrite(src,dimx*dimy*3,1,fp);
    fclose(fp);
}

void ScreenShot::flip(unsigned char *srcData, unsigned char *dstData) {
    int pitch = width*3;
    unsigned char *dstRow = &dstData[(height-1)*pitch];
    unsigned char *srcRow = &srcData[0];
    for (int j = 0; j < height; j++) {
        unsigned char *dstRow = &dstData[(height-1-j)*pitch];
        unsigned char *srcRow = &srcData[j*pitch];
        memcpy(dstRow,srcRow,pitch);
    }
}

void ScreenShot::save() {
    char buf[1024];
    sprintf(buf,"%s/screenshot%04d.ppm",&basePath[0],shotIndex);
    printf("saving %s...\n",buf);
    glReadPixels(x0,y0,width,height,GL_RGB,GL_UNSIGNED_BYTE,screenShot);
    flip(screenShot,flippedImage);
    writePPM(buf,width,height,&flippedImage[0]);
    //imwrite(buf,flippedImage);
    shotIndex++;
}

