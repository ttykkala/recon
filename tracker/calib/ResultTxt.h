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

#define MAX_POSE_COUNT 10000

class ResultTxt {
public:
	ResultTxt();
	~ResultTxt();
        void reset(bool addIdentityFlag=false);
        void init(const char *fn,bool addIdentityFlag=false);
        void save(int transpose=0, int inverse=0);
        void addPose(float *m4x4);
        void canonize();
private:
    int nRows;
    float *data;
    char fn[512];
};
