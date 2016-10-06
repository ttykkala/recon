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
#include <tracker/phototracker.h>

typedef struct {
    TrackMode trackMode;
    int frame;
    float lambda;
} TRACK_PARAMS;

class TrackingParams {
public:
    TrackingParams();
    ~TrackingParams();
        void reset();
        void init(const char *fn);
        TRACK_PARAMS *getParams(int frameKey);
private:
    char fn[512];
    TRACK_PARAMS *params;
    TRACK_PARAMS defaultParams;
    int nParams;
    char *nextLine(char *ptr);
    int countLines(char *basePtr, int sz);
};
