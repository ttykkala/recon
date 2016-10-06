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

//#include <GL/glew.h>
//#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
//#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
//#include <rendercheck_gl.h>
struct cudaDeviceProp;

void checkCudaError(const char *message);
void printFreeDeviceMemory();
//void printDevProp( cudaDeviceProp devProp );
void cudaTest();
