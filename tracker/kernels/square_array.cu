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

#include "hostUtils.h"
#include <cwchar>

// Kernel that executes on the CUDA device
__global__ void square_array( float *a, int N )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < N ) a[idx] = a[idx] * a[idx];
}


extern "C" void launch_cudaProcess(int n_blocks, int block_size, float *a, int N)
{
	square_array <<< n_blocks, block_size >>> ( a, N );
	checkCudaError("launcheri");
}
