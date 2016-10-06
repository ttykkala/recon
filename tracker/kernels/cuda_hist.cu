/*
 Copyright Ramtin Shams (hereafter referred to as 'the author'). All rights 
 reserved. **Citation required in derived works or publications** 
 
 NOTICE TO USER:   
 
 Users and possessors of this source code are hereby granted a nonexclusive, 
 royalty-free license to use this source code for non-commercial purposes only, 
 as long as the author is appropriately acknowledged by inclusion of this 
 notice in derived works and citation of appropriate publication(s) listed 
 at the end of this notice in any derived works or publications that use 
 or have benefited from this source code in its entirety or in part.
   
 
 THE AUTHOR MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 IMPLIED WARRANTY OF ANY KIND.  THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
 REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 OR PERFORMANCE OF THIS SOURCE CODE.  
 
 Relevant publication(s):
	@inproceedings{Shams_ICSPCS_2007,
		author        = "R. Shams and R. A. Kennedy",
		title         = "Efficient Histogram Algorithms for {NVIDIA} {CUDA} Compatible Devices",
		booktitle     = "Proc. Int. Conf. on Signal Processing and Communications Systems ({ICSPCS})",
		address       = "Gold Coast, Australia",
		month         = dec,
		year          = "2007",
		pages         = "418-422",
	}

	@inproceedings{Shams_DICTA_2007a,
		author        = "R. Shams and N. Barnes",
		title         = "Speeding up Mutual Information Computation Using {NVIDIA} {CUDA} Hardware",
		booktitle     = "Proc. Digital Image Computing: Techniques and Applications ({DICTA})",
		address       = "Adelaide, Australia",
		month         = dec,
		year          = "2007",
		pages         = "555-560",
		doi           = "10.1109/DICTA.2007.4426846",
	};
*/

// includes, system
#include <stdlib.h>
//#include <tchar.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil.h>
//#include "cuda_basics.h"
//#include "cuda_hist.h"
#include <helper_cuda.h>
#include <cwchar>
// includes, kernels
#include "gpu_hist.cu"

struct cudaHistOptions
{
    int threads, blocks;
};


//Round a / b to the nearest higher integer value
extern "C" int iDivUp2(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a to the nearest multiple of b
extern "C" int iRoundUp(int a, int b)
{
    return iDivUp2(a, b) * b;
}

__global__ void gpuSumGlobalMem(float *src, float *dst, int num, int len)
{
    int data_per_block = ceil((float) len / gridDim.x);
    int data_per_thread = ceil((float) data_per_block / blockDim.x);
    int baseBlockIdx = IMUL(blockIdx.x, data_per_block);
    int baseIdx = IMUL(threadIdx.x, data_per_thread) + baseBlockIdx;
    int end = min(baseIdx + data_per_thread, baseBlockIdx + data_per_block);		//Clamp to block boundary
    end = min(end, len);					//Clamp to len

    for (int i = baseIdx; i < end; i++)
    {
        float sum = 0;
        for (int j = 0; j < num; j++)
            sum += src[IMUL(len, j) + i];
        dst[i] = sum;
    }
}

extern "C" double cudaHista(float *src, float *hist, int length, int bins,float *d_hist, cudaStream_t stream)
{
	dim3 grid, block;
    float *d_src;
	double time = 0;
//	unsigned int hTimer;
	cudaHistOptions options;
    d_src = src;				//Do not copy hist!

    options.threads = 160;
    options.blocks = 64;

	//Perform sanity checks
	if (options.threads > MAX_THREADS)
		printf("'threads' exceed the maximum."), exit(1);
	if (options.threads % WARP_SIZE != 0)
		printf("'threads' must be a multiple of the WARP_SIZE."), exit(1);
	if (options.blocks > MAX_BLOCKS_PER_DIM)
		printf("'blocks' exceed the maximum."), exit(1);

	//Prepare the execution configuration
	int warps = options.threads / WARP_SIZE;
	int max_bins = MAX_USABLE_SHARED / sizeof(unsigned int) / warps;
	block.x = WARP_SIZE; block.y = warps; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int shared_mem_size = max_bins * warps * sizeof(unsigned int);
	if (shared_mem_size> MAX_USABLE_SHARED)
		printf("Maximum shared memory exceeded."), exit(1);

	//Initialize histogram memory
    cudaMemsetAsync(d_hist, 0, options.blocks * bins * sizeof(float),stream);

    int calls = iDivUp2(bins, max_bins);
    for (int i = 0; i < calls; i++) {
        gpuHista<<<grid, block, shared_mem_size,stream>>>(d_src, d_hist + max_bins * i, length, bins, min(max_bins, bins - max_bins * i), max_bins * i);
        getLastCudaError("gpuHista() execution failed\n");
	}

	//Sum up the histograms
	int numHists = grid.x;
    if (numHists > 1) {
		block.x = MAX_THREADS; block.y = 1; block.z = 1;
		grid.x = ceil((float) bins / block.x); grid.y = 1; grid.z = 1;

        gpuSumGlobalMem<<<grid, block,0,stream>>>(d_hist, d_hist, numHists, bins);
        getLastCudaError("gpuSumGlobalMem() execution failed\n");
	}
  //  cudaThreadSynchronize();
    cudaMemcpyAsync(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToDevice,stream);

	return time;
}

__global__ void gpuZeroMem(float *mem, int len)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int threads = blockDim.x;
    const unsigned int bid = blockIdx.x + IMUL(blockIdx.y, gridDim.x);

    int g_ofs = IMUL(bid, threads) + tid;
    if (g_ofs < len)
        mem[g_ofs] = 0.0f;
}

extern "C" void cudaZeroMem(float *d_mem, int length)
{
    dim3 grid, block;
//	TIMER_CREATE;

    const int max_threads = MAX_THREADS;
    int good_len = iRoundUp(length, WARP_SIZE);

    block.x = max_threads; block.y = 1; block.z = 1;
    int blocks = iDivUp2(good_len, max_threads);
    if (blocks > MAX_BLOCKS_PER_DIM)
    {
        grid.x = ceil(sqrtf(blocks)); grid.y = grid.x; grid.z = 1;
    }
    else
    {
        grid.x = blocks; grid.y = 1; grid.z = 1;
    }

    //TIMER_START;
    gpuZeroMem<<<grid, block>>>(d_mem, length);
//	CUT_CHECK_ERROR("gpuZeroMem() execution failed\n");
//	TIMER_PRINT("gpuZeroMem", length)
//	TIMER_DELETE;
}


extern "C" double cudaHistb(float *src, float *hist, int length, int bins)
{
	dim3 grid, block;
//	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
//	unsigned int hTimer;
//    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
//	TIMER_CREATE;

    d_src = src; d_hist = hist;

	cudaHistOptions options;
    options.threads = 128;
    options.blocks = 8;

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(block.x)) + ceilf(log2f(grid.x))); 
	int hist_len = cell_len * bins;

    cudaThreadSynchronize();

    cudaMalloc((void**) &d_interim, hist_len * sizeof(float));

	cudaZeroMem(d_interim, hist_len);				//This much faster than cudaMemset

    //TIMER_START;
	int shared_len_pt = GPUHIST_SHARED_LEN >> (int) ceil(log2f(options.threads));							//Length of shared memory available to each thread (in int32)
	int n = (shared_len_pt << 5) / bins;
	const int bits_pbin = n > 0 ? min((1 << (int)log2f(n)), 32) : 0;					 			//Number of bits per bin per thread 0, 1, 2, 4, 8, 16, 32	

    gpuHistb<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin);
    getLastCudaError("gpuHistb execution failed\n");

	//Reduce the interim histogram 
	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduceHist requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

    //TIMER_START;
    gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
    getLastCudaError("gpuReduceHist execution failed\n");

//    cudaThreadSynchronize();
    //CUT_SAFE_CALL(cutStopTimer(hTimer));
    //time = cutGetTimerValue(hTimer);
    //CUT_SAFE_CALL(cutDeleteTimer(hTimer));

    //TIMER_START;
    cudaFree(d_interim);

//	TIMER_PRINT("Storing data", 0);
//	TIMER_DELETE;

	return time;
}
/*
extern "C" double cudaHistc(float *src, float *hist, int length, int bins, cudaHistOptions *p_options, bool device )
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0)
	}
	else
	{
		d_src = src; d_hist = hist;				
	}

	cudaHistOptions options;
	if (p_options)
		options = *p_options;
	else
	{
		options.threads = 128;
		options.blocks = 8;
	}

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(block.x)) + ceilf(log2f(grid.x))); 
	int hist_len = cell_len * bins;

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_interim, hist_len * sizeof(float)));
	TIMER_PRINT("Allocating memory", 0);

	cudaZeroMem(d_interim, hist_len);				//This much faster than cudaMemset

	TIMER_START;
	int shared_len_pw = GPUHIST_SHARED_LEN >> (int) ceil(log2f(options.threads >> LOG2_WARP_SIZE));	//Length of shared memory available to each warp (in int32)
	int bits_pbin = max(min((shared_len_pw * 27) / bins, 27), 0);
	for (int i = 1; i <= 28; i++)
	{
		if (bits_pbin >= 27 / i)
		{
			bits_pbin = 27 / i;
			break;
		}
	}
#ifdef VERBOSE
	printf("bits per bin: %d\n", bits_pbin);
#endif
	gpuHistc<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin);
	CUT_CHECK_ERROR("gpuHistc() execution failed\n");
	TIMER_PRINT("gpuHistc", length);


	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduce requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);


	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	CUDA_SAFE_CALL(cudaFree(d_interim));
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_hist));
	}
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}

extern "C" double cudaHist_Approx(float *src, float *hist, int length, int bins, cudaHistOptions *p_options, bool device)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src, *d_hist, *d_interim;
	double time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	if (!device)
	{
		TIMER_START;
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, bins * sizeof(float)));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
		TIMER_PRINT("Loading data", 0)
	}
	else
	{
		d_src = src; d_hist = hist;				
	}

	cudaHistOptions options;
	if (p_options)
		options = *p_options;
	else
	{
		options.threads = 256;
		options.blocks = 16;
	}

	//Prepare execution configuration
	block.x = options.threads; block.y = 1; block.z = 1;
	grid.x = options.blocks; grid.y = 1; grid.z = 1;

	int cell_len = powf(2.0f, ceilf(log2f(grid.x)));
	int hist_len = cell_len * bins;

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	TIMER_START;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_interim, hist_len * sizeof(float)));
	TIMER_PRINT("Allocating memory", 0);

	cudaZeroMem(d_interim, hist_len);				//This is much faster than cudaMemset

	TIMER_START;
	int n = (GPUHIST_SHARED_LEN << 5) / bins;
	const int bits_pbin = n > 0 ? min((1 << (int)log2f(n)), 32) : 0;					 			//Number of bits per bin per thread 0, 1, 2, 4, 8, 16, 32	
#ifdef VERBOSE
	printf("bits per bin: %d\n", bits_pbin);
#endif
	gpuHist_Approx<<<grid, block>>>(d_src, d_interim, length, bins, bits_pbin, (float) bins / options.threads);
	CUT_CHECK_ERROR("gpuHist_Approx() execution failed\n");
	TIMER_PRINT("gpuHist_Approx", length);

	//Reduce the interim histogram 
	if (bins > MAX_BLOCKS_PER_DIM)						//We want the bins to fit in horizontal grid.x
		printf("Maximimum bins exceeded."), exit(1);

	block.x = max(min(cell_len, MAX_THREADS), 64); block.y = 1; block.z = 1;			//gpuReduceHist requires at least 64 threads
	grid.x = bins; grid.y = 1; grid.z = 1;

	TIMER_START;
	gpuReduceHist<<<grid, block, block.x * sizeof(float)>>>(d_interim, d_hist, cell_len);
	CUT_CHECK_ERROR("gpuReduceHist() execution failed\n");
	TIMER_PRINT("gpuReduceHist", hist_len);

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	TIMER_START;
	CUDA_SAFE_CALL(cudaFree(d_interim));
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMemcpy(hist, d_hist, bins * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaFree(d_src));
		CUDA_SAFE_CALL(cudaFree(d_hist));
	}
	TIMER_PRINT("Storing data", 0);
	TIMER_DELETE;

	return time;
}
*/
