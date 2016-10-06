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

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

template <class T>
__global__ void reduceProduct0(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduceProduct1(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduceProduct2(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduceProduct3(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;
    if (i + blockDim.x < n)
        mySum += g_idataA[i+blockDim.x]*g_idataB[i+blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T, unsigned int blockSize>
__global__ void reduceProduct4(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;
    if (i + blockSize < n)
        mySum += g_idataA[i+blockSize]*g_idataB[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T, unsigned int blockSize>
__global__ void reduceProduct5(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T mySum = (i < n) ? (g_idataA[i]*g_idataB[i]) : 0;
    if (i + blockSize < n)
        mySum += g_idataA[i+blockSize]*g_idataB[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceProduct6(T *g_idataA, T *g_idataB, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idataA[i]*g_idataB[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idataA[i+blockSize]*g_idataB[i+blockSize];
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void reduceProducts(int size, int threads, int blocks,
       int whichKernel, T *d_idataA, T *d_idataB, T *d_odata, cudaStream_t stream)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
    case 0:
        reduceProduct0<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size);
        break;
    case 1:
        reduceProduct1<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size);
        break;
    case 2:
        reduceProduct2<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size);
        break;
    case 3:
        reduceProduct3<T><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size);
        break;
    case 4:
        switch (threads)
        {
        case 1024:
            reduceProduct4<T, 1024><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 512:
            reduceProduct4<T, 512><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 256:
            reduceProduct4<T, 256><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 128:
            reduceProduct4<T, 128><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 64:
            reduceProduct4<T,  64><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 32:
            reduceProduct4<T,  32><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 16:
            reduceProduct4<T,  16><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  8:
            reduceProduct4<T,   8><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  4:
            reduceProduct4<T,   4><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  2:
            reduceProduct4<T,   2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  1:
            reduceProduct4<T,   1><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        }
        break;
    case 5:
        switch (threads)
        {
        case 1024:
            reduceProduct5<T, 1024><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 512:
            reduceProduct5<T, 512><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 256:
            reduceProduct5<T, 256><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 128:
            reduceProduct5<T, 128><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 64:
            reduceProduct5<T,  64><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 32:
            reduceProduct5<T,  32><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case 16:
            reduceProduct5<T,  16><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  8:
            reduceProduct5<T,   8><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  4:
            reduceProduct5<T,   4><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  2:
            reduceProduct5<T,   2><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        case  1:
            reduceProduct5<T,   1><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
        }
        break;
    case 6:
    default:
        if (isPow2(size))
        {
            switch (threads)
            {
            case 1024:
                reduceProduct6<T, 1024, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 512:
                reduceProduct6<T, 512, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 256:
                reduceProduct6<T, 256, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 128:
                reduceProduct6<T, 128, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 64:
                reduceProduct6<T,  64, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 32:
                reduceProduct6<T,  32, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 16:
                reduceProduct6<T,  16, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  8:
                reduceProduct6<T,   8, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  4:
                reduceProduct6<T,   4, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  2:
                reduceProduct6<T,   2, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  1:
                reduceProduct6<T,   1, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 1024:
                reduceProduct6<T, 1024, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 512:
                reduceProduct6<T, 512, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 256:
                reduceProduct6<T, 256, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 128:
                reduceProduct6<T, 128, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 64:
                reduceProduct6<T,  64, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 32:
                reduceProduct6<T,  32, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case 16:
                reduceProduct6<T,  16, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  8:
                reduceProduct6<T,   8, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  4:
                reduceProduct6<T,   4, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  2:
                reduceProduct6<T,   2, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            case  1:
                reduceProduct6<T,   1, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idataA, d_idataB, d_odata, size); break;
            }
        }
        break;
    }
}

template void
reduceProducts<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idataA, float *d_idataB, float *d_odata, cudaStream_t stream);
