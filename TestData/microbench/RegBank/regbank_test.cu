#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cuptr.hpp"

#include <cstdio>
#include <iostream>
using namespace std;

typedef unsigned int uint;


#define NUM (256)

__device__ float f1(float a, float b, float c)
{
    float r = fma(a, b, c);

    return r;    
}

__device__ float f2(float a, float b, float c)
{
    float r = a+b+c;
    return r;
}

__global__ void regbank_test_kernel(const int2 c, const int NIter, float* a)
{
    int tid = threadIdx.x;
    int wid = tid / 32;

    float v0 = a[tid];
    float v1 = a[tid+1];
    float v2 = a[tid+2];

    if (wid<c.x)
    {
        for(int i=0; i<NIter; i++)
        {
            for(int n=0; n<NUM; n++)
                v0 = f1(v0, v1, v2);
        }
    }
    else{
        for(int i=0; i<NIter; i++)
        {
            for(int n=0; n<NUM; n++)
                v0 = f2(v0, v1, v2);
        }
    }

    if( v0 > 1e38)
        a[tid] = v0;
}

float regbank_test_run(const int2 c, const int NIter, float* a, cudaEvent_t &event_start, cudaEvent_t &event_stop)
{
    float elapsedTime;
    checkCudaErrors(cudaEventRecord(event_start, 0));

    regbank_test_kernel<<<400, 256>>>(c, NIter, a);

    checkCudaErrors(cudaEventRecord(event_stop, 0));
    checkCudaErrors(cudaEventSynchronize(event_stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));

    return elapsedTime;
}

void dotest()
{
    CuPtr<float> da(4096);
    int NIter = 128;

    cudaEvent_t event_start, event_stop;
    checkCudaErrors(cudaEventCreate(&event_start));
    checkCudaErrors(cudaEventCreate(&event_stop));

    printf("### Warm up...\n");
    for(int i=0; i<3; i++)
    {
        int2 c = make_int2(4, 0);
        float elapsedTime = regbank_test_run(c, NIter, da.GetPtr(), event_start, event_stop);
        printf("  Warmup %2d: %10.3f ms\n", i, elapsedTime);
    }
    
    printf("### Testing...\n");
    for(int i=0; i<8; i++)
    {
        int2 c = make_int2(4, 0);
        float elapsedTime = regbank_test_run(c, NIter, da.GetPtr(), event_start, event_stop);
        printf("  Test %2d: %10.3f ms\n", i, elapsedTime);
    }

    checkCudaErrors(cudaEventDestroy(event_start));
    checkCudaErrors(cudaEventDestroy(event_stop));

}

int main()
{
    dotest();
    return 0;
}