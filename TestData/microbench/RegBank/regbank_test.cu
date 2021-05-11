#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cuptr.hpp"
#include "cuda.h"

#include <cstdio>
#include <iostream>
using namespace std;

typedef unsigned int uint;

#define GRD_SIZE (1600)
#define BLK_SIZE (128)
#define WARP_CUT (4)
#define N_ITER (256)
#define N_UNROLL (256)
#define N_WARMUP (3)
#define N_TEST (5)

__global__ void regbank_test_kernel(const int2 c, const int NIter, const float4 v, float* a)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int wid = tid / 32;
    int lid = tid % 32;

    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;

    if (wid<c.x)
    {
        for(int i=0; i<NIter; i++)
        {
            #pragma unroll
            for(int n=0; n<N_UNROLL; n++)
                v0 = v0+v1;
        }
    }
    else{
        for(int i=0; i<NIter; i++)
        {
            #pragma unroll
            for(int n=0; n<N_UNROLL; n++)
                v0 = fmaf(v0, v2, v1);
        }
    }

    __syncthreads();
    
    // only first lane of warp in first block writes to memory
    if( bid ==0 && lid==0)
        a[wid] = v0;
}

float regbank_test_run(const int2 c, const int NIter, const float4 v, float* a, cudaEvent_t &event_start, cudaEvent_t &event_stop)
{
    float elapsedTime;
    checkCudaErrors(cudaEventRecord(event_start, 0));

    regbank_test_kernel<<<GRD_SIZE, BLK_SIZE>>>(c, NIter, v, a);

    checkCudaErrors(cudaEventRecord(event_stop, 0));
    checkCudaErrors(cudaEventSynchronize(event_stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));

    return elapsedTime;
}

float regbank_test_run_drv(const int2 c, const int NIter, const float4 v, float* a, cudaEvent_t &event_start, cudaEvent_t &event_stop)
{
    static CUmodule cuModule;
    static CUfunction kernel;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        cuInit(0);

        // Create module from binary file
        cuModuleLoad(&cuModule, "regbank_test.rep.sm_50.cubin");

        // Get function handle from module
        cuModuleGetFunction(&kernel, cuModule, "_Z19regbank_test_kernel4int2i6float4Pf");

        printf("cuModule = %#llx\n", (unsigned long long)cuModule);
        printf("cuFunction = %#llx\n", (unsigned long long)kernel);
        isInitialized = true;
    }

    float elapsedTime;
    checkCudaErrors(cudaEventRecord(event_start, 0));

    void* args[] = { (void*)&c, (void*)&NIter, (void*)&v, (void*)&a };
    cuLaunchKernel(kernel,
                   GRD_SIZE, 1, 1, BLK_SIZE, 1, 1,
                    0, 0, args, 0);
    //regbank_test_kernel<<<GRD_SIZE, BLK_SIZE>>>(c, NIter, v, a);

    checkCudaErrors(cudaEventRecord(event_stop, 0));
    checkCudaErrors(cudaEventSynchronize(event_stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));

    return elapsedTime;
}

void dotest()
{
    CuPtr<float> da(4096);
    int NIter = N_ITER;

    cudaEvent_t event_start, event_stop;
    checkCudaErrors(cudaEventCreate(&event_start));
    checkCudaErrors(cudaEventCreate(&event_stop));

    //double giga_instr = (1e-9 * GRD_SIZE) * BLK_SIZE * N_ITER * N_UNROLL;

    int2 c = make_int2(WARP_CUT, 0);
    float4 v = make_float4(0, 1.0f, 0, 0);

    printf("### Warm up...\n");
    for(int i=0; i<N_WARMUP; i++)
    {
        da.SetZeros();
        float elapsedTime = regbank_test_run_drv(c, NIter, v, da.GetPtr(), event_start, event_stop); // in ms
        printf("  Warmup %2d: %10.3f ms\n", i, elapsedTime);
    }
    
    printf("### Testing...\n");
    for(int i=0; i<N_TEST; i++)
    {
        da.SetZeros();
        float elapsedTime = regbank_test_run_drv(c, NIter, v, da.GetPtr(), event_start, event_stop); // in ms
        printf("  Test %2d: %10.3f ms\n", i, elapsedTime);
    }

    checkCudaErrors(cudaEventDestroy(event_start));
    checkCudaErrors(cudaEventDestroy(event_stop));

    printf("\n### Result checking...\n");

    HostPtr<float> ha;
    da.ToHostPtr(ha);
    for(int i=0; i<BLK_SIZE/32; i++)
    {
        unsigned int xa = *(unsigned int *)(&ha(i));
        printf("res[%2d] : %8g  0x%08x\n", i, ha(i), xa);
    }
}

int main()
{
    dotest();
    return 0;
}