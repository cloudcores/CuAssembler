#include "cuda_runtime.h"

__global__ void func1(int* a)
{
    int idx = threadIdx.x;
    int v;
    if (idx>32)
        v = idx;
    else
        v = 32;
    
    a[idx] = v;
}

__global__ void func2(int* a)
{
    int idx = threadIdx.x;
    int v = a[idx]*2;
    if (idx>32)
        v += 1;
    
    a[idx] = v;
}

__global__ void func3(int* a)
{
    int idx = threadIdx.x;
    int v;
    if (idx>32)
    {
        v = a[idx] + 1;
    }
    else{
        v = a[idx+1024] * 2;
    }

    a[idx+2048] = v;
}

/*
__global__ void func(int4 c, float* a)
{
    __shared__ float S[4096];
    unsigned int tix = threadIdx.x;
    unsigned int bix = blockIdx.x;
    unsigned int wid = tix/warpSize;
    S[tix] = a[tix];
    __syncthreads();

    if(wid==0)
    {
        float v0=0.14f;
        float v1=1.14f;
        float v2=2.14f;
        float v3=3.14f;

        float va = S[1024-tix];

        #pragma unroll 256
        for(int i=0; i<4096; i++)
        {
            v0 = fmaf(v0, va, v0);
            v1 = fmaf(v1, va, v1);
            v2 = fmaf(v2, va, v2);
            v3 = fmaf(v3, va, v3);
        }

        if(bix==0 && tix==0) a[0] = v0+v1+v2+v3;
    }
    else{
        
    }
}

__global__ void switchTest(int* a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int v = idx %8;
    switch(v){
        case 0:
            a[idx] = idx;
            break;
        case 1:
            a[idx] = idx * v;
            break;
        case 2:
            a[idx] = idx + v;
            break;
        case 3:
            a[idx] = idx - v;
            break;
        case 4:
        case 5:
            a[idx] = idx + 2*v;
            break;
        case 6:
        case 7:
            a[idx] = idx + v*v;
            break;
        default:
            break;

    }
}*/

int main()
{
    return 0;
}