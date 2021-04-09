#include "cuda_runtime.h"
#include <cstdio>
using namespace std;

__constant__ int C1[11];
__constant__ int C2[65];
__constant__ char C3[17];
__device__ int GlobalC1[7];
__device__ int GlobalC2[33];

texture<float, cudaTextureType1D, cudaReadModeElementType> texRef1d;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef2d;

surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;

typedef int(*FUNC) (int, int);

__device__ int f1(int i, int j)
{
    return i;
}

__device__ int f2(int i, int j)
{
    return i;
}

__device__ int f3(int i, int j)
{
    return i;
}

__device__ int f4(int i, int j)
{
    return i;
}

__device__ int f5(int i, int j)
{
    return i+j;
}

__device__ int f6(int i, int j)
{
    return i/j;
}

__device__ int f7(int i, int j)
{
    return i%j;
}

__device__ FUNC flist[7] = {f1, f2, f3, f4, f5, f6, f7};

__global__ void test(const float4 v0, float4* v)
{
    int tid = threadIdx.x;
    float4 vv = v[tid];
    for(int i=0; i<16; i++)
    {
        vv.x = fmaf(vv.x, v0.x, vv.x);
        vv.y = fmaf(vv.y, v0.y, vv.y);
        vv.z = fmaf(vv.z, v0.z, vv.z);
        vv.w = fmaf(vv.w, v0.w, vv.w);
    }

    float v1 = (vv.x * vv.y + vv.z);
    float v2 = (vv.x + vv.z);
    
    vv.w += v1*v2;
    v[tid] = vv;
}

__global__ void child(int* v, int VAL)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    v[idx] *= VAL + GlobalC1[idx%7]+ GlobalC2[idx%16];
    v[idx] += C1[VAL] + C2[VAL];
}

__global__ void simpletest(const int4 VAL, int* v)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int a = v[idx]*VAL.x + GlobalC1[idx%16];
    //__shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    a = __shfl_up_sync(0xffffffff, a, 1);
    if (VAL.z > 0)
        a += C1[VAL.y];
    v[idx] = a;
}

__global__ void argtest(int ArgC[8], int* a, int* b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int v = flist[ArgC[0]](idx, a[idx]);

    float tu = float(v)+0.5f;
    float tv = tu - 1.0f;

    a[idx] = sinf(v*0.35f) + tex2D(texRef2d, tu, tv) + tex1D(texRef1d, tu);

    float data;
    surf2Dread(&data,  inputSurfRef, idx * 4, idx);
    surf2Dwrite(data, outputSurfRef, idx * 4, idx);

    if (ArgC[1]==idx)
        printf("a[%d] = %d\n", idx, a[idx]);
}

__global__ void shared_test(float c, float* x)
{
    __shared__ float ShMem_s[1025];
    extern __shared__ float ShMem_d[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ShMem_s[threadIdx.x] = (x[idx]>c) ? x[idx] : c;
    ShMem_d[idx%2048] = x[idx]*c;
    __syncthreads();

    x[idx] = ShMem_s[1024 - threadIdx.x] + ShMem_d[2048-idx];
}

__global__ void local_test(int c, int d, int* x)
{
    int LocalV[17];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    LocalV[c+1] = idx+1;
    LocalV[c] = idx;
    LocalV[2*c] = 2*idx;

    x[idx] = LocalV[d] + LocalV[d+1];
}

__global__ void nvinfo_test(int c, int d, int* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    x[idx] += blockIdx.x + blockIdx.y + blockIdx.z;
}

int main()
{
    return 0;
}