#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cuptr.hpp"

#include <cstdio>
#include <iostream>
using namespace std;

template<typename DType>
__global__ void l2bank_test_kernel(const int2 c, const int NIter, DType* a)
{
    int tid = threadIdx.x;
    int warpid = tid>>5;

    DType* p;
    if (warpid%2==0) 
        p = a + c.x;// + (tid&31);
    else
        p = a + c.y;// + (tid&31);

    DType val(0);

    for(int iter=0; iter<NIter; iter++)
    {
        #pragma unroll
        for(int n=0; n<16; n++)
        {
            // val += *p;
            atomicAdd(p, (DType)1);
        }    
            //atomicAdd(p, (DType)1);
    }

    if (val>1e12)
        *p = val;
}

template<typename DType>
float l2bank_test_run(const int2 c, const int NIter, DType* a, cudaEvent_t &event_start, cudaEvent_t &event_stop)
{
    float elapsedTime;
    checkCudaErrors(cudaEventRecord(event_start, 0));

    l2bank_test_kernel<DType><<<40, 128>>>(c, NIter, a);

    checkCudaErrors(cudaEventRecord(event_stop, 0));
    checkCudaErrors(cudaEventSynchronize(event_stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));

    return elapsedTime;
}

template<typename DType>
void dotest()
{
    int N = 1024*1024*256; 
    int NTest = N/32;

    CuPtr<DType> da(N);
    int NIter = 4;

    cudaEvent_t event_start, event_stop;
    checkCudaErrors(cudaEventCreate(&event_start));
    checkCudaErrors(cudaEventCreate(&event_stop));

    printf("### Warm up...\n");
    for(int i=0; i<3; i++)
    {
        int2 c = make_int2(0, i*32);
        float elapsedTime = l2bank_test_run<DType>(c, NIter, da.GetPtr(), event_start, event_stop);
        printf("  Warmup %2d: %10.3f ms\n", i, elapsedTime);
    }

    float thres = 0.0;
    float tmin(1e10), tmax(-1.f);

    printf("### Scanning threshold...\n");
    for(int i=0; i<1024; i++)
    {
        int2 c = make_int2(0, i);

        float elapsedTime = l2bank_test_run<DType>(c, NIter, da.GetPtr(), event_start, event_stop);
        printf("  Scan %2d: %10.3f ms\n", i, elapsedTime);

        if (elapsedTime<tmin)
            tmin = elapsedTime;
        
        if (elapsedTime>tmax)
            tmax = elapsedTime;
    }

    thres = (tmax+tmin)/2;
    printf("### Range =[%8.3f, %8.3f], thres = %8.3f ms\n", tmin, tmax, thres);

    int nextPos = 0;
    int nextGroup = 0;

    HostPtr<int> group(NTest);

    while(nextPos>=0)
    {
        nextGroup += 1;
        
        printf("### Scanning for group %d...\n", nextGroup);

        int cnt = 0;
        int startPos = nextPos;
        nextPos = -1; // will be updated when first non-grouped pos found

        int dsp_counter = 0;
        for(int j=startPos; j<NTest; j++)
        {
            dsp_counter++; 

            if (group(j)>0)
                continue;

            int2 c = make_int2(startPos*32, j*32);

            float elapsedTime = l2bank_test_run<DType>(c, NIter, da.GetPtr(), event_start, event_stop);
            
            if ((dsp_counter%(NTest/32)==0) )
                printf("    %8d  %8d  %10.4f ms.\n", nextGroup, dsp_counter, elapsedTime);

            // tmat(i*NTest+j) = elapsedTime;

            if (elapsedTime>=thres) // current pos is the same group
            {
                group(j) = nextGroup;
                cnt ++;
            }    

            else if (nextPos==-1) // get the first pos not in any known group
            {
                nextPos = j;
            }
        }  

        printf("###  %8d elements found in group %d.\n", cnt, nextGroup);

        // saving it every group iteration, still keep the results in case of early termination
        group.SaveToFile("group.dat");
    }
    
    checkCudaErrors(cudaEventDestroy(event_start));
    checkCudaErrors(cudaEventDestroy(event_stop));

}

int main()
{
    dotest<int>();
    // dotest<float>();
    // dotest<uint64_t>();
    return 0;
}