#pragma once

#include "nvml.h"

#define checkNVMLCall(x) __checkNVMLError(x, __FILE__, __LINE__)

inline void __checkNVMLError(const nvmlReturn_t &res, const char *file,
                                 const int line) {
  
  const char* err = nvmlErrorString(res);
  if (NVML_SUCCESS != res) {
    fprintf(stderr, "%s:%i : NVML call error!\n  Code=%d:%s!\n",
            file, line, static_cast<int>(res), err);
  }
}

void showClockStatus(const nvmlDevice_t& dev, const nvmlClockId_t& clockid)
{
    unsigned int clk_gfx; // in MHz
    unsigned int clk_sm;
    unsigned int clk_mem;

    // nvmlClockType_t:
    //   NVML_CLOCK_GRAPHICS = 0
    //     Graphics clock domain. 
    //   NVML_CLOCK_SM = 1
    //     SM clock domain. 
    //   NVML_CLOCK_MEM = 2
    //     Memory clock domain. 
    //   NVML_CLOCK_VIDEO = 3
    //     Video encoder/decoder clock domain. 
    // nvmlClockID_t:
    //   NVML_CLOCK_ID_CURRENT = 0
    //     Current actual clock value. 
    //   NVML_CLOCK_ID_APP_CLOCK_TARGET = 1
    //     Target application clock. 
    //   NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2
    //     Default application clock target. 
    //   NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
    //     OEM-defined maximum clock rate. 

    // proto: nvmlReturn_t nvmlDeviceGetClock ( nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz )
    checkNVMLCall(nvmlDeviceGetClock (dev, NVML_CLOCK_GRAPHICS, clockid, &clk_gfx ));
    checkNVMLCall(nvmlDeviceGetClock (dev, NVML_CLOCK_SM, clockid, &clk_sm ));
    checkNVMLCall(nvmlDeviceGetClock (dev, NVML_CLOCK_MEM, clockid, &clk_mem ));

    /*
    char *tag="";
    switch(clockid)
    {
        case NVML_CLOCK_ID_CURRENT:
            tag = "Current";
            break;
        case NVML_CLOCK_ID_APP_CLOCK_TARGET:
            tag = "Target";
            break;
        case NVML_CLOCK_ID_APP_CLOCK_DEFAULT:
            tag = "Default";
            break;
        case NVML_CLOCK_ID_CUSTOMER_BOOST_MAX:
            tag = "BoostMax";
            break;
        default:
            break;
    }*/

    printf(" %4u | %4u | %4u |", clk_gfx, clk_sm, clk_mem);
}

// nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons ( nvmlDevice_t device, unsigned long long* clocksThrottleReasons ) 
// #define nvmlClocksThrottleReasonAll
// #define nvmlClocksThrottleReasonApplicationsClocksSetting 0x0000000000000002LL
// #define nvmlClocksThrottleReasonDisplayClockSetting 0x0000000000000100LL
// #define nvmlClocksThrottleReasonGpuIdle 0x0000000000000001LL
// #define nvmlClocksThrottleReasonHwPowerBrakeSlowdown 0x0000000000000080LL
// #define nvmlClocksThrottleReasonHwSlowdown 0x0000000000000008LL
// #define nvmlClocksThrottleReasonHwThermalSlowdown 0x0000000000000040LL
// #define nvmlClocksThrottleReasonNone 0x0000000000000000LL
// #define nvmlClocksThrottleReasonSwPowerCap 0x0000000000000004LL
// #define nvmlClocksThrottleReasonSwThermalSlowdown 0x0000000000000020LL
// #define nvmlClocksThrottleReasonSyncBoost 0x0000000000000010LL
void dispCurrentClocksThrottleReasons(unsigned long long reasons)
{
    printf(" 0x%llx:", reasons);
    if (reasons == nvmlClocksThrottleReasonNone)
    {
        printf("None");
        return;
    }    


    if (reasons & nvmlClocksThrottleReasonApplicationsClocksSetting)
        printf("AppSet,");
    
    if (reasons & nvmlClocksThrottleReasonDisplayClockSetting)
        printf("DispSet,");
    
    if (reasons & nvmlClocksThrottleReasonGpuIdle)
        printf("Idle,");
    
    if (reasons & nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
        printf("HwPowerBrake,");

    if (reasons & nvmlClocksThrottleReasonHwSlowdown)
        printf("HwSlow,");

    if (reasons & nvmlClocksThrottleReasonHwThermalSlowdown)
        printf("HwThermal,");

    if (reasons & nvmlClocksThrottleReasonSwPowerCap)
        printf("SwPowerCap,");

    if (reasons & nvmlClocksThrottleReasonSwThermalSlowdown)
        printf("SwThermal,");

    if (reasons & nvmlClocksThrottleReasonSyncBoost)
        printf("SyncBoost,");
}

void showDeviceStatus()
{
    checkNVMLCall(nvmlInit());

    unsigned int ndev;
    
    checkNVMLCall(nvmlDeviceGetCount(&ndev));
    
    printf("======================================================================================================\n");
    printf("  Device  | Utilization | Temp |    Memory Usage (GB)     |     Clock (MHz)    | ClockThrottleReason  \n");
    printf("  i (id)  |  GPU |  MEM |      |  Total |  Used  |  Free  |  GFX |  SM  |  MEM |                      \n");

    for (int i=0; i<ndev; i++)
    {
        nvmlDevice_t dev;
        checkNVMLCall(nvmlDeviceGetHandleByIndex(i, &dev));

        //nvmlReturn_t nvmlDeviceGetBoardId ( nvmlDevice_t device, unsigned int* boardId ) 
        unsigned int boardId;
        checkNVMLCall(nvmlDeviceGetBoardId(dev, &boardId));

        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        checkNVMLCall(nvmlDeviceGetName(dev, name, NVML_DEVICE_NAME_BUFFER_SIZE));

        nvmlUtilization_t utilization;
        checkNVMLCall(nvmlDeviceGetUtilizationRates(dev, &utilization));

        nvmlMemory_t mem;
        checkNVMLCall(nvmlDeviceGetMemoryInfo(dev, &mem));

        unsigned int temperature = 0;
        checkNVMLCall(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temperature));

        double gb_ratio = 1./(1024.*1024.*1024.);
        printf("%2d (%4x) |", i, boardId);
        printf("  %3d |  %3d |", utilization.gpu, utilization.memory);
        printf(" %3dC |", temperature);
        printf(" %6.3f | %6.3f | %6.3f |", mem.total*gb_ratio, mem.used*gb_ratio, mem.free*gb_ratio);

        showClockStatus(dev, NVML_CLOCK_ID_CURRENT);
        // showClockStatus(dev, NVML_CLOCK_ID_APP_CLOCK_TARGET);
        // showClockStatus(dev, NVML_CLOCK_ID_APP_CLOCK_DEFAULT);        
        // showClockStatus(dev, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX);

        unsigned long long reasons;
        checkNVMLCall(nvmlDeviceGetCurrentClocksThrottleReasons (dev, &reasons));
        dispCurrentClocksThrottleReasons(reasons);

        printf("\n");
    }

    printf("======================================================================================================\n");
}


