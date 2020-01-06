#ifndef __UTILS_H__
#define __UTILS_H__
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) do{\
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    {\
        printf("ERROR: %s:%d,",__FILE__, __LINE__); \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}while(0)

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d : %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

#endif