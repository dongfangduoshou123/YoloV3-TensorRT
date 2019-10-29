
#include <cuda.h>
#include <cuda_runtime_api.h>

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize,
                            const uint& numOutputClasses, const uint& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream);
