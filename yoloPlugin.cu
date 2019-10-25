/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "yoloPlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayerV3(const float* input, float* output, const uint gridSize, const uint numOutputClasses,
                               const uint numBBoxes)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize,
                            const uint& numOutputClasses, const uint& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
                          (gridSize / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSize, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}

using namespace nvinfer1;
using nvinfer1::plugin::Yolo;
using nvinfer1::plugin::YoloPluginCreator;

namespace
{
const char* Yolo_PLUGIN_VERSION{"1"};
const char* Yolo_PLUGIN_NAME{"Yolo_TRT"};
} // namespace

PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;


Yolo::Yolo(int numclass, int stride, int gridesize, int numanchors)
{
    numanchors_ = numanchors;
    numclass_ = numclass;
    stride_ = stride;
    gridesize_ = gridesize;
}

Yolo::Yolo(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    numanchors_ = read<int>(d);
    numclass_ = read<int>(d);
    stride_ = read<int>(d);
    gridesize_ = read<int>(d);
    ASSERT(d == a + length);
}

int Yolo::getNbOutputs() const
{
    return 1;
}

Dims Yolo::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(inputs[0].nbDims == 3);
    return inputs[0];
}

int Yolo::initialize()
{
    return 0;
}

void Yolo::terminate()
{

}

size_t Yolo::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int Yolo::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    ASSERT(cudaYoloLayerV3(inputs[0],outputs[0],batchSize, gridesize_,numclass_, numanchors_, gridesize_*gridesize_*numanchors_*(5 + numclass_),stream)
            == cudaSuccess);
    return 0;
}

size_t Yolo::getSerializationSize() const
{
    return 4 * sizeof(int);
}

void Yolo::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, numanchors_);
    write(d, numclass_);
    write(d, stride_);
    write(d, gridesize_);
    ASSERT(d == a + getSerializationSize());
}

bool Yolo::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

// Set plugin namespace
void Yolo::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* Yolo::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType Yolo::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Yolo::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Yolo::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void Yolo::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
    assert(nbInputs == 1);
    assert(inputDims != nullptr);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Yolo::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Yolo::detachFromContext() {}

const char* Yolo::getPluginType() const
{
    return Yolo_PLUGIN_NAME;
}

const char* Yolo::getPluginVersion() const
{
    return Yolo_PLUGIN_VERSION;
}

void Yolo::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2Ext* Yolo::clone() const
{
    // Create a new instance
    IPluginV2Ext* plugin = new Yolo(numclass_,stride_,gridesize_,numanchors_);
    // Set the namespace
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

YoloPluginCreator::YoloPluginCreator()
{

    mPluginAttributes.emplace_back(PluginField("numclass", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gridsize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numanchors", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const
{
    return Yolo_PLUGIN_NAME;
}

const char* YoloPluginCreator::getPluginVersion() const
{
    return Yolo_PLUGIN_VERSION;
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    ASSERT(!strcmp(name, getPluginName()));
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "numclass"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            numclass_ = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride_ = *(static_cast<const int*>(fields[i].data));
        }else if(!strcmp(attrName, "gridesize")){
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            gridesize_ = *(static_cast<const int*>(fields[i].data));
        }else if(!strcmp(attrName, "numanchors")){
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            numanchors_ = *(static_cast<const int*>(fields[i].data));
        }
    }
    Yolo* obj = new Yolo(numclass_, stride_, gridesize_, numanchors_);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    Yolo* obj = new Yolo(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
