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

//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "yoloPlugin.h"
#include <opencv2/opencv.hpp>
struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    int classId; // For coco benchmarking
    float prob;
};

struct TensorInfo
{
    std::string blobName;
    uint stride{0};
    uint gridSize{0};
    uint numClasses{0};
    uint numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};
float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}
void convertBBoxImgRes(const float scalingFactor, const float& xOffset, const float& yOffset,
                       BBox& bbox)
{
    // Undo Letterbox
    bbox.x1 -= xOffset;
    bbox.x2 -= xOffset;
    bbox.y1 -= yOffset;
    bbox.y2 -= yOffset;

    // Restore to input resolution
    bbox.x1 /= scalingFactor;
    bbox.x2 /= scalingFactor;
    bbox.y1 /= scalingFactor;
    bbox.y2 /= scalingFactor;
}

BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh,
                       const uint& stride, const uint& netW, const uint& netH)
{
    BBox b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;

    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    b.x1 = clamp(b.x1, 0, netW);
    b.x2 = clamp(b.x2, 0, netW);
    b.y1 = clamp(b.y1, 0, netH);
    b.y2 = clamp(b.y2, 0, netH);

    return b;
}
int m_InputH = 416;
int m_InputW = 416;
float m_ProbThresh = 0.5;
int m_BatchSize = 1;
void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                            const uint stride, const float scalingFactor, const float xOffset,
                            const float yOffset, const int maxIndex, const float maxProb,
                            std::vector<BBoxInfo>& binfo)
{
    BBoxInfo bbi;
    bbi.box = convertBBoxNetRes(bx, by, bw, bh, stride, m_InputW, m_InputH);
    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
    {
        return;
    }
    convertBBoxImgRes(scalingFactor, xOffset, yOffset, bbi.box);
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}
std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                           const TensorInfo& tensor)
{

    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    float xOffset = (m_InputW - scalingFactor * imageW) / 2;
    float yOffset = (m_InputH - scalingFactor * imageH) / 2;

    const float* detections = static_cast<float*>(tensor.hostBuffer);

    std::vector<BBoxInfo> binfo;
    for (uint y = 0; y < tensor.gridSize; ++y)
    {
        for (uint x = 0; x < tensor.gridSize; ++x)
        {
            for (uint b = 0; b < tensor.numBBoxes; ++b)
            {
                const float pw = tensor.anchors[tensor.masks[b] * 2];
                const float ph = tensor.anchors[tensor.masks[b] * 2 + 1];

                const int numGridCells = tensor.gridSize * tensor.gridSize;
                const int bbindex = y * tensor.gridSize + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0)];

                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 1)];
                const float bw
                    = pw * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 2)];
                const float bh
                    = ph * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 3)];

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < tensor.numClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + tensor.numClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
//                if(objectness > 0){
//                    std::cout << "obj:" << objectness << std::endl;
//                }
//                if(maxProb > 0){
//                    std::cout << "maxprob:" << maxProb << std::endl;
//                }
                maxProb = objectness * maxProb;


                if (maxProb > m_ProbThresh)
//                if(objectness > 0)
                {
                    addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}

void doInference(const unsigned char* input, const uint batchSize, cudaStream_t& m_CudaStream, cudaEvent_t& m_CudaEvent, nvinfer1::IExecutionContext*m_Context, std::vector<void*>& m_DeviceBuffers, int m_InputBindingIndex,int m_InputSize,std::vector<TensorInfo>&m_OutputTensors)
{
    assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
    ASSERT(cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndex), input,
                                  batchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
                                  m_CudaStream) == cudaSuccess);

//    m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
    ASSERT(m_Context->enqueueV2(m_DeviceBuffers.data(),m_CudaStream, &m_CudaEvent));

    for (auto& tensor : m_OutputTensors)
    {
        ASSERT(cudaMemcpyAsync(tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
                                      batchSize * tensor.volume * sizeof(float),
                                      cudaMemcpyDeviceToHost, m_CudaStream) == cudaSuccess);
    }
    cudaStreamSynchronize(m_CudaStream);
}
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims)
{
    std::cout << inputDims.nbDims << std::endl;
    std::cout << inputDims.d[0] << " " << inputDims.d[1] << " " << inputDims.d[2] << " " << inputDims.d[3] << std::endl;
//    assert(inputDims.nbDims == 3);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
}
std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}
std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo,
                                    const uint numClasses)
{
    std::vector<BBoxInfo> result;
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.label).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}



template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

void runWithYoloPlugin(){
    cv::Mat img = cv::imread("/opt/caffe2_yolov3/TensorRT-6.0.1.5/samples/python/yolov3_onnx/dog.jpg");
    cv::Mat imgf;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(416, 416));
    resized.convertTo(imgf,CV_32FC3, 1/255.0);


    std::vector<cv::Mat>channles(3);
    cv::split(imgf,channles);
    for(int i = 0;i < 416;i ++){
        for(int j = 0;j < 416;j ++){
            std::cout << channles[0].at<float>(i,j) << "\n";
        }
    }
    std::vector<float>data;
    float* ptr1 = (float*)(channles[0].data);
    float* ptr2 = (float*)(channles[1].data);
    float* ptr3 = (float*)(channles[2].data);
    data.insert(data.end(),ptr1,ptr1 + 416*416);
    data.insert(data.end(),ptr2,ptr2 + 416*416);
    data.insert(data.end(),ptr3,ptr3 + 416*416);

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U));
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(
        "/opt/caffe2_yolov3/TensorRT-6.0.1.5/samples/python/yolov3_onnx/yolov3.onnx", static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    auto *out13 = network->getOutput(0);
    auto *out26 = network->getOutput(1);
    auto *out52 = network->getOutput(2);
    Yolo* yolo13 = new Yolo(7, 32, 13, 3);
    Yolo* yolo26 = new Yolo(7, 16, 26, 3);
    Yolo* yolo52 = new Yolo(7, 8, 52, 3);
    IPluginV2Layer* p1 = network->addPluginV2(&out13, 1, *yolo13);
    IPluginV2Layer* p2 = network->addPluginV2(&out26, 1, *yolo26);
    IPluginV2Layer* p3 = network->addPluginV2(&out52, 1, *yolo52);

    network->markOutput(*p1->getOutput(0));
    network->markOutput(*p2->getOutput(0));
    network->markOutput(*p3->getOutput(0));
    network->unmarkOutput(*out13);
    network->unmarkOutput(*out26);
    network->unmarkOutput(*out52);
    p1->getOutput(0)->setName("yolo1");
    p2->getOutput(0)->setName("yolo2");
    p3->getOutput(0)->setName("yolo3");
    network->getInput(0)->setName("data");
    std::string m_InputBlobName = "data";
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1024 * sizeof(char) *1024 * 10);

    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;
    std::cout << "build yolove engine successfull.\n";
    std::cout << mEngine->getNbBindings() << " binds\n";
    std::vector<void*> m_DeviceBuffers;
    int m_InputBindingIndex = 0;
    int m_InputSize = 416*416*3;


    auto m_Context = mEngine->createExecutionContext();
    assert(m_Context != nullptr);
    m_InputBindingIndex = mEngine->getBindingIndex(m_InputBlobName.c_str());
    assert(m_InputBindingIndex != -1);
    assert(m_BatchSize <= static_cast<uint>(mEngine->getMaxBatchSize()));
    m_DeviceBuffers.resize(mEngine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");
    ASSERT(cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex),
                             m_BatchSize * m_InputSize * sizeof(float)) == cudaSuccess);

    std::vector<TensorInfo>m_OutputTensors;
    TensorInfo tmp;
    tmp.anchors = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
    tmp.bindingIndex = 1;
    tmp.blobName = "yolo1";
    tmp.gridSize = 13;
    tmp.hostBuffer;
    tmp.masks = {6,7,8};
    tmp.numBBoxes = 3;
    tmp.numClasses = 7;
    tmp.stride = 32;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    tmp.bindingIndex = 2;
    tmp.blobName = "yolo2";
    tmp.gridSize = 26;
    tmp.masks = {3,4,5};
    tmp.stride = 16;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    tmp.bindingIndex = 3;
    tmp.blobName = "yolo3";
    tmp.gridSize = 52;
    tmp.masks = {0,1,2};
    tmp.stride = 8;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    for (TensorInfo& tensor : m_OutputTensors)
    {
        tensor.bindingIndex = mEngine->getBindingIndex(tensor.blobName.c_str());
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");
        ASSERT(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex),
                                 m_BatchSize * tensor.volume * sizeof(float))==cudaSuccess);
        ASSERT(
            cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float))==cudaSuccess);
    }
    cudaStream_t m_CudaStream;
    cudaEvent_t m_CudaEvent;
    ASSERT(cudaEventCreate(&m_CudaEvent) == cudaSuccess);
    ASSERT(cudaStreamCreate(&m_CudaStream) == cudaSuccess);

    assert((mEngine->getNbBindings() == (1 + m_OutputTensors.size())
            && "Binding info doesn't match between cfg and engine file \n"));

    for (auto tensor : m_OutputTensors)
    {
        assert(!strcmp(mEngine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str())
               && "Blobs names dont match between cfg and engine file \n");
        std::cout << get3DTensorVolume(mEngine->getBindingDimensions(tensor.bindingIndex)) << " vs "
                  << tensor.volume;
        assert(get3DTensorVolume(mEngine->getBindingDimensions(tensor.bindingIndex))
                   == tensor.volume
               && "Tensor volumes dont match between cfg and engine file \n");
    }

    assert(mEngine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(mEngine->getBindingName(m_InputBindingIndex) == m_InputBlobName
           && "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume(mEngine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);

    //inference

    time_t t;
    time(&t);
    std::vector<BBoxInfo>remaining;
    for(int i =0;i < 1000;i ++){
        doInference((void*)data.data(), 1, m_CudaStream,m_CudaEvent ,m_Context, m_DeviceBuffers,
                    m_InputBindingIndex, m_InputSize,
                    m_OutputTensors);
        std::vector<BBoxInfo> binfo;
        for (auto& tensor : m_OutputTensors)
        {
            std::vector<BBoxInfo> curBInfo = decodeTensor(0, img.rows, img.cols, tensor);
            binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
        }
        remaining = nmsAllClasses(0.5, binfo, 7);
    }

    time_t t1;
    time(&t1);
    std::cout << "cost " << difftime(t,t1) << " seconds for 1000 iters\n";
    for (BBoxInfo &b : remaining)
    {
        cv::rectangle(img, cv::Rect(b.box.x1,b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1),cv::Scalar(255,0,0));
    }
    cv::imshow("l", img);
    cv::waitKey();
}

void runOnlyConvBlocks(){
    cv::Mat img = cv::imread("/opt/caffe2_yolov3/TensorRT-6.0.1.5/samples/python/yolov3_onnx/dog.jpg");
    cv::Mat imgf;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(416, 416));
    resized.convertTo(imgf,CV_32FC3, 1/255.0);


    std::vector<cv::Mat>channles(3);
    cv::split(imgf,channles);
    for(int i = 0;i < 416;i ++){
        for(int j = 0;j < 416;j ++){
            std::cout << channles[0].at<float>(i,j) << "\n";
        }
    }
    std::vector<float>data;
    float* ptr1 = (float*)(channles[0].data);
    float* ptr2 = (float*)(channles[1].data);
    float* ptr3 = (float*)(channles[2].data);
    data.insert(data.end(),ptr1,ptr1 + 416*416);
    data.insert(data.end(),ptr2,ptr2 + 416*416);
    data.insert(data.end(),ptr3,ptr3 + 416*416);

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U));
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(
        "/opt/caffe2_yolov3/TensorRT-6.0.1.5/samples/python/yolov3_onnx/yolov3.onnx", static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    network->getInput(0)->setName("data");
    network->getOutput(0)->setName("yolo1");
    network->getOutput(1)->setName("yolo2");
    network->getOutput(2)->setName("yolo3");
    std::string m_InputBlobName = "data";
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1024 * sizeof(char) *1024 * 10);

    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;
    std::cout << "build yolove engine successfull.\n";
    std::cout << mEngine->getNbBindings() << " binds\n";
    std::vector<void*> m_DeviceBuffers;
    int m_InputBindingIndex = 0;
    int m_InputSize = 416*416*3;


    auto m_Context = mEngine->createExecutionContext();
    assert(m_Context != nullptr);
    m_InputBindingIndex = mEngine->getBindingIndex(m_InputBlobName.c_str());
    assert(m_InputBindingIndex != -1);
    assert(m_BatchSize <= static_cast<uint>(mEngine->getMaxBatchSize()));
    m_DeviceBuffers.resize(mEngine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");
    ASSERT(cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex),
                             m_BatchSize * m_InputSize * sizeof(float)) == cudaSuccess);

    std::vector<TensorInfo>m_OutputTensors;
    TensorInfo tmp;
    tmp.anchors = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
    tmp.bindingIndex = 1;
    tmp.blobName = "yolo1";
    tmp.gridSize = 13;
    tmp.hostBuffer;
    tmp.masks = {6,7,8};
    tmp.numBBoxes = 3;
    tmp.numClasses = 7;
    tmp.stride = 32;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    tmp.bindingIndex = 2;
    tmp.blobName = "yolo2";
    tmp.gridSize = 26;
    tmp.masks = {3,4,5};
    tmp.stride = 16;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    tmp.bindingIndex = 3;
    tmp.blobName = "yolo3";
    tmp.gridSize = 52;
    tmp.masks = {0,1,2};
    tmp.stride = 8;
    tmp.volume = tmp.gridSize
            * tmp.gridSize
            * (tmp.numBBoxes * (5 + tmp.numClasses));
    m_OutputTensors.push_back(tmp);
    for (TensorInfo& tensor : m_OutputTensors)
    {
        tensor.bindingIndex = mEngine->getBindingIndex(tensor.blobName.c_str());
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");
        ASSERT(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex),
                                 m_BatchSize * tensor.volume * sizeof(float))==cudaSuccess);
        ASSERT(
            cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float))==cudaSuccess);
    }
    cudaStream_t m_CudaStream;
    cudaEvent_t m_CudaEvent;
    ASSERT(cudaEventCreate(&m_CudaEvent) == cudaSuccess);
    ASSERT(cudaStreamCreate(&m_CudaStream) == cudaSuccess);

    assert((mEngine->getNbBindings() == (1 + m_OutputTensors.size())
            && "Binding info doesn't match between cfg and engine file \n"));

    for (auto tensor : m_OutputTensors)
    {
        assert(!strcmp(mEngine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str())
               && "Blobs names dont match between cfg and engine file \n");
        std::cout << get3DTensorVolume(mEngine->getBindingDimensions(tensor.bindingIndex)) << " vs "
                  << tensor.volume;
        assert(get3DTensorVolume(mEngine->getBindingDimensions(tensor.bindingIndex))
                   == tensor.volume
               && "Tensor volumes dont match between cfg and engine file \n");
    }

    assert(mEngine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(mEngine->getBindingName(m_InputBindingIndex) == m_InputBlobName
           && "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume(mEngine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);

    //inference

    time_t t;
    time(&t);
    for(int i =0;i < 1000;i ++){
        doInference((void*)data.data(), 1, m_CudaStream,m_CudaEvent,m_Context, m_DeviceBuffers,
                    m_InputBindingIndex, m_InputSize,
                    m_OutputTensors);
//        std::vector<BBoxInfo> binfo;
//        for (auto& tensor : m_OutputTensors)
//        {
//            std::vector<BBoxInfo> curBInfo = decodeTensor(0, img.rows, img.cols, tensor);
//            binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
//        }
//        auto remaining = nmsAllClasses(0.5, binfo, 7);
    }

    time_t t1;
    time(&t1);
    std::cout << "cost " << difftime(t,t1) << " seconds for 1000 iters\n";
}

int main(int argc, char** argv)
{
    runWithYoloPlugin();
}
