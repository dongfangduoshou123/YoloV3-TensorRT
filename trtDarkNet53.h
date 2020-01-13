#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include <thread>

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
#include <dirent.h>
#include <string>
#include <vector>
#include <string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include "yoloPlugin.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include "trtNetWork.h"

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

class YoloIInt8Calibrator : public IInt8EntropyCalibrator2 {
public:
    YoloIInt8Calibrator(int bcsz, int bcs, std::string datadir)
        :batchsize(bcsz),batches(bcs),clibdatadir(datadir),curbatch(0),
    imgindex(0),mInputBlobName("data"){
        clbimagepaths = ls(clibdatadir, batchsize * batches);
        assert(clbimagepaths.size() == batchsize * batches);
        mInputCount = batchsize * 3 * 416 * 416;
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    }

    virtual int getBatchSize() const override{
        return batchsize;
    }


    virtual bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    virtual const void* readCalibrationCache(std::size_t& length) override{

    }

    virtual void writeCalibrationCache(const void* ptr, std::size_t length) override{

    }

private:

    inline cv::Mat getImage(int index){
        cv::Mat img = cv::imread(clbimagepaths[index]);
        return std::move(img);
    }
    std::vector<std::string> ls(std::string path,long howmany);
    std::string clibdatadir;
    int batches;
    int batchsize;
    int curbatch;
    int imgindex;
    std::vector<std::string> clbimagepaths;
    void* mDeviceInput{nullptr};
    int mInputCount;
    const char* mInputBlobName;
};

class YoloTrtNet : public trt::trtNetWork{
public:
    YoloTrtNet(std::string engineFilePath, trt::ModelInfo minfo,trt::BuildInfo binfo)
        :trt::trtNetWork(engineFilePath, minfo, binfo){
    }

    inline std::vector<BBoxInfo> doDetect(void *data, int contextIndex, int batch){
        Inference(data, contextIndex, batch);
        std::vector<BBoxInfo> bxinfo;
        for (auto& tensor : m_OutputTensors[0])
        {
            std::vector<BBoxInfo> curBInfo = decodeTensor(0, 416, 416, tensor);
            bxinfo.insert(bxinfo.end(), curBInfo.begin(), curBInfo.end());
        }
        return std::move(nmsAllClasses(0.5, bxinfo, 7));
    }

    virtual bool CreateNetwork() override;

    virtual bool CreateEngineAndSerialize(long maxWorkspaceSize, BuilderFlag flag, int maxBatch) override;
    virtual bool EngineDeserialize() override;

private:
    std::vector<std::vector<TensorInfo> >m_OutputTensors;
    int m_InputH = 416;
    int m_InputW = 416;
    float m_ProbThresh = 0.5;

    std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);
    std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo,
                                        const uint numClasses);

    inline float clamp(const float val, const float minVal, const float maxVal)
    {
        assert(minVal <= maxVal);
        return std::min(maxVal, std::max(minVal, val));
    }
    void convertBBoxImgRes(const float scalingFactor, const float& xOffset, const float& yOffset,
                           BBox& bbox);

    BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh,
                           const uint& stride, const uint& netW, const uint& netH);

    void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                                const uint stride, const float scalingFactor, const float xOffset,
                                const float yOffset, const int maxIndex, const float maxProb,
                                std::vector<BBoxInfo>& binfo);

    std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                               const TensorInfo& tensor);

};
