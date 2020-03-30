#pragma once
#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include <string>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <numeric>

class CudaEvent;
inline void cudaCheck(cudaError_t ret, std::ostream& err = std::cerr){
    if(ret != cudaSuccess){
        err << "Cuda failure:" << cudaGetErrorString(ret) << std::endl;
        abort();
    }
}

namespace
{

void cudaSleep(void* sleep)
{
    std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
}

}

class CudaStream{
public:
    CudaStream(){cudaCheck(cudaStreamCreate(&stream));}

    CudaStream(const CudaStream&) = delete;

    CudaStream(CudaStream&&) = delete;

    CudaStream& operator =(const CudaStream&) = delete;

    CudaStream& operator =(CudaStream&&) = delete;

    ~CudaStream() {cudaCheck(cudaStreamDestroy(stream));}

    cudaStream_t get() const{
        return stream;
    }

    void wait(CudaEvent& event);

    void sleep(int* ms){
        cudaCheck(cudaLaunchHostFunc(stream, cudaSleep, ms));
    }
private:
    cudaStream_t stream{};
};

class CudaEvent{
public:
    CudaEvent(unsigned int flags){
        cudaCheck(cudaEventCreateWithFlags(&event,flags));
    }

    CudaEvent(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&&) = delete;

    CudaEvent& operator=(CudaEvent&&) = delete;

    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent(){
        cudaCheck(cudaEventDestroy(event));
    }

    cudaEvent_t get() const{
        return event;
    }

    void record(const CudaStream& stream) {
        cudaCheck(cudaEventRecord(event, stream.get()));
    }

    void synchronize(){
        cudaCheck(cudaEventSynchronize(event));
    }

    void reset(unsigned int flags = cudaEventDefault){
        cudaCheck(cudaEventDestroy(event));
        cudaCheck(cudaEventCreateWithFlags(&event, flags));
    }

    float operator-(const CudaEvent& ohs){
        float time(0);
        cudaCheck(cudaEventElapsedTime(&time, ohs.get(), get()));
        return time;
    }

private:
    cudaEvent_t event{};
};


inline void CudaStream::wait(CudaEvent &event){
    cudaCheck(cudaStreamWaitEvent(stream, event.get(),0));
}

template <typename Allocator, typename Deletor>
class Buffer{
public:
    Buffer() = default;

    Buffer(const Buffer&) = delete;

    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& otr){
        reset(otr.get());
        otr.ptr = nullptr;
    }

    Buffer& operator=(Buffer&& oth){
        reset(oth.get());
        oth.ptr = nullptr;
    }

    Buffer(size_t size){
        Allocator()(&ptr, size);
        siz = size;
    }

    ~Buffer() {
        reset();
    }

    void allocate(size_t size){
        reset();
        Allocator()(&ptr, size);
        siz = size;
    }

    void reset(void* p=nullptr){
        if(ptr != nullptr){
            Deletor()(ptr);
        }
        ptr = p;
    }

    void* get(){
        return ptr;
    }

    size_t size(){
        return siz;
    }

private:
    void* ptr{nullptr};
    size_t siz;
};

struct DeviceAllocator
{
    void operator()(void** ptr, size_t size) { cudaCheck(cudaMalloc(ptr, size)); }
};

struct DeviceDeallocator
{
    void operator()(void* ptr) { cudaCheck(cudaFree(ptr)); }
};

struct HostAllocator
{
    void operator()(void** ptr, size_t size) { cudaCheck(cudaMallocHost(ptr, size)); }
};

struct HostDeallocator
{
    void operator()(void* ptr) { cudaCheck(cudaFreeHost(ptr)); }
};

using DeviceBuffer = Buffer<DeviceAllocator, DeviceDeallocator>;

using HostBuffer = Buffer<HostAllocator, HostDeallocator>;


struct TrtBinds{
    std::vector<DeviceBuffer> deviceBufferBinds;
    std::vector<HostBuffer> hostBufferBinds;
//    std::unordered_map<std::string, int> tensor2Index;
//    std::unordered_map<int, std::string> index2Tensor;
};
template<typename T>
struct trtDeletor{
    void operator()(T* p){p->destroy();}
};

template<typename T> using trtUniquePtr=std::unique_ptr<T, trtDeletor<T> >;

struct Parser
{
    trtUniquePtr<nvcaffeparser1::ICaffeParser> caffeParser;
    trtUniquePtr<nvuffparser::IUffParser> uffParser;
    trtUniquePtr<nvonnxparser::IParser> onnxParser;
};



namespace trt{
enum class ModelFormat
{
    kCAFFE,
    kONNX,
    kUFF
};

struct UffInput
{
    std::vector<std::pair<std::string, nvinfer1::Dims>> inputs;
    bool NHWC{false};
};
using ShapeMinOptMax = std::array<nvinfer1::Dims, 3>;

struct ModelInfo{
    ModelFormat format;
    //for onnx
    std::string onnxmodelfile;
    //for caffe
    std::string caffeprotoxt;
    std::string caffemodel;
    //for uff
    std::string uffmodel;
    UffInput uffInputs;
    //caffe model and uffmodel must implement, onnx model does not need
    std::vector<std::string> outputs;
    //optional
    std::unordered_map<std::string, ShapeMinOptMax> shapes;
    //general the number of inputsformat and outputsformat must be equal to the network's inputs number and outputs number.
    std::vector<std::pair<nvinfer1::DataType, nvinfer1::TensorFormat> > inputsformat;
    std::vector<std::pair<nvinfer1::DataType, nvinfer1::TensorFormat> > outputsformat;
};

struct BuildInfo{
    BuildInfo() {}
    long maxWorkspaceSize;
    int contextNum;
    nvinfer1::BuilderFlag flag;
    std::shared_ptr<nvinfer1::IInt8EntropyCalibrator2> Int8Calibrator{nullptr};
    bool safe;
    int dlacore;
    int maxBatch;
    BuildInfo(const BuildInfo& l) {
        maxWorkspaceSize = l.maxWorkspaceSize;
        contextNum = l.contextNum;
        flag = l.flag;
        Int8Calibrator = l.Int8Calibrator;
        safe = l.safe;
        dlacore = l.dlacore;
        maxBatch = l.maxBatch;
    }
};
inline int volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}
class trtNetWork{
public:
    trtNetWork(std::string engineFilePath, ModelInfo minfo,BuildInfo binfo)
        :engineFilePath(engineFilePath), modelInfo(minfo),safe(binfo.safe),DLACore(binfo.dlacore),buildinfo(binfo)
    {
        builder.reset(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        network.reset(builder->createNetworkV2(1U));
    }

    virtual ~trtNetWork() {
        Destroy();
    }
    //inference output is not returned and the output can get from the output memory bindings.
    virtual void Inference(void*data,int contextIndex=0, int batch=-1);

    virtual bool CreateNetwork();

    virtual bool CreateEngineAndSerialize(long maxWorkspaceSize,nvinfer1::BuilderFlag flag, int maxBatch);

    virtual bool EngineDeserialize();

    virtual void Destroy();

    void GetEngine(){
        if(access(engineFilePath.c_str(),0) == 0){
            EngineDeserialize();
        }else{
            CreateNetwork();
            CreateEngineAndSerialize(buildinfo.maxWorkspaceSize, buildinfo.flag, buildinfo.maxBatch);
        }
    }

protected:
    std::string engineFilePath;
    trtUniquePtr<nvinfer1::IBuilder> builder{nullptr};
    trtUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    trtUniquePtr<nvinfer1::INetworkDefinition> network{nullptr};
    trtUniquePtr<nvinfer1::IRuntime> runtime{nullptr};
    std::vector<trtUniquePtr<nvinfer1::IExecutionContext>> exeContexts;
    std::vector<TrtBinds> binds;
    //store the engine's input output pointers, these pointers are not mangaged by TrtBinds
    std::vector<std::vector<void*> >m_DeviceBuffers;
    std::vector<CudaStream*> streams;
    Parser parser;
    ModelInfo modelInfo;
    bool safe{false};
    int DLACore{-1};
    BuildInfo buildinfo;
};

}//namespace trt
