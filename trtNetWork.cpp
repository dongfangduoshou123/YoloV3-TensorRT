#include "trtNetWork.h"
#include "NvInferRuntimeCommon.h"
#include <strstream>
#include <fstream>
#include <iostream>

namespace trt{
using namespace nvinfer1;
struct BufferShutter
{
    ~BufferShutter()
    {
        nvcaffeparser1::shutdownProtobufLibrary();
    }
};


bool trtNetWork::CreateNetwork()
{
    if(modelInfo.format == ModelFormat::kCAFFE){
        using namespace nvcaffeparser1;
        parser.caffeParser.reset(createCaffeParser());
        BufferShutter bufferShutter;
        const auto blobNameToTensor = parser.caffeParser->parse(
            modelInfo.caffeprotoxt.c_str(), modelInfo.caffemodel.empty() ? nullptr : modelInfo.caffemodel.c_str(), *network, DataType::kFLOAT);
        if (!blobNameToTensor)
        {
            std::cerr << "Failed to parse caffe model or prototxt, tensors blob not found" << std::endl;
            parser.caffeParser.reset();
            return false;
        }

        for (const auto& s : modelInfo.outputs)
        {
            if (blobNameToTensor->find(s.c_str()) == nullptr)
            {
                std::cerr << "Could not find output blob " << s << std::endl;
                parser.caffeParser.reset();
                return false;
            }
            network->markOutput(*blobNameToTensor->find(s.c_str()));
        }
    }else if(modelInfo.format == ModelFormat::kUFF){
        BufferShutter bufferShutter;
        using namespace nvuffparser;
        parser.uffParser.reset(createUffParser());
        for (const auto& s : modelInfo.uffInputs.inputs)
        {
            if (!parser.uffParser->registerInput(
                    s.first.c_str(), s.second, modelInfo.uffInputs.NHWC ? UffInputOrder::kNHWC : UffInputOrder::kNCHW))
            {
                std::cerr << "Failed to register input " << s.first << std::endl;
                parser.uffParser.reset();
                return false;
            }
        }

        for (const auto& s : modelInfo.outputs)
        {
            if (!parser.uffParser->registerOutput(s.c_str()))
            {
                std::cerr << "Failed to register output " << s << std::endl;
                parser.uffParser.reset();
                return false;
            }
        }

        if (!parser.uffParser->parse(modelInfo.uffmodel.c_str(), *network))
        {
            std::cerr << "Failed to parse uff file" << std::endl;
            parser.uffParser.reset();
            return false;
        }
    }else if(modelInfo.format == ModelFormat::kONNX){
        using namespace nvonnxparser;
        parser.onnxParser.reset(createParser(*network, gLogger.getTRTLogger()));
        if (!parser.onnxParser->parseFromFile(
                modelInfo.onnxmodelfile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
        {
            std::cerr << "Failed to parse onnx file" << std::endl;
            parser.onnxParser.reset();
            return false;
        }
    }else{
        std::cerr << "unsupported model format\n";
        assert(false);
        abort();
    }
    return true;
}

void trtNetWork::Inference(void *data, int contextIndex, int batch)
{
    assert(contextIndex < buildinfo.contextNum);
    std::vector<void*>& cur = m_DeviceBuffers[contextIndex];
    if(batch > 0){
        assert(cudaMemcpyAsync(cur[0], data,
                                     binds[contextIndex].deviceBufferBinds[0].size() , cudaMemcpyHostToDevice,
                                      streams[contextIndex]->get()) == cudaSuccess);

        auto ret = exeContexts[contextIndex]->enqueue(batch, cur.data(), streams[contextIndex]->get(),nullptr);
    }else{
        assert(cudaMemcpyAsync(cur[0], data,
                                     binds[contextIndex].deviceBufferBinds[0].size() , cudaMemcpyHostToDevice,
                                      streams[contextIndex]->get()) == cudaSuccess);

        assert(exeContexts[contextIndex]->enqueueV2(cur.data(), streams[contextIndex]->get(), nullptr));

    }

    for (int i =1;i < cur.size();i ++)
    {
        assert(cudaMemcpyAsync(binds[contextIndex].hostBufferBinds[i].get(), cur[i],
                                      binds[contextIndex].hostBufferBinds[i].size(),
                                      cudaMemcpyDeviceToHost, streams[contextIndex]->get()) == cudaSuccess);
    }
    cudaStreamSynchronize(streams[contextIndex]->get());
}


bool trtNetWork::CreateEngineAndSerialize(long MaxWorkspaceSize,nvinfer1::BuilderFlag flag, int maxBatch=-1)
{
    trtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};

    IOptimizationProfile* profile{nullptr};
    if (maxBatch)
    {
        builder->setMaxBatchSize(maxBatch);
    }
    else
    {
        if (!modelInfo.shapes.empty())
        {
            profile = builder->createOptimizationProfile();
        }
    }


    for (unsigned int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        auto input = network->getInput(i);
        if (!modelInfo.inputsformat.empty())
        {
            input->setType(modelInfo.inputsformat[i].first);
            input->setAllowedFormats(1U << static_cast<int>(modelInfo.inputsformat[i].second));
        }
        else
        {
            input->setType(DataType::kFLOAT);
            input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }

        if (profile)
        {
            Dims dims = input->getDimensions();
            if (std::any_of(dims.d + 1, dims.d + dims.nbDims, [](int dim) { return dim == -1; }))
            {
                std::cerr << "Only dynamic batch dimension is currently supported, other dimensions must be static"
                    << std::endl;
                return false;
            }
            dims.d[0] = -1;
            Dims profileDims = dims;
            auto shape = modelInfo.shapes.find(input->getName());
            if (shape == modelInfo.shapes.end())
            {
                std::cerr << "Dynamic dimensions required for input " << input->getName() << std::endl;
                return false;
            }
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMIN)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kOPT)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMAX)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, profileDims);

            input->setDimensions(dims);
        }
    }

    if (profile)
    {
        if (!profile->isValid())
        {
            std::cerr << "Required optimization profile is invalid" << std::endl;
            return false;
        }
        config->addOptimizationProfile(profile);
    }

    for (unsigned int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        auto output = network->getOutput(i);
        if (!modelInfo.outputsformat.empty())
        {
            output->setType(modelInfo.outputsformat[i].first);
            output->setAllowedFormats(1U << static_cast<int>(modelInfo.outputsformat[i].second));
        }
        else
        {
            output->setType(DataType::kFLOAT);
            output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }
    }

    config->setMaxWorkspaceSize(static_cast<size_t>(MaxWorkspaceSize) << 20);


    config->setFlag(flag);

    if (flag == BuilderFlag::kINT8)
    {
        if(buildinfo.Int8Calibrator == nullptr){
            std::cerr << "Int8Calibrator is not implemented." << std::endl;
            return false;
        }
        config->setInt8Calibrator(buildinfo.Int8Calibrator.get());
    }

    if (safe)
    {
        config->setEngineCapability(DLACore != -1 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }

    if (DLACore != -1)
    {
        if (DLACore < builder->getNbDLACores())
        {
            config->setDefaultDeviceType(DeviceType::kDLA);
            config->setDLACore(DLACore);
            config->setFlag(BuilderFlag::kSTRICT_TYPES);

            config->setFlag(BuilderFlag::kGPU_FALLBACK);
            if (flag != BuilderFlag::kINT8)
            {
                config->setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            std::cerr << "Cannot create DLA engine, " << DLACore << " not available" << std::endl;
            return false;
        }
    }

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    if(engine != nullptr){
        IHostMemory* engine_serialize = engine->serialize();
        std::ofstream out(engineFilePath, std::ios::binary);
        out.write(engine_serialize->data(),engine_serialize->size());
        for(int i = 0;i < buildinfo.contextNum;i ++){
            exeContexts.emplace_back(engine->createExecutionContext());
            TrtBinds bin;
            bin.deviceBufferBinds.resize(engine->getNbBindings());
            bin.hostBufferBinds.resize(engine->getNbBindings());
            std::vector<void*> bufs;
            for(int i =0;i < engine->getNbBindings();i ++){
                bin.deviceBufferBinds[i].allocate(volume(engine->getBindingDimensions(i))*sizeof(float));
                bin.hostBufferBinds[i].allocate(volume(engine->getBindingDimensions(i))*sizeof(float));
                bufs.push_back(bin.deviceBufferBinds[i].get());
            }
            m_DeviceBuffers.push_back(bufs);
            binds.push_back(std::move(bin));
            CudaStream* stream =new CudaStream();
            streams.push_back(stream);
        }
        return true;
    }
    return false;
}

bool trtNetWork::EngineDeserialize()
{
    std::ifstream engineFile(engineFilePath.c_str(), std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Error opening engine file: " << engineFilePath << std::endl;
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cerr << "Error loading engine file: " << engineFilePath << std::endl;
        return false;
    }

    runtime.reset(createInferRuntime(gLogger.getTRTLogger()));
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    if(engine != nullptr){
        streams.resize(buildinfo.contextNum);
        for(int i = 0;i < buildinfo.contextNum;i ++){
            exeContexts.emplace_back(engine->createExecutionContext());
            TrtBinds bin;
            bin.deviceBufferBinds.resize(engine->getNbBindings());
            bin.hostBufferBinds.resize(engine->getNbBindings());
            std::vector<void*> bufs;
            for(int i =0;i < engine->getNbBindings();i ++){
                bin.deviceBufferBinds[i].allocate(volume(engine->getBindingDimensions(i))*sizeof(float));
                bin.hostBufferBinds[i].allocate(volume(engine->getBindingDimensions(i))*sizeof(float));
                bufs.push_back(bin.deviceBufferBinds[i].get());
            }
            m_DeviceBuffers.push_back(bufs);
            binds.push_back(std::move(bin));
            streams[i] = new CudaStream();
        }
        return true;
    }
    return false;
}

void trtNetWork::Destroy()
{
    std::vector<TrtBinds> binds;
    for(TrtBinds& it : binds){
        for(DeviceBuffer& dv: it.deviceBufferBinds){
            dv.reset(nullptr);
        }
        for(HostBuffer& hs: it.hostBufferBinds){
            hs.reset(nullptr);
        }
    }
    for(auto& it : exeContexts){
        it.reset(nullptr);
    }
    runtime.reset(nullptr);
    engine.reset(nullptr);
    network.reset(nullptr);
    builder.reset(nullptr);
}



}//namespace trt
