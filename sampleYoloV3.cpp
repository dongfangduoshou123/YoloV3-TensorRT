#include "trtDarkNet53.h"
//add by wzq 2020.1.13
int main()
{
    using namespace trt;
    using namespace nvinfer1;
    ModelInfo minfo;
    BuildInfo binfo;
    minfo.format = ModelFormat::kONNX;
    minfo.onnxmodelfile = "/opt/TensorRT-6.0.1.5/samples/python/yolov3_onnx/yolov3.onnx";
    minfo.outputs;

    binfo.maxWorkspaceSize = 1024;
    binfo.contextNum = 3;
    binfo.flag = nvinfer1::BuilderFlag::kGPU_FALLBACK;
    binfo.safe = false;
    binfo.dlacore = -1;
    binfo.maxBatch = 1;
    YoloTrtNet trtnet("/opt/onnx.trt",minfo, binfo);
    trtnet.GetEngine();

    cv::VideoCapture cap;
    cap.open("/home/dtt/Videos/daytime/v_1509084073705.avi");
    if(cap.isOpened()){
        while(true){
            cv::Mat img;
            cap >> img;
            cv::Mat imgf;
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(416, 416));
            resized.convertTo(imgf,CV_32FC3, 1/255.0);
            std::vector<cv::Mat>channles(3);
            cv::split(imgf,channles);
            std::vector<float>data;
            float* ptr1 = (float*)(channles[0].data);
            float* ptr2 = (float*)(channles[1].data);
            float* ptr3 = (float*)(channles[2].data);
            data.insert(data.end(),ptr1,ptr1 + 416*416);
            data.insert(data.end(),ptr2,ptr2 + 416*416);
            data.insert(data.end(),ptr3,ptr3 + 416*416);

            auto remaining = trtnet.doDetect(data.data(),0, -1);
            for (BBoxInfo &b : remaining)
            {
                cv::rectangle(resized, cv::Rect(b.box.x1,b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1),cv::Scalar(255,0,0));
            }

            cv::imshow("l", resized);
            cv::waitKey(1);
        }
    }
}
