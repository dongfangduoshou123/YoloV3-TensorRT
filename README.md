# YoloV3-TensorRT
Run YoloV3 with the newest TensorRT6.0 at 37 fps on  NVIIDIA 1060.

I use the TensorRT's yolov3 python example's script(location at TensorRT-ROOT/samples/python/yolov3_onnx), to convert the yolov3 model from darknet to onnx format named yolov3.onnx, which does not contains the yolo layer.

In C++, I first parse the yolov3.onnx, then use the TensorRT's api to edit the parsed network(add the yoloplugin to the network,and mark the yoloplugin's output as network's output, and unmark the original output), then build the engine, and run inference.

Run in fp16 and int8 has not coded and tested from now.

Feature:

    1:Use TensorRT's new plugin interface IPluginV2Ext to implement the yolo plugin.
    2:Run on the current newest TensorRT version 6.0.1.5.
    3:test on NVIDIA 1060 at 37 fps.
    4.include the post process such as nms to get the final detection result.
    
Requirment:

    Install the TensorRT6.0.1.5,(I use CUDA10.1 version, but other cuda version's TensorRT should also work well).


Reference:

    1:https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps (restructure branch, not master branch)
    2:https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide
