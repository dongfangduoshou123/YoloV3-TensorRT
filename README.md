# YoloV3-TensorRT
Run YoloV3 with the newest TensorRT6.0 at 37 fps on  NVIIDIA 1060.

I use the TensorRT's yolov3 python example's script(location at TensorRT-ROOT/samples/python/yolov3_onnx), to convert the yolov3 model from darknet to onnx format named yolov3.onnx, which does not contains the yolo layer.

In C++, I first parse the yolov3.onnx, then use the TensorRT's api to edit the parsed network(add the yoloplugin to the network,and mark the yoloplugin's output as network's output, and unmark the original output), then build the engine, and run inference.

Feature:

    1:Use TensorRT's new plugin interface IPluginV2Ext to implement the yolo plugin.
    2:Run on the current newest TensorRT version 6.0.1.5.
    3:test on NVIDIA 1060 at 37 fps in f32 mode and 77 fps in int8 mode.
    4.include the post process such as nms to get the final detection result.
    
Build:

    1.mkdir build && cd build
    2.cmake .. -DTensorRT-ROOT=/path/to/tensorrt/root -DCUDA-ROOT=path/to/cuda/root -DOpenCV-ROOT=/path/to/opencv/root
    3.make -j8
    Note: I use opencv4.0.0, If you use opencv2.x or opencv3.x, please edit the ProcessDependency.cmake about the opencv lib config , beacuse may some so is not exsit in older opencv version, and opencv include config from: include_directories${OpenCV-ROOT}/include/opencv4) to include_directories(${OpenCV-ROOT}/include)
Requirment:

    Install the TensorRT6.0.1.5,(I use CUDA10.1 version, but other cuda version's TensorRT should also work well).

Problem need to solve:
   
     I have Solved this problem, more detail jump to this issue:https://github.com/NVIDIA/TensorRT/issues/178(after add the yolo plugin in c++, it always failed to serialize the builded engine, I think this is a bug of the tensorrt's
     binary Components...currently I add a python script to serialize the trt engine and then use it in c++ for deploy.)

Reference:

    1:https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps (restructure branch, not master branch)
    2:https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide
