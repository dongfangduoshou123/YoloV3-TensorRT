# YoloV3-TensorRT
Run YoloV3 with the newest TensorRT6.0 at 37 fps on  NVIIDIA 1060.

Feature:
    1:Use TensorRT's new plugin interface IPluginV2Ext to implement the yolo plugin.
    2:Run on the current newest TensorRT version 6.0.1.5.
    3:test on NVIDIA 1060 at 37 fps.
    4.include the post process such as nms to get the final detection result.

Reference:
    https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps (restructure branch, not master branch)
    https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide
