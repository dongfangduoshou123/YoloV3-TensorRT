#!/usr/bin/env python2

from __future__ import print_function
from __future__ import print_function
import numpy as np
import ctypes
ctypes.cdll.LoadLibrary('/opt/caffe2_yolov3/TensorRT-6.0.1.5/lib/libnvinfer_plugin.so')
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()
logger = trt.Logger(trt.Logger.INFO)
ctypes.cdll.LoadLibrary('./libyoloPlugin.so')


def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    def build_engiinewithyoloplugin():
        plugin_creator = get_plugin_creator('Yolo_TRT')
        if plugin_creator == None:
            print('Plugin not found. Exiting')
            exit()
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                                     TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('add the yolo plugin to original network')
            tensor1 = network.get_output(0)
            tensor2 = network.get_output(1)
            tensor3 = network.get_output(2)
            ytensor1 = network.add_plugin_v2(
                [tensor1],
                plugin_creator.create_plugin('Yolo_TRT', trt.PluginFieldCollection([
                    trt.PluginField("numclass", np.array(7, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("stride", np.array(32, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("gridesize", np.array(13, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("numanchors", np.array(3, dtype=np.int32), trt.PluginFieldType.INT32)
                ]))
            ).get_output(0)

            ytensor2 = network.add_plugin_v2(
                [tensor2],
                plugin_creator.create_plugin('Yolo_TRT', trt.PluginFieldCollection([
                    trt.PluginField("numclass", np.array(7, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("stride", np.array(16, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("gridesize", np.array(26, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("numanchors", np.array(3, dtype=np.int32), trt.PluginFieldType.INT32)
                ]))
            ).get_output(0)

            ytensor3 = network.add_plugin_v2(
                [tensor3],
                plugin_creator.create_plugin('Yolo_TRT', trt.PluginFieldCollection([
                    trt.PluginField("numclass", np.array(7, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("stride", np.array(8, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("gridesize", np.array(52, dtype=np.int32), trt.PluginFieldType.INT32),
                    trt.PluginField("numanchors", np.array(3, dtype=np.int32), trt.PluginFieldType.INT32)
                ]))
            ).get_output(0)

            network.mark_output(ytensor1)
            network.mark_output(ytensor2)
            network.mark_output(ytensor3)
            network.unmark_output(tensor1)
            network.unmark_output(tensor2)
            network.unmark_output(tensor3)

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine


    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engiinewithyoloplugin()



get_engine("yolov3.onnx", "yolov3.trt")
