import sys
import os
import cv2

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from PIL import Image

from utils.model import ModelData
from utils import common

# ../../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir
    )
)
from utils.common import HostDeviceMem
import utils.calibrator as calibrator
import struct

def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.
    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"images": np.float32, "output": np.float32}

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


class ModelData(object):
    MODEL_PATH = "yolov5l.onnx"
    INPUT_SHAPE = (3, 640, 640)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_onnx(onnx_model_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_engine(network, config)



def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()




def build_engine(onnx_model_path, trt_logger, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1, silent=False, save=''):

    with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.OnnxParser(network, trt_logger) as parser:


        with open(onnx_model_path,'rb') as model :
            parser.parse(model.read())

        builder.max_batch_size = batch_size
        builder_config = builder.create_builder_config()
        builder_config.max_workspace_size = 1 << 20

        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        elif trt_engine_datatype == trt.DataType.INT8:
            builder.fp16_mode = True
            builder.int8_mode = True
            builder.int8_calibrator = calibrator.Yolov5EntropyCalibrator(data_dir=calib_dataset, cache_file='INT8CacheFile')
        
        if save is not None : 
            serialized_engine = builder.build_serialized_network(network, builder_config)
            with open(save + '/yolov5l.engine', 'wb') as f:
                f.write(serialized_engine)

        #parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        #parser.register_output("MarkOutput_0")
        #parser.parse(onnx_model_path, network)

        if not silent:
            print("Building TensorRT engine. This may take few minutes.")

        return builder.build_engine(network, builder_config)

def save_engine(engine, engine_dest_path):
    print('Engine:', engine)
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine