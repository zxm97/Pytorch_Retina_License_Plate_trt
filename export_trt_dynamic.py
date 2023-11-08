
import os
import numpy as np

import common
import tensorrt as trt

import pycuda.driver as cuda

input_w = 640 #720
input_h = 640 #1160
# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from image_batcher import ImageBatcher
# logger to capture errors, warnings, and other information during the build and inference phases

TRT_LOGGER = trt.Logger()


precision = "fp16" # choices=["fp16", "int8"]
ONNX_FILE_PATH = 'weights/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101_dynamic.onnx'
ENGINE_FILE_PATH = 'weights/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101' + '_' + precision + '_dynamic.trt'
# calib_input = "./prepare_data/ccpd_dataset/train/" # The directory holding images to use for calibration
calib_input = "E:/plate_recognition/CCPD_custom/img_pos/"
# calib_input = "E:/plate_recognition/CCPD2020/ccpd_green/train/"
calib_cache = "./calibration.cache" # The file path for INT8 calibration cache to use
calib_num_images = 5700 # The maximum number of images to use for calibration
calib_batch_size = 32





# def build_engine(onnx_file_path):
#     # initialize TensorRT engine and parse ONNX model
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
#
#     profile = builder.create_optimization_profile()####################
#
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#     print('Completed parsing of ONNX file')
#     # allow TensorRT to use up to 1GB of GPU memory for tactic selection
#
#     builder_config = builder.create_builder_config()
#     builder_config.add_optimization_profile(profile) ########################
#     builder_config.max_workspace_size = 1 << 30
#
#     # we have only one image in batch
#     builder.max_batch_size = 1
#     # use FP16 mode if possible
#
#
#
#     inputs = [network.get_input(i) for i in range(network.num_inputs)]
#
#     if precision in ["fp16", "int8",]:
#         if not builder.platform_has_fast_fp16:
#             print("FP16 is not supported natively on this platform/device")
#         builder_config.set_flag(trt.BuilderFlag.FP16)
#     if precision in ["int8",]:
#         if not builder_config.platform_has_fast_int8:
#             print("INT8 is not supported natively on this platform/device")
#         builder_config.set_flag(trt.BuilderFlag.INT8)
#         builder_config.int8_calibrator = EngineCalibrator(calib_cache)
#         if calib_cache is None or not os.path.exists(calib_cache):
#             calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
#             calib_dtype = trt.nptype(inputs[0].dtype)
#             builder_config.int8_calibrator.set_image_batcher(
#                 ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
#                              exact_batches=True, shuffle_files=True))
#
#
#
#
#     # if builder.platform_has_fast_fp16:
#     #     builder_config.set_flag(trt.BuilderFlag.FP16)
#     # generate TensorRT engine optimized for the target platform
#     print("Building {} Engine...".format(precision))
#     # engine = builder.build_cuda_engine(network)
#     # context = engine.create_execution_context()
#
#     engine  = builder.build_engine(network, builder_config)
#     context = engine.create_execution_context()
#
#     print("Completed creating Engine")
#
#     return engine, context
#
# def main_not_used():
#     # initialize TensorRT engine and parse ONNX model
#     engine, context = build_engine(ONNX_FILE_PATH)
#     # get sizes of input and output and allocate memory required for input data and for output data
#     for binding in engine:
#         if engine.binding_is_input(binding):  # we expect only one input
#             input_shape = engine.get_binding_shape(binding)
#             input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize     # in bytes
#             device_input = cuda.mem_alloc(input_size)
#         else:  # and one output
#             output_shape = engine.get_binding_shape(binding)
#             # create page-locked memory buffers (i.e. won't be swapped to disk)
#             host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
#             device_output = cuda.mem_alloc(host_output.nbytes)
#
#     # Create a stream in which to copy inputs/outputs and run inference.
#     stream = cuda.Stream()
#     # preprocess input data
#     host_input = np.array(preprocess("prepare_data/ccpd_dataset/train/1698799233428.jpg").numpy(), dtype=np.float32, order='C')
#     cuda.memcpy_htod_async(device_input, host_input, stream)
#     # run inference
#     context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
#     cuda.memcpy_dtoh_async(host_output, device_output, stream)
#     stream.synchronize()
#     # postprocess results
#     output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
#     # postprocess(output_data)

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch_efficientdet(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            print("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            print("Finished calibration batches")
            return None

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, = next(self.batch_generator)
            print("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            print("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 30  # 2GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, input_h, input_w]




            #-----------------------------
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            # print('--------------',list(inputs[0].shape[:]))
            if precision in ["fp16", "int8",]:
                if not builder.platform_has_fast_fp16:
                    print("FP16 is not supported natively on this platform/device")
                config.set_flag(trt.BuilderFlag.FP16)
            if precision in ["int8",]:
                if not builder.platform_has_fast_int8:
                    print("INT8 is not supported natively on this platform/device")
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = EngineCalibrator(calib_cache)
                if calib_cache is None or not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                     exact_batches=True, shuffle_files=True))


            # #-----------------------------



            print("Completed parsing of ONNX file")
            print("Building an {} engine from file {}; this may take a while...".format(precision, onnx_file_path))
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

get_engine(ONNX_FILE_PATH,ENGINE_FILE_PATH)



