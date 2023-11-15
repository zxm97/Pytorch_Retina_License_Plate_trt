
## Environment
Windows 10

CUDA 11.7

cuDNN 8.6.0.163

TensorRT-8.5.1.7



## Requirements
numpy==1.21.6

onnx==1.14.1

opencv_python==4.8.1.78

opencv_python_headless==4.8.1.78

Pillow==9.5.0

pycuda==2021.1+cuda115

tensorrt==8.5.1.7

torch==1.13.1+cu116

torchvision==0.14.1+cu116


## How to use
### for retinaface (fpn_num=3):
#### step 1
run export_onnx.py to export onnx model
#### step 2
run export_trt.py to convert onnx to trt engine
#### step 3
run detect_trt.py to detect images

### for retinaface (fpn_num=2, reduced 1 FPN layer and anchor density):
#### step 1
run export_onnx_reduced.py to export onnx model
#### step 2
run export_trt_reduced.py to convert onnx to trt engine
#### step 3
run detect_trt_reduced.py to detect images

### for tensorrtx (fpn_num=3):
run gen_wts_for_tensorrtx.py to get weight map for tensorrtx


## References

https://github.com/gm19900510/Pytorch_Retina_License_Plate

https://github.com/1996scarlet/faster-mobile-retinaface

https://github.com/wang-xinyu/tensorrtx
