from models.retina import Retina
from detect import load_model
from data import cfg_mnet
import numpy as np
import cv2
import torch
import onnx


input_h = 640
input_w = 640
torch_model_path = 'weights/ccpd_custom/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101.pth'
ONNX_FILE_PATH = 'weights/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101.onnx'
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_LINEAR ####################### cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # resize_scale = 640 / min(img.shape[0], img.shape[1])
    # img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
    img = resizeAndPad(img, (input_h, input_w)) ################################################# 1160, 720
    cv2.imshow('asd', img)
    cv2.waitKey(0)
    print(img.shape)
    # img = cv2.resize(img, (640, 640))
    img = np.float32(img)
    print(img.shape)
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    print(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    device = torch.device("cuda:0")
    img = img.to(device)
    return img


model_fp32 = Retina(cfg_mnet, phase='export') # phase can be any value except "train"
model_fp32 = load_model(model_fp32, torch_model_path, True)

model_fp32.eval()
model_fp32.cuda()

# input= torch.rand((1, 3, 1160, 720)).cuda() # n c h w

# input = preprocess("prepare_data/ccpd_dataset/train/1698799233428.jpg").cuda()
input = preprocess("E:/plate_recognition/CCPD_custom/img_pos/02-6_2-365&355_557&442-553&442_365&422_369&355_557&375-0_0_33_26_25_23_30-16-17.jpg").cuda()
output = model_fp32(input)


torch.onnx.export(model_fp32, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True, opset_version=11,
                  # dynamic_axes={
                  #         'input': {2: 'input_height', 3: 'input_width'},
                  #         'output': {2: 'output_height', 3: 'output_width'}}

                  )


model = onnx.load(ONNX_FILE_PATH)
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)
