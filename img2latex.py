# ****************************************************************************************************************
# ****************************************************************************************************************
# -*-coding:UTF-8-*-
# @Company    :   ××××
# @File       :   img2latex.py
# @Author     :   ytf
# @Date       :   2021/11/15  10:18
# @Tool       :   vscode
# ****************************************************************************************************************
# ****************************************************************************************************************
from config import AppConfig
from numpy.core.shape_base import hstack
from img2latex_dataset.dataset import test_transform
from PIL import Image
import os
# import logging
import yaml
import numpy as np
import torch
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from img2latex_models import get_model
from img2latex_utils.utils import *



# *****************************************************************************************************************
#                                               对输入的图片进行大小调整
# *****************************************************************************************************************
def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img

# *****************************************************************************************************************
#                                                    初始化相关参数
# *****************************************************************************************************************
def initialize(arguments=None):
    # logging.getLogger().setLevel(logging.FATAL)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if arguments is None:
        arguments = Munch({'config': AppConfig.APP_ROOT_DIR + '/img2latex_settings/config.yaml', 'checkpoint': AppConfig.APP_ROOT_DIR + '/checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.update(**vars(arguments))
    args.wandb = False
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = get_model(args)
    model.load_state_dict(torch.load(AppConfig.img2latex_weights_path, map_location=device))
    # 对输入图片进行大小调整的模型进行初始化
    if 'image_resizer.pth' in os.listdir(os.path.dirname(AppConfig.img2latex_weights_path)) and not arguments.no_resize:
        image_resizer = ResNetV2(layers=[2, 3, 3], 
                                 num_classes=22, 
                                 global_pool='avg', 
                                 in_chans=1, 
                                 drop_rate=.05,
                                 preact=True, 
                                 stem_type='same', 
                                 conv_layer=StdConv2dSame).to(device)
        # 加载模型
        image_resizer.load_state_dict(torch.load(AppConfig.img2latex_resizer_path, map_location=device))
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=AppConfig.tokenizer)
    return args, model, image_resizer, tokenizer

args, *objs = initialize()

# *****************************************************************************************************************
#                                                    由图片获取latex
# *****************************************************************************************************************
def get_latex(args, model, image_resizer, tokenizer, img=None):
    encoder, decoder = model.encoder, model.decoder
    if type(img) is bool:
        img = None
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    if image_resizer is not None and args.no_resize:
        with torch.no_grad():
            input_image = pad(img).convert('RGB').copy()
            r, w = 1, input_image.size[0]
            for _ in range(10):
                img = pad(minmax_size(input_image.resize((w, int(input_image.size[1]*r)), Image.BILINEAR if r > 1 else Image.LANCZOS), args.max_dimensions, args.min_dimensions))
                t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                w = (image_resizer(t.to(args.device)).argmax(-1).item()+1)*32
                if (w == img.size[0]):
                    break
                r = w/img.size[0]
    else:
        img = np.array(pad(img).convert('RGB'))
        t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), 
                               args.max_seq_len,
                               eos_token=args.eos_token, 
                               context=encoded.detach(), 
                               temperature=AppConfig.temperature)
        pred = post_process(token2str(dec, tokenizer)[0])
        print("pred =", pred)
        # 去除公式左右两侧的left和right,避免公式截取不完整时导致识别错误
        pred = pred.replace('\\right','')
        pred = pred.replace('\\left','')
        pred = pred.replace('\\mathcal','')
        pred = pred.replace('\\mathbf','')
        pred = pred.replace('\\Longrightarrow','=')
        pred = pred.replace('\\equiv','=')
        pred = pred.replace('\\circ','\cdot')
        pred = pred.replace('\\bf','')
        pred = pred.replace('\\-','-')
        pred = pred.replace('\\^','^')
        pred = pred.replace('\\uparrow','|')
    
        pred = pred.rstrip()
       
        # 查找'\'字符
        fanxg = pred.count("\\")
        c=-1
        num = '0123456789'
        ABC = 'ABCDEFG'
        need_del_index = []
        for i in range(fanxg):
            # 获取指定字符的所有索引
            c = pred.find("\\", c + 1, len(pred))
            # 判断"\"后面跟的是否为数字,若为数字则将“\”删除
            if pred[c+1] in num or pred[c+1] in ABC:
                need_del_index.append(c)
        if len(need_del_index) != 0:
            for del_index in range(len(need_del_index)):
                num = need_del_index[del_index]
                pred = pred[:num-del_index] + pred[num-del_index+1:]  # 使用切片去除指定字符
                
        # print("\n")
        # print(pred)
        # print("\n")

    return pred

# *****************************************************************************************************************
#                                                  输入图片，来获取Latex
# *****************************************************************************************************************
def image2latex(imglist):
    formula_list = []
    for img_index,img in enumerate(imglist):
        imgh, imgw = img.shape[:2]
        if imgh==0 and imgw==0:
            continue
        if 60 > imgh >= 27 and imgw > 27:
            re_img = cv2.resize(img,(int(imgw/(imgh/26)), 26))
        elif imgh >= 60 and imgw > 27:
            re_img = cv2.resize(img,(int(imgw/(imgh/50)), 50))

        else:
            re_img = img
    
        # re_img = gama_change(re_img)                                  # 伽马增强

        # 灰度处理
        gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)
        # 图片开闭运算
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
        re_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        re_img = extern_pix(re_img, 1,1,2,2)

        re_img = Image.fromarray(re_img)  
        ret = get_latex(args, *objs, img=re_img)
        formula_list.append(ret)

    return formula_list
# *****************************************************************************************************************
#                                                        补像素
# *****************************************************************************************************************
def extern_pix(img, top_value, bottom_value, left_value, right_value):
    """
    img: 输入图片
    top_value: 输入图片上边需要补的像素行数
    bottom_value: 输入图片下边需要补的像素行数
    left_value: 输入图片左边需要补的像素行数
    right_value: 输入图片右边需要补的像素行数
    """
    # 四周补白
    white = [255, 255, 255]
    # 补像素
    extentImg = cv2.copyMakeBorder(img, top_value, bottom_value,
                                    left_value, right_value,
                                    cv2.BORDER_CONSTANT, value=white)
    return extentImg
# *****************************************************************************************************************
#                                                      图片锐化
# *****************************************************************************************************************
def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)

    return dst
# *****************************************************************************************************************
#                                                      伽马增强
# *****************************************************************************************************************
def gama_change(img):
    ret_img = 255*np.power(img/255, 1.1)
    ret_img = np.around(ret_img)
    ret_img[ret_img>255] = 255
    ret_img = ret_img.astype(np.uint8)
    
    return ret_img
# *****************************************************************************************************************
#                                                        主函数
# *****************************************************************************************************************
if __name__ == "__main__":
    
    # img = APP_ROOT_DIR + '/0.jpg'
    img = AppConfig.APP_ROOT_DIR + '/images/7.png'
    try: 
        img = Image.open(img)
        get_latex(args, *objs, img=img) 
    except KeyboardInterrupt:
        pass
