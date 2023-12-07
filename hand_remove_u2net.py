# *****************************************************************************************************************
# -*-coding:UTF-8-*-
# @Company    :   ××××
# @File       :   hand_remove_u2net.py
# @Author     :   ytf
# @Date       :   2021/3/31  16:32
# @Tool       :   Pycharm
# ****************************************************************************************************************


# *****************************************************************************************************************
#                                                     导入相关库
# *****************************************************************************************************************
import cv2
import numpy as np
import datetime
import time
import torch
from threshold import u2net_to_handremve_threshold as U2netThreshold
import logging
from PIL import Image
logger = logging.getLogger()


# *****************************************************************************************************************
#                                             水平分割图片(1刀，得到2张图片）
# *****************************************************************************************************************
def cut_image_h(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    height, width = image.shape[:2]
    item_width = width
    item_height = int(height/count)
    box_list = []
    for i in range(0,count):
        if i == 0:   
            box = (0,i*item_height,item_width,(i+1)*item_height+20)

        elif i == 1:
            box = (0,i*item_height-20,item_width,(i+1)*item_height)

        box_list.append(box)  
        
    image_list_h = [Image.fromarray(image).crop(box) for box in box_list]

    # self.save_images(image_list)
    return image_list_h

# *****************************************************************************************************************
#                                             水平分割图片(2刀，得到3张图片）
# *****************************************************************************************************************
def cut_image_h_2(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    height, width = image.shape[:2]
    item_width = width
    item_height = int(height/count)
    box_list = []
    for i in range(0,count):
        if i == 0:   
            box = (0,i*item_height,item_width,(i+1)*item_height+20)

        elif i == 1:
            box = (0,i*item_height-20,item_width,(i+1)*item_height+20)

        elif i == 2:
            box = (0,i*item_height-20,item_width,(i+1)*item_height)

        box_list.append(box)  
        
    image_list_h = [Image.fromarray(image).crop(box) for box in box_list]

    return image_list_h

# *****************************************************************************************************************
#                                             水平分割图片(3刀，得到4张图片）
# *****************************************************************************************************************
def cut_image_h_3(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    height, width = image.shape[:2]
    item_width = width
    item_height = int(height/count)
    box_list = []
    for i in range(0,count):
        if i == 0:   
            box = (0,i*item_height,item_width,(i+1)*item_height+20)
        elif i == 1:
            box = (0,i*item_height-20,item_width,(i+1)*item_height+20)
        elif i == 2:
            box = (0,i*item_height-20,item_width,(i+1)*item_height+20)
        elif i == 3:
            box = (0,i*item_height-20,item_width,(i+1)*item_height)

        box_list.append(box)  
        
    image_list_h = [Image.fromarray(image).crop(box) for box in box_list]

    return image_list_h

# *****************************************************************************************************************
#                                             垂直分割图片(1刀，得到2张图片）
# *****************************************************************************************************************
def cut_image_v(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    width, height = image.size
    item_width = int(width / count) 
    item_height = height
    box_list = []

    for i in range(0, count):
        if i == 0:
            box = (i*item_width,0,(i+1)*item_width+20,item_height)
        elif i ==1:
            box = (i*item_width-20,0,(i+1)*item_width,item_height)
        box_list.append(box)
 
    image_list_v = [image.crop(box) for box in box_list]

    return image_list_v

# *****************************************************************************************************************
#                                             垂直分割图片(2刀，得到3张图片）
# *****************************************************************************************************************
def cut_image_v_2(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    width, height = image.size
    item_width = int(width / count) 
    item_height = height
    box_list = []
    for i in range(0, count):
        if i == 0:
            box = (i*item_width,0,(i+1)*item_width+20,item_height)
        elif i ==1:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==2:
            box = (i*item_width-20,0,(i+1)*item_width,item_height)
        box_list.append(box)

    image_list_v = [image.crop(box) for box in box_list]

    return image_list_v

# *****************************************************************************************************************
#                                             垂直分割图片(3刀，得到4张图片）
# *****************************************************************************************************************
def cut_image_v_3(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    width, height = image.size
    item_width = int(width / count) 
    item_height = height
    box_list = []
    for i in range(0, count):
        if i == 0:
            box = (i*item_width,0,(i+1)*item_width+20,item_height)
        elif i ==1:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==2:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==3:
            box = (i*item_width-20,0,(i+1)*item_width,item_height)
        box_list.append(box)

    image_list_v = [image.crop(box) for box in box_list]

    return image_list_v

# *****************************************************************************************************************
#                                             垂直分割图片(4刀，得到5张图片）
# *****************************************************************************************************************
def cut_image_v_4(image, count):
    """
    image: 输入图片
    count: 需要切割的数量
    """
    width, height = image.size
    item_width = int(width / count) 
    item_height = height
    box_list = []
    for i in range(0, count):
        if i == 0:
            box = (i*item_width,0,(i+1)*item_width+20,item_height)
        elif i ==1:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==2:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==3:
            box = (i*item_width-20,0,(i+1)*item_width+20,item_height)
        elif i ==4:
            box = (i*item_width-20,0,(i+1)*item_width,item_height)
        box_list.append(box)

    image_list_v = [image.crop(box) for box in box_list]

    return image_list_v

# *****************************************************************************************************************
#                                                       补像素
# *****************************************************************************************************************
def externPix(img, top_value, bottom_value, left_value, right_value):
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
#                                                      图片去手写
# *****************************************************************************************************************
def img_hand_remove(rotate_res, inputImg):
    """
    rotate_res: 旋转后的图片
    inputImg: 需要去手写的图片
    """
    logger.info("[handleOneImg] start")
    # 将输入图片变为灰度图
    img = cv2.cvtColor(inputImg, cv2.COLOR_RGB2GRAY)
    # 定义剪切后的图片列表
    final_cut_imglist = []
    # 转换图像格式
    image = Image.fromarray(img)

    # 对输入的图片进行判定
    img_H, img_W = img.shape[:2]
    if img_H == 0 or img_W == 0:
        return -1
    
    # *****************************************切割成为2张图******************************************
    # 垂直切割一刀
    if img_W>img_H and 2000 > img_W > 1000 and ((img_W//img_H)>=2 or float(img_W/img_H)>1.5):
        cut_v_imglist = []
        cut_v_imglist = cut_image_v(image, 2)

        final_cut_imglist.append(cut_v_imglist[0])
        final_cut_imglist.append(cut_v_imglist[1])

    
    # 水平切割一刀
    elif img_H>img_W and 2000 > img_H > 1000 and ((img_H//img_W)>=2 or float(img_H/img_W)>1.5):
        cut_v_imglist = []
        cut_h_imglist = cut_image_h(img, 2)

        final_cut_imglist.append(cut_h_imglist[0])
        final_cut_imglist.append(cut_h_imglist[1])

    

    # *****************************************切割成为3张图******************************************
    # 水平切2刀，得到三张图片
    elif img_H>img_W and 3000>img_H>=2000 and (img_H//img_W)>=3:
        cut_h_imglist  = []
        cut_h_imglist = cut_image_h_2(img, 3)
        final_cut_imglist.append(cut_h_imglist[0])
        final_cut_imglist.append(cut_h_imglist[1])
        final_cut_imglist.append(cut_h_imglist[2])

    # 垂直切割2刀，得到三张图片
    elif img_W>img_H and 3000>img_W>=2000 and (img_W//img_H)>=3:
        cut_v_imglist  = []
        cut_v_imglist = cut_image_v_2(image, 3)
        final_cut_imglist.append(cut_v_imglist[0])
        final_cut_imglist.append(cut_v_imglist[1])
        final_cut_imglist.append(cut_v_imglist[2])
        
    # *****************************************切割成为4张图******************************************
    # 切割成为4张图
    elif 2000 > img_H >= 1000 and ((img_H//img_W)==1 or (img_W//img_H)==1):
        # 横切1刀
        imglist1= cut_image_h(img, 2)
        cut_v_imglist = []
        for im in imglist1:
            # 竖切1刀
            cut_v_imglist.append(cut_image_v(im, 2))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])

    # *****************************************切割成为6张图******************************************
    #  切割成为6张图(横切2刀，竖切1刀)
    elif img_H>img_W and 3000 >img_H > 2000 and (img_H//img_W)==2:
        imglist1 = cut_image_h_2(img, 3)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v(im, 2))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[2][0])
        final_cut_imglist.append(cut_v_imglist[2][1])

    #  切割成为6张图(横切1刀，竖切2刀)
    elif img_H<img_W and 3000 >img_W > 2000 and (img_W//img_H)==2:
        imglist1 = cut_image_h(img, 2)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v_2(im, 3))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[0][2])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[1][2])


    # ****************************************切割成为8张图*******************************************
    # 切割成为8张图(横切3刀，竖切1刀)
    elif img_H>img_W and 4608> img_H >= 3000 and (img_H//img_W)==2:
        imglist1 = cut_image_h_3(img, 4)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v(im, 2))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[2][0])
        final_cut_imglist.append(cut_v_imglist[2][1])
        final_cut_imglist.append(cut_v_imglist[3][0])
        final_cut_imglist.append(cut_v_imglist[3][1])
    # 切割成为8张图(横切1刀，竖切3刀)
    elif img_H<img_W and 4608> img_W >= 3000 and (img_W//img_H)==2:
        imglist1 = cut_image_h(img, 2)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v_3(im, 4))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[0][2])
        final_cut_imglist.append(cut_v_imglist[0][3])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[1][2])
        final_cut_imglist.append(cut_v_imglist[1][3])

    # *****************************************切割成为9张图******************************************
    # 切割成为9张图，横切2刀，竖切2刀
    elif 3456>img_H>2000 and ((img_H//img_W)==1 or (img_W//img_H)==1):
        imglist1 = cut_image_h_2(img, 3)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v_2(im, 3))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[0][2])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[1][2])
        final_cut_imglist.append(cut_v_imglist[2][0])
        final_cut_imglist.append(cut_v_imglist[2][1])
        final_cut_imglist.append(cut_v_imglist[2][2])

    # *************************************** *切割成为12张图*****************************************
    # 切割成为12张图（横切3张，竖切2刀）
    elif img_H > img_W and img_W >= 2700 and img_H>3456:
        imglist1 = cut_image_h_3(img, 4)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v_2(im, 3))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[0][2])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[1][2])
        final_cut_imglist.append(cut_v_imglist[2][0])
        final_cut_imglist.append(cut_v_imglist[2][1])
        final_cut_imglist.append(cut_v_imglist[2][2])
        final_cut_imglist.append(cut_v_imglist[3][0])
        final_cut_imglist.append(cut_v_imglist[3][1])
        final_cut_imglist.append(cut_v_imglist[3][2])
    # 切割成为12张图（横切2张，竖切3刀）
    elif img_H<img_W and img_H>= 2700 and img_W>3456:
        imglist1 = cut_image_h_2(img, 3)
        cut_v_imglist = []
        final_cut_imglist = []
        for im in imglist1:
            cut_v_imglist.append(cut_image_v_3(im, 4))
        final_cut_imglist.append(cut_v_imglist[0][0])
        final_cut_imglist.append(cut_v_imglist[0][1])
        final_cut_imglist.append(cut_v_imglist[0][2])
        final_cut_imglist.append(cut_v_imglist[0][3])
        final_cut_imglist.append(cut_v_imglist[1][0])
        final_cut_imglist.append(cut_v_imglist[1][1])
        final_cut_imglist.append(cut_v_imglist[1][2])
        final_cut_imglist.append(cut_v_imglist[1][3])
        final_cut_imglist.append(cut_v_imglist[2][0])
        final_cut_imglist.append(cut_v_imglist[2][1])
        final_cut_imglist.append(cut_v_imglist[2][2])
        final_cut_imglist.append(cut_v_imglist[2][3])
    
    # ******************************************保留原图大小******************************************* 
    # 保留原图大小
    else:
        final_cut_imglist.append(image)

    # *****************************************开始进行去手写******************************************
    # 1、将剪切后的图片进行补像素，然后送给去手写模型，得到干净的黑白图
    img_list = []
    i = 1
    torch.cuda.empty_cache()
    for cut_img in final_cut_imglist:
        # 将图片转化为np格式
        img_need = np.asarray(cut_img)
        cut_img_h, cut_img_w = img_need.shape[:2]

        # 初始化需要补像素的值
        top_value = 0
        bottom_value = 0
        left_value = 0 
        right_value = 0
        # 判断补像素方式
        if cut_img_h>=cut_img_w:
            left_value = int((cut_img_h-cut_img_w)/2)
            right_value = left_value
            img_need = externPix(img_need, top_value, bottom_value, left_value, right_value)
  
        elif cut_img_h < cut_img_w:
            top_value = int((cut_img_w-cut_img_h)/2)
            bottom_value = top_value
            img_need = externPix(img_need, top_value, bottom_value, left_value, right_value)
        # cv2.imwrite("ddfdf.jpg", img_need)

        # 进行去手写和二值化操作
      
        ret_img = U2netThreshold.threshold(img_need)
        torch.cuda.empty_cache()
        ret_img_h, ret_img_w = ret_img.shape[:2]

        # 去除补加的多余像素
        box = (left_value, top_value, ret_img_w-right_value, ret_img_h-bottom_value)
        ret_img = np.array(Image.fromarray(ret_img).crop(box)) 

        # 伽马增强
        ret_img = 255*np.power(ret_img/255, 1.2)
        ret_img = np.around(ret_img)
        ret_img[ret_img>255] = 255
        ret_img = ret_img.astype(np.uint8)
        # 图像锐化
        ret_img = custom_blur_demo(ret_img)
        
        # 经过去手写处理后，手写体部分字体有的没有被才擦除干净，进行自适应阈值二值化操作
        # ret_img = cv2.adaptiveThreshold(ret_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 55)

        img_list.append(ret_img)  
    
    
    # 2、将切割的图片进行重组
    # 原图
    if len(final_cut_imglist) == 1:
        # 将原图与结果图拼在一起
        ret_img1 = cv2.cvtColor(ret_img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rotate_res = cv2.resize(rotate_res, (img.shape[1], img.shape[0]))
        if img_H > img_W:
            # final_ret_img = np.hstack((rotate_res, inputImg, ret_img1)) # 将原图、校正图和结果图拼接在一起
            final_ret_img = np.hstack((rotate_res, ret_img1)) # 将原图和结果图拼接在一起
        else:
            # final_ret_img = np.vstack((rotate_res, inputImg, ret_img1)) # 将原图、校正图和结果图拼接在一起
            final_ret_img = np.vstack((rotate_res, ret_img1)) # 将原图和结果图拼接在一起
        
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cv2.imwrite("/home/ytfwork/output/" + t + "_ret.jpg", final_ret_img)
 
        return ret_img

    # 剪切后得到两张图
    elif len(final_cut_imglist) == 2:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        img1 = img_list[0]
        img2 = img_list[1]
        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        # 判断是否为横向切割为两张图
        if img_H > img_W:
            # 剪切
            box1 = (0, 0, img1_w, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (0, 20, img2_w, img2_h)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)

            # 组合
            ret_img = np.vstack((new_img_list[0], new_img_list[1]))

            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

        # 纵向切割两张图
        else :
            # 剪切
            box1 = (0, 0, img1_w-20, img1_h)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w, img2_h)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)

            # 组合
            ret_img = np.hstack((new_img_list[0], new_img_list[1]))

            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

    # 剪切后得到3张图
    elif len(final_cut_imglist) == 3:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        # 判断是否为横向切割为3张图
        if img_H > img_W:
            # 剪切
            box1 = (0, 0, img1_w, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (0, 20, img2_w, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (0, 20, img3_w, img3_h)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)

            # 组合
            ret_img = np.vstack((new_img_list[0], new_img_list[1], new_img_list[2]))
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

        # 纵向切割3张图
        else :
            # 剪切
            box1 = (0, 0, img1_w-20, img1_h)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w-20, img2_h)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (20, 0, img3_w, img3_h)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)

            # 组合
            ret_img = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2]))
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

    # 剪切成4张图
    elif len(final_cut_imglist) == 4:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        img4 = img_list[3]
        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        img4_h, img4_w = img4.shape[:2]

        # 剪切
        box1 = (0, 0, img1_w-20, img1_h-20)
        img1 = np.array(Image.fromarray(img1).crop(box1)) 
        box2 = (20, 0, img2_w, img2_h-20)
        img2 = np.array(Image.fromarray(img2).crop(box2)) 
        box3 = (0, 20, img3_w-20, img3_h)
        img3 = np.array(Image.fromarray(img3).crop(box3)) 
        box4 = (20, 20, img4_w, img4_h)
        img4 = np.array(Image.fromarray(img4).crop(box4)) 

        # 将剪切后的图片放到新的列表中
        new_img_list.append(img1)
        new_img_list.append(img2)
        new_img_list.append(img3)
        new_img_list.append(img4)

        # 组合
        ret_img1 = np.hstack((new_img_list[0], new_img_list[1]))
        ret_img2 = np.hstack((new_img_list[2], new_img_list[3]))
        ret_img = np.vstack((ret_img1, ret_img2))
        ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
        
    # 剪切成6张图
    elif len(final_cut_imglist) == 6:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        img4 = img_list[3]
        img5 = img_list[4]
        img6 = img_list[5]

        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        img4_h, img4_w = img4.shape[:2]
        img5_h, img5_w = img5.shape[:2]
        img6_h, img6_w = img6.shape[:2]

        # 剪切
        if img_H > img_W:
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (0, 20, img3_w-20, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (20, 20, img4_w, img4_h-20)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (0, 20, img5_w-20, img5_h)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w, img6_h)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)

            # 组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1]))
            ret_img2 = np.hstack((new_img_list[2], new_img_list[3]))
            ret_img3 = np.hstack((new_img_list[4], new_img_list[5]))
            ret_img = np.vstack((ret_img1, ret_img2, ret_img3))
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

        else:
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w-20, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (20, 0, img3_w, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (0, 20, img4_w-20, img4_h)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (20, 20, img5_w-20, img5_h)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w, img6_h)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)

            # 组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2]))
            ret_img2 = np.hstack((new_img_list[3], new_img_list[4], new_img_list[5]))
            ret_img = np.vstack((ret_img1, ret_img2))
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

    # 切割成为8张图
    elif len(final_cut_imglist) == 8:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        # 拿到8张图片
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        img4 = img_list[3]
        img5 = img_list[4]
        img6 = img_list[5]
        img7 = img_list[6]
        img8 = img_list[7]
        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        img4_h, img4_w = img4.shape[:2]
        img5_h, img5_w = img5.shape[:2]
        img6_h, img6_w = img6.shape[:2]
        img7_h, img7_w = img7.shape[:2]
        img8_h, img8_w = img8.shape[:2]

        # 剪切
        if img_H > img_W:
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (0, 20, img3_w-20, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (20, 20, img4_w, img4_h-20)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (0, 20, img5_w-20, img5_h-20)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w, img6_h-20)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 
            box7 = (0, 20, img7_w-20, img7_h)
            img7 = np.array(Image.fromarray(img7).crop(box7)) 
            box8 = (20, 20, img8_w, img8_h)
            img8 = np.array(Image.fromarray(img8).crop(box8)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)
            new_img_list.append(img7)
            new_img_list.append(img8)

            # 组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1]))
            ret_img2 = np.hstack((new_img_list[2], new_img_list[3]))
            ret_img3 = np.hstack((new_img_list[4], new_img_list[5]))
            ret_img4 = np.hstack((new_img_list[6], new_img_list[7]))
            ret_img = np.vstack((ret_img1, ret_img2, ret_img3, ret_img4))

            # 缩放
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
   
        else:
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w-20, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (20, 0, img3_w-20, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (20, 0, img4_w, img4_h-20)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (0, 20, img5_w-20, img5_h)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w-20, img6_h)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 
            box7 = (20, 20, img7_w-20, img7_h)
            img7 = np.array(Image.fromarray(img7).crop(box7)) 
            box8 = (20, 20, img8_w, img8_h)
            img8 = np.array(Image.fromarray(img8).crop(box8)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)
            new_img_list.append(img7)
            new_img_list.append(img8)

            # 组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2], new_img_list[3]))
            ret_img2 = np.hstack((new_img_list[4], new_img_list[5], new_img_list[6], new_img_list[7]))
        
            ret_img = np.vstack((ret_img1, ret_img2))

            # 缩放
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

    # 切割成为9张图
    elif len(final_cut_imglist) == 9:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        # 拿到9张图片
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        img4 = img_list[3]
        img5 = img_list[4]
        img6 = img_list[5]
        img7 = img_list[6]
        img8 = img_list[7]
        img9 = img_list[8]
        # 分别求得每部分图片的宽高
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        img4_h, img4_w = img4.shape[:2]
        img5_h, img5_w = img5.shape[:2]
        img6_h, img6_w = img6.shape[:2]
        img7_h, img7_w = img7.shape[:2]
        img8_h, img8_w = img8.shape[:2]
        img9_h, img9_w = img9.shape[:2]

        # 剪切
        
        box1 = (0, 0, img1_w-20, img1_h-20)
        img1 = np.array(Image.fromarray(img1).crop(box1)) 
        box2 = (20, 0, img2_w-20, img2_h-20)
        img2 = np.array(Image.fromarray(img2).crop(box2)) 
        box3 = (20, 0, img3_w, img3_h-20)
        img3 = np.array(Image.fromarray(img3).crop(box3)) 
        box4 = (0, 20, img4_w-20, img4_h-20)
        img4 = np.array(Image.fromarray(img4).crop(box4)) 
        box5 = (20, 20, img5_w-20, img5_h-20)
        img5 = np.array(Image.fromarray(img5).crop(box5)) 
        box6 = (20, 20, img6_w, img6_h-20)
        img6 = np.array(Image.fromarray(img6).crop(box6)) 
        box7 = (0, 20, img7_w-20, img7_h)
        img7 = np.array(Image.fromarray(img7).crop(box7)) 
        box8 = (20, 20, img8_w-20, img8_h)
        img8 = np.array(Image.fromarray(img8).crop(box8)) 
        box9 = (20, 20, img9_w, img9_h)
        img9 = np.array(Image.fromarray(img9).crop(box9)) 

        # 将剪切后的图片放到新的列表中
        new_img_list.append(img1)
        new_img_list.append(img2)
        new_img_list.append(img3)
        new_img_list.append(img4)
        new_img_list.append(img5)
        new_img_list.append(img6)
        new_img_list.append(img7)
        new_img_list.append(img8)
        new_img_list.append(img9)

        # 组合
        ret_img1 = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2]))
        ret_img2 = np.hstack((new_img_list[3], new_img_list[4], new_img_list[5]))
        ret_img3 = np.hstack((new_img_list[6], new_img_list[7], new_img_list[8]))

        ret_img = np.vstack((ret_img1, ret_img2, ret_img3))

        # 缩放
        ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
   

    # 切割成为12张图
    elif len(final_cut_imglist) == 12:
        new_img_list = []           
        # 由于剪切时，每一部分多加了20个像素，所有需要将多剪切部分去掉
        # 拿到12张图片
        img1 = img_list[0]
        img2 = img_list[1]
        img3 = img_list[2]
        img4 = img_list[3]
        img5 = img_list[4]
        img6 = img_list[5]
        img7 = img_list[6]
        img8 = img_list[7]
        img9 = img_list[8]
        img10 = img_list[9]
        img11 = img_list[10]
        img12 = img_list[11]
        # 分别求得每部分图片的宽高，用于下面的剪切
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        img4_h, img4_w = img4.shape[:2]
        img5_h, img5_w = img5.shape[:2]
        img6_h, img6_w = img6.shape[:2]
        img7_h, img7_w = img7.shape[:2]
        img8_h, img8_w = img8.shape[:2]
        img9_h, img9_w = img9.shape[:2]
        img10_h, img10_w = img10.shape[:2]
        img11_h, img11_w = img11.shape[:2]
        img12_h, img12_w = img12.shape[:2]

        # 判断切割方式
        if img_H>img_W:
            # 剪切（将多剪切出来的20个像素去掉）
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w-20, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (20, 0, img3_w, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (0, 20, img4_w-20, img4_h-20)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (20, 20, img5_w-20, img5_h-20)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w, img6_h-20)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 
            box7 = (0, 20, img7_w-20, img7_h-20)
            img7 = np.array(Image.fromarray(img7).crop(box7)) 
            box8 = (20, 20, img8_w-20, img8_h-20)
            img8 = np.array(Image.fromarray(img8).crop(box8)) 
            box9 = (20, 20, img9_w, img9_h-20)
            img9 = np.array(Image.fromarray(img9).crop(box9)) 
            box10 = (0, 20, img10_w-20, img10_h)
            img10 = np.array(Image.fromarray(img10).crop(box10)) 
            box11 = (20, 20, img11_w-20, img11_h)
            img11 = np.array(Image.fromarray(img11).crop(box11)) 
            box12 = (20, 20, img12_w, img12_h)
            img12 = np.array(Image.fromarray(img12).crop(box12)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)
            new_img_list.append(img7)
            new_img_list.append(img8)
            new_img_list.append(img9)
            new_img_list.append(img10)
            new_img_list.append(img11)
            new_img_list.append(img12)

            # 将剪切的12个小图进行组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2]))
            ret_img2 = np.hstack((new_img_list[3], new_img_list[4], new_img_list[5]))
            ret_img3 = np.hstack((new_img_list[6], new_img_list[7], new_img_list[8]))
            ret_img4 = np.hstack((new_img_list[9], new_img_list[10], new_img_list[11]))
            ret_img = np.vstack((ret_img1, ret_img2, ret_img3, ret_img4))

            # 经过剪切后，会有像素损失，所以需要resize原始图片大小
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
        else:
            # 剪切（将多剪切出来的20个像素去掉）
            box1 = (0, 0, img1_w-20, img1_h-20)
            img1 = np.array(Image.fromarray(img1).crop(box1)) 
            box2 = (20, 0, img2_w-20, img2_h-20)
            img2 = np.array(Image.fromarray(img2).crop(box2)) 
            box3 = (20, 0, img3_w-20, img3_h-20)
            img3 = np.array(Image.fromarray(img3).crop(box3)) 
            box4 = (20, 0, img4_w, img4_h-20)
            img4 = np.array(Image.fromarray(img4).crop(box4)) 
            box5 = (0, 20, img5_w-20, img5_h-20)
            img5 = np.array(Image.fromarray(img5).crop(box5)) 
            box6 = (20, 20, img6_w-20, img6_h-20)
            img6 = np.array(Image.fromarray(img6).crop(box6)) 
            box7 = (20, 20, img7_w-20, img7_h-20)
            img7 = np.array(Image.fromarray(img7).crop(box7)) 
            box8 = (20, 20, img8_w, img8_h-20)
            img8 = np.array(Image.fromarray(img8).crop(box8)) 
            box9 = (0, 20, img9_w-20, img9_h)
            img9 = np.array(Image.fromarray(img9).crop(box9)) 
            box10 = (20, 20, img10_w-20, img10_h)
            img10 = np.array(Image.fromarray(img10).crop(box10)) 
            box11 = (20, 20, img11_w-20, img11_h)
            img11 = np.array(Image.fromarray(img11).crop(box11)) 
            box12 = (20, 20, img12_w, img12_h)
            img12 = np.array(Image.fromarray(img12).crop(box12)) 

            # 将剪切后的图片放到新的列表中
            new_img_list.append(img1)
            new_img_list.append(img2)
            new_img_list.append(img3)
            new_img_list.append(img4)
            new_img_list.append(img5)
            new_img_list.append(img6)
            new_img_list.append(img7)
            new_img_list.append(img8)
            new_img_list.append(img9)
            new_img_list.append(img10)
            new_img_list.append(img11)
            new_img_list.append(img12)

            # 将剪切的12个小图进行组合
            ret_img1 = np.hstack((new_img_list[0], new_img_list[1], new_img_list[2], new_img_list[3]))
            ret_img2 = np.hstack((new_img_list[4], new_img_list[5], new_img_list[6], new_img_list[7]))
            ret_img3 = np.hstack((new_img_list[8], new_img_list[9], new_img_list[10], new_img_list[11]))
        
            ret_img = np.vstack((ret_img1, ret_img2, ret_img3))

            # 经过剪切后，会有像素损失，所以需要resize原始图片大小
            ret_img = cv2.resize(ret_img, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
            
            # 经过去手写处理后，手写体部分字体有的没有被才擦除干净，进行自适应阈值二值化操作
            # ret_img = cv2.adaptiveThreshold(ret_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 55) 
    
    # 将原图与结果图拼在一起
    ret_img1 = cv2.cvtColor(ret_img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rotate_res = cv2.resize(rotate_res, (img.shape[1], img.shape[0]))
    if img_H > img_W:
        # final_ret_img = np.hstack((rotate_res, inputImg, ret_img1)) # 将原图、校正图和结果图拼接在一起
        final_ret_img = np.hstack((rotate_res, ret_img1)) # 将原图和结果图拼接在一起
    else:
        # final_ret_img = np.vstack((rotate_res, inputImg, ret_img1)) # 将原图、校正图和结果图拼接在一起
        final_ret_img = np.vstack((rotate_res, ret_img1)) # 将原图和结果图拼接在一起
    t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv2.imwrite("/home/ytfwork/output/" + t + "_ret.jpg", ret_img)

    return ret_img

# *****************************************************************************************************************
#                                                      图片锐化
# *****************************************************************************************************************
def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)

    return dst
