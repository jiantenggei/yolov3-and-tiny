import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo import YOLO,detect_video
import os
from tqdm import tqdm
def predict():
    #-------------------------------------------------------------
    # 新建 yolo 对象，使用模型权重路径，以及其他参数，到yolo.py 中更改
    # predict_model 为 img 预测图片， video  输入为视频，记得更改视频路劲   video_path 为 0 时调用摄像头  为视频路径时 读取视频
    # predict_model 为 dir_predict 时输入为存放图片的路径 ，预测完成后 放入out_img 使用时记得修改路劲
    # 要预测  指定类别是 ，可以设置好重新训练模型，或者进入detect_image 修改参数 if predicted_classes='car'
    #--------------------------------------------------------------
    yolo=YOLO()  
    predict_model='dir_predict'

    video_path= r'D:\Program Files\JiJiDown\Download\4K疫情下的城市生活戴口罩人群-视频素材-凌晨两点素材网 - 1.468a8b613ffe9db966ad1e0d2f061273(Av625038589,P1).mp4'  
    video_save_path=""

    dir_img_input='img/'
    dir_save_path='out_img/'
    if predict_model=='img':
        while(True):
            img=input('Input image filename:')
            try:
                image=Image.open(img)
            except:
                print('Open Image Error！ Please Try Again')
                continue
            else:
                out_image=yolo.detect_image(image)
                out_image.show()
    elif predict_model=='video':
        detect_video(yolo,video_path,video_save_path) # 可以获取fps
    
    elif predict_model=='dir_predict':
        #---------------------------------
        #拿到所有图片 通过detect image 检测
        #-------------------------------
        imgs=os.listdir(dir_img_input)
        for img_name in tqdm(imgs):
             if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_img_input, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name)) 
    else:
           raise AssertionError("Please specify the correct mode: 'img', 'video', 'dir_predict'.")

if __name__=='__main__':
    predict()