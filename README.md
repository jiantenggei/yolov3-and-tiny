# yolov3-and-yolov3-tiny
这是yolov3 和 yolov3-tiny 的目标检测网络，可用于训练自己的数据集

# 环境要求
工具   | 版本
-------- | -----
python  | 3.8.0
TensorFlow-gpu  | 2.5.0

# 训练数据和权重摆放
```bash
─VOC2007
    ├─Annotations
    │	└─000005.xml
    │	└─000006.xml
    │	└─xxxx.xml	
    ├─ImageSets
    │  └─Main
    └─JPEGImages
    │	└─000005.jpg
    │	└─000006.jpg
    │	└─xxxx.jpg
```
model_data 目录下摆放.h5 文件

# 训练步骤
1.运行 voc_annotation.py 生成 训练索引 2007_train.txt 和2007——val.txt （注意生成时，自己的图片是png 还是jpg ，需要更改一下 保证 VOCdevkit\VOC2007\ImageSets\文件夹下有Main 文件夹）
2. 运行 kmeans.py 聚类 生存anchor boxes,存入在yolo_anchors1.txt 中
3. 找到 train_yolov3.py 或者train_tiny_model.py,更改配置路径，点击运行

# 预测步骤
1. 在根目录下的yolo.py 文件中 设置权重文件路径，achor boxes 和 classes的路劲
2. 在predict.py 中 选择预测模式，点击运行即可
**注意：训练时根据设备条件 设置合适的batch_size**
# 其他地址：
csdn 博客地址： https://blog.csdn.net/qq_38676487/article/details/120443059?spm=1001.2014.3001.5501

b站 配置讲解: https://www.bilibili.com/video/BV13r4y127wr（已删除）
内容| 链接
-------- | -----
VOC2007 数据集 | [链接](https://www.kaggle.com/yihaoyang/voc2007)
戴口罩数据集| [链接](https://www.kaggle.com/andrewmvd/face-mask-detection?select=images)
权重文件  | [链接](https://pan.baidu.com/s/1Oc6wEXIIoLJKxekHQb9qyg) 提取码：y32m
完整项目地址（包含所有文件）|[链接](https://pan.baidu.com/s/1MmXLlsGxmIOxw_blcT6LMA)  提取码  jmpl

conda虚拟环境一键导入：

```javascript
conda env create -f tf2.5.yaml
```
其他问题私信：1308659229@qq.com

**如果觉得有用清给我点star**

# 参考
https://github.com/bubbliiiing/yolo3-tf2


https://github.com/qqwweee/keras-yolo3
