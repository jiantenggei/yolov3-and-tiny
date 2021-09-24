import os
import random
import xml.etree.ElementTree as ET
from util import get_classes
from os import getcwd


# 训练集 验证集划分
train_percent=0.9
trainval_percent=0.9

#------------------------------
# VOCdevkit_path 数据集的路径 记得更改
# classes_path 里存放的你类别的定义 
#-----------------------------

VOCdevkit_path  = 'MyVOCdevkit' # 
classes_path=r'model_data\mask_classes.txt' # 
VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]

classes=get_classes(classes_path)
#print(classes)


#从xml -> txt  标注
def convert_annotation(year, image_id, list_file):
    in_file = open(VOCdevkit_path+'/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# 在ImageSets 生成索引
def generate_index():
        random.seed(0)
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")


wd=getcwd()
#先在VOCdevkit/VOC%s/ImageSets/Main 生成索引
#然后再生成 2007_train.txt 2007_val.txt
generate_index()
for year, image_set in VOCdevkit_sets:
    image_ids = open(VOCdevkit_path+'/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/%s/VOC%s/JPEGImages/%s.png'%(wd,VOCdevkit_path ,year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()