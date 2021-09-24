from util import get_anchors,create_model,create_tiny_model
#测试模型定义是否正确
if __name__=='__main__':
    # 输入大小必须是13的倍数
    input_shape=(416,416) # 
    #tiny 参数
    tiny_anchors_path = r'model_data\tiny_yolo_anchors.txt' # your anchor path
    num_classes=20 #Voc 数据集分20类
    tiny_anchors=get_anchors(tiny_anchors_path)
    yolov3_tiny_model=create_tiny_model(input_shape=input_shape,num_classes=num_classes,anchors=tiny_anchors)
    yolov3_tiny_model.summary()
    
    #yolo3参数
    # anchors_path=r'model_data\yolo_anchors.txt'
    # anchors=get_anchors(anchors_path)
    # yolov3_model =create_model(input_shape=input_shape,num_classes=num_classes,anchors=anchors)
    # yolov3_model.summary()
    
    
