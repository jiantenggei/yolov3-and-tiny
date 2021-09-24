from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
import tensorflow as tf
from util import *

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def train():
    # 你的 数据集文件路劲
    train_annotation_path = r"2007_train.txt"
    val_annotation_path = r"2007_val.txt"

    anchors_path = r"model_data\mask_anchor.txt"
    classes_path = r"model_data\mask_classes.txt"
    log_dir = "logs/yolov3_log/"
    weights_dir = "weights/yolov3/"
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  #  必须是32的 倍数  yolo 的设定

    model = create_model(
        input_shape=input_shape,
        anchors=anchors,
        num_classes=num_classes,
        # freeze_body=0,
        weights_path="model_data\yolov3.h5",
    )
    print(len(model.layers))
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        weights_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        period=3,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1
    )

    # 读取数据集对应的 .txt 文件
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 配置训练参数
    Freeze_Train = True
    # ------------------------------------------------------------
    # 先冻结一定网络层进行训练，这样训练比较快 ,得到一个loss稳定的model
    # -------------------------------------------------------------
    if Freeze_Train:
        batch_size = 4
        model.compile(
            optimizer=Adam(1e-3),
            loss={# use custom yolo_loss Lambda layer.
            "yolo_loss": lambda y_true, y_pred: y_pred}
        )
        print("Train on {} samples, val on {} samples, with batch size {}.".format(num_train, num_val, batch_size))
        model.fit(
            data_generator_wrapper(train_lines,batch_size,input_shape,anchors,num_classes),
            steps_per_epoch=max(1,num_train//batch_size),
            validation_data=data_generator_wrapper(val_lines,batch_size,input_shape,anchors,num_classes),
            validation_steps=max(1,num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging,checkpoint]
        )
        model.save_weights(weights_dir+'trained_weights_stage_1.h5')
     #-----------------------------------------------------------------
     # 解冻所有层，并调小学习率训练
     #-----------------------------------------------------------------
    if True:
        for i in range(len(model.layers)): model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


if __name__=='__main__':
    train()