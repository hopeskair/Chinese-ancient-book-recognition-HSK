# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from .model import work_net, compile, get_layer, add_metrics
from .data_pipeline import data_generator


from util import check_or_makedirs
from config import CTPN_CKPT_DIR, CTPN_LOGS_DIR
from config import CTPN_BOOK_PAGE_TAGS_FILE
from config import BATCH_SIZE_BOOK_PAGE


def tf_config():
    tf.config.set_soft_device_placement(True)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except: pass


def get_callbacks(model_struc="densenet_gru", text_type="horizontal"):
    check_or_makedirs(dir_name=CTPN_CKPT_DIR)
    checkpoint = ModelCheckpoint(filepath=os.path.join(CTPN_CKPT_DIR, model_struc + "_" + text_type + "_ctpn_{epoch:04d}.h5"),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=1e-4)
    
    check_or_makedirs(CTPN_LOGS_DIR)
    logs = TensorBoard(log_dir=CTPN_LOGS_DIR)
    
    return [checkpoint, lr_reducer, logs]


def main(data_file, src_type, text_type, epochs, init_epochs=0, model_struc="densenet_gru", weights_path=""):
    tf_config()
    K.set_learning_phase(True)
    
    # 加载模型
    train_model, val_model = work_net(stage="train", batch_size=BATCH_SIZE_BOOK_PAGE, text_type=text_type, model_struc=model_struc)
    compile(train_model, loss_names=['ctpn_class_loss', 'ctpn_regress_loss', 'side_regress_loss'])
    
    # 增加度量和图片汇总
    outputs = get_layer(train_model, 'ctpn_target').output
    add_metrics(train_model, ['gt_num', 'pos_num', 'neg_num', 'gt_min_iou', 'gt_avg_iou'], outputs[-5:])
    
    train_model.summary()
    
    # load model
    load_path = os.path.join(CTPN_CKPT_DIR, model_struc + "_" + text_type + "_ctpn_{:04d}.h5".format(init_epochs))
    weights_path = weights_path if os.path.exists(weights_path) else load_path
    if os.path.exists(weights_path):
        train_model.load_weights(weights_path, by_name=True)
        print("\nLoad model weights from %s\n"%weights_path)
    
    training_generator, validation_generator = data_generator(data_file=data_file,
                                                              batch_size=BATCH_SIZE_BOOK_PAGE,
                                                              src_type=src_type,
                                                              text_type=text_type)

    summary_writer = tf.summary.create_file_writer(CTPN_LOGS_DIR)
    steps_per_epoch = 200
    for epoch in range(epochs):
        # 开始训练
        train_model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epoch+1,
                                  initial_epoch=epoch,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  validation_steps=20,
                                  callbacks=get_callbacks(model_struc, text_type),
                                  max_queue_size=100,
                                  workers=2,
                                  use_multiprocessing=True)
        
        
        summary_images = val_model.predict(x=validation_generator, steps=5, verbose=1)   # 汇总图片
        with summary_writer.as_default():
            tf.summary.image("image", summary_images.astype("uint8"), step=epoch*steps_per_epoch, max_outputs=20)
        summary_writer.flush()
    
    summary_writer.close()
    train_model.save_weights(os.path.join(CTPN_CKPT_DIR, model_struc + "_" + text_type + "_ctpn_finished.h5"))    # 保存模型


if __name__ == '__main__':
    main(data_file=CTPN_BOOK_PAGE_TAGS_FILE, src_type="tfrecords", text_type="vertical", epochs=500, init_epochs=0)
    print("Done !")
