# -*- coding: utf-8 -*-

import os
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from .model import ctpn_net, compile, get_layer, add_metrics
from .data_pipeline import data_generator

from ..util import check_or_makedirs
from ..config import CTPN_CKPT_DIR, CTPN_LOGS_DIR
from ..config import BATCH_SIZE_BOOK_PAGE


def get_callbacks(model_struc="densenet_gru"):
    
    check_or_makedirs(dir_name=CTPN_CKPT_DIR)
    checkpoint = ModelCheckpoint(filepath=os.path.join(CTPN_CKPT_DIR, model_struc + "_ctpn_{epoch:03d}.h5"),
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


def main(data_file, src_type, model_type, epochs, init_epochs=0, model_struc="densenet_gru", weights_path=""):
    K.set_learning_phase(True)
    
    # 加载模型
    ctpn_model = ctpn_net(stage="train", model_struc=model_struc)
    compile(ctpn_model, loss_names=['ctpn_class_loss', 'ctpn_regress_loss'])
    
    # 增加度量
    output = get_layer(ctpn_model, 'ctpn_target').output
    add_metrics(ctpn_model, ['gt_num', 'pos_num', 'neg_num', 'gt_min_iou', 'gt_avg_iou'], output[-5:])
    
    if os.path.exists(weights_path):
        ctpn_model.load_weights(weights_path, by_name=True)
    ctpn_model.summary()

    training_generator, validation_generator = data_generator(data_file=data_file,
                                                              batch_size=BATCH_SIZE_BOOK_PAGE,
                                                              src_type=src_type,
                                                              text_type=model_type)
    # 开始训练
    ctpn_model.fit_generator(generator=training_generator,
                             steps_per_epoch=1000,
                             epochs=epochs,
                             initial_epoch=init_epochs,
                             validation_data=validation_generator,
                             validation_steps=100,
                             verbose=1,
                             callbacks=get_callbacks(),
                             max_queue_size=100,
                             workers=2,
                             use_multiprocessing=True)

    # 保存模型
    ctpn_model.save(os.path.join(CTPN_CKPT_DIR, model_type+"_"+model_struc+"_ctpn_finished.h5"))


if __name__ == '__main__':
    main(data_file="", src_type="images", model_type="vertical", epochs=100)
    print("Done !")
