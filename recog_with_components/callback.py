# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from util import check_or_makedirs
from config import CHAR_RECOG_CKPT_DIR, CHAR_RECOG_LOGS_DIR


def tf_config():
    tf.config.set_soft_device_placement(True)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass


def get_callbacks(model_struc="densenet"):
    check_or_makedirs(dir_name=CHAR_RECOG_CKPT_DIR)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_components_" + model_struc + "_{epoch:04d}.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.6,
                                   cooldown=0,
                                   patience=4,  # num of epochs
                                   min_lr=0)
    
    check_or_makedirs(CHAR_RECOG_LOGS_DIR)
    logs = TensorBoard(log_dir=CHAR_RECOG_LOGS_DIR)
    
    return [checkpoint, lr_reducer, logs]

