# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from .utils import get_segment_task_path

from util import check_or_makedirs


def tf_config():
    tf.config.set_soft_device_placement(True)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass


def get_callbacks(segment_task, model_struc="densenet_gru"):
    _, ckpt_dir, logs_dir = get_segment_task_path(segment_task)
    
    check_or_makedirs(dir_name=ckpt_dir)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, segment_task + "_segment_" + model_struc + "_{epoch:04d}.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.6,
                                   cooldown=0,
                                   patience=3,  # num of epochs
                                   min_lr=0)
    
    check_or_makedirs(logs_dir)
    logs = TensorBoard(log_dir=logs_dir)
    
    return [checkpoint, lr_reducer, logs]

