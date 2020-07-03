# -*- encoding: utf-8 -*-
# Author: hushukai

import functools
import threading
import tensorflow as tf
import numpy as np

from config import SEGMENT_ROOT_DIR, SEGMENT_CKPT_DIR, SEGMENT_LOGS_DIR
from config import SEGMENT_BATCH_SIZE, SEGMENT_FIXED_HEIGHT, SEGMENT_FEATURE_STRIDE
from config import SEGMENT_CLS_SCORE_THRESH, SEGMENT_DISTANCE_THRESH, SEGMENT_NMS_MAX_OUTPUTS


def pad_to_fixed_size_tf(input_tensor, fixed_size):
    """padding到固定长度, 在第二维度末位增加一个padding_flag, no_pad:1, pad:0.

    Parameter:
        input_tensor: 二维张量
    """
    input_size = tf.shape(input_tensor)[0]
    x = tf.pad(input_tensor, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=1)
    
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)  # padding
    
    return x[:fixed_size]


def remove_pad_tf(input_tensor):
    """no_pad:1, pad:0; Be in order."""
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


def pad_to_fixed_size_np(input_array, fixed_size):
    """padding到固定长度, 在第二维度末位增加一个padding_flag, no_pad:1, pad:0.

    Parameter:
        input_tensor: 二维张量
    """
    shape = input_array.shape
    np_array = np.pad(input_array, ((0, 0), (0, 1)), mode='constant', constant_values=1)  # 增加tag

    pad_num = max(0, fixed_size - shape[0])
    x = np.pad(np_array, ((0, pad_num), (0, 0)), mode='constant', constant_values=0)  # 增加padding

    return x[:fixed_size]


def remove_pad_np(input_array):
    """no_pad:1, pad:0; Be in order."""
    pad_tag = input_array[..., -1]
    real_size = int(np.sum(pad_tag))
    return input_array[:real_size, :-1]


def get_segment_task_path(segment_task):
    root_dir = SEGMENT_ROOT_DIR[segment_task]
    ckpt_dir = SEGMENT_CKPT_DIR[segment_task]
    logs_dir = SEGMENT_LOGS_DIR[segment_task]
    return root_dir, ckpt_dir, logs_dir


def get_segment_task_params(segment_task):
    batch_size = SEGMENT_BATCH_SIZE[segment_task]
    fixed_h = SEGMENT_FIXED_HEIGHT[segment_task]
    feat_stride = SEGMENT_FEATURE_STRIDE[segment_task]
    return batch_size, fixed_h, feat_stride


def get_segment_task_thresh(segment_task):
    cls_score_thresh = SEGMENT_CLS_SCORE_THRESH[segment_task]
    distance_thresh = SEGMENT_DISTANCE_THRESH[segment_task]
    nms_max_outputs = SEGMENT_NMS_MAX_OUTPUTS[segment_task]
    return cls_score_thresh, distance_thresh, nms_max_outputs


class ThreadSafeGenerator:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    
    def __iter__(self):
        return self
    
    def __next__(self):  # python3
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    @functools.wraps(f)
    def g(*args, **kwargs):
        return ThreadSafeGenerator(f(*args, **kwargs))
    
    return g