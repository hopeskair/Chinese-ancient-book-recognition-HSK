# -*- coding: utf-8 -*-
"""
   File Name：     tf_utils
   Description :  tensorflow工具类
   Author :       mick.yi
   date：          2019/3/13
"""
import tensorflow as tf


def pad_to_fixed_size(input_tensor, fixed_size):
    """padding到固定长度, 在第二维度末位增加一个padding_flag, no_pad:1, pad:0.
    
    Parameter:
        input_tensor: 二维张量
    """
    input_size = tf.shape(input_tensor)[0]
    x = tf.pad(input_tensor, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=1)
    
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)  # padding
    
    return x


def remove_pad(input_tensor):
    """no_pad:1, pad:0; Be in order."""
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


def clip_boxes(boxes, window):
    """
    将boxes裁剪到指定的窗口范围内
    :param boxes: 边框坐标，[N,(y1,x1,y2,x2)]
    :param window: 窗口坐标，[(y1,x1,y2,x2)]
    :return:
    """
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)  # split后维数不变

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)  # wy1<=y1<=wy2
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

    clipped_boxes = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    # clipped_boxes.([boxes.shape[0], 4])
    return clipped_boxes


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)
    w *= tf.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return tf.stack([y1, x1, y2, x2], axis=1)
