# -- coding: utf-8 --
# Author: hushukai

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

from .utils import np_utils
from .text_connector.detectors import TextDetector

from config import CTPN_ROOT_DIR


def draw_text_lines(np_img, text_lines):
    np_img = np_img.astype(np.uint8)
    h, w = np_img.shape[:2]
    
    PIL_img = Image.fromarray(np_img)
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert(mode="RGB")
    
    draw = ImageDraw.Draw(PIL_img)
    # font_path = os.path.join(CTPN_ROOT_DIR, "font", "FiraMono-Medium.otf")
    # font_object = ImageFont.truetype(font=font_path, size=int(3e-2 * h))  # 字体
    
    for i, text_line in enumerate(text_lines):
        score = text_line[8]  # 得分
        
        label = '{:.2f}'.format(score)  # 标签
        # label_size = draw.textsize(label, font_object)  # 标签文字
        
        # top, left, bottom, right = text_line
        x1, y1, x2, y2, x3, y3, x4, y4 = np.round(text_line[:8]).astype(np.int32)
        
        text_origin = np.array([x1 + 1, y1 + 1])
        
        # draw.rectangle([left, top, right, bottom], outline=colors[c], width=2)  # 画框
        draw.polygon(xy=[x1, y1, x2, y2, x3, y3, x4, y4], outline="blue")
        # draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=self.colors[c])    # 文字背景
        # draw.text(text_origin, label, fill="blue", font=font_object)  # 文案
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    return np_img


def draw_text_lines_py(image, boxes, scores):
    im_shape = image.shape[:2]
    text_lines = combine_boxes_py(boxes, scores, im_shape)
    
    raw_type = image.dtype
    image = draw_text_lines(image, text_lines)
    image = image.astype(raw_type)
    
    return image


def combine_boxes_py(boxes, scores, im_shape):
    boxes = np_utils.remove_pad(boxes)
    scores = np_utils.remove_pad(scores)[:, 0]
    
    textdetector = TextDetector(DETECT_MODE='H')  # 文本行检测器
    text_lines = textdetector.detect(boxes, scores, im_shape)
    
    return text_lines


def images_to_summary_tf(inputs, text_type="horizontal"):
    """只汇总批的第一张图片"""
    images, boxes, scores = inputs

    # 用python拼接，用tf画框
    im_shape = tf.shape(images)[1:3]
    line_boxes = tf.numpy_function(combine_boxes_py, inp=[boxes[0], scores[0], im_shape], Tout=tf.float32)

    x1, y1, x2, y2 = line_boxes[:, 0], line_boxes[:, 1], line_boxes[:, 2], line_boxes[:, 3]
    x3, y3, x4, y4 = line_boxes[:, 4], line_boxes[:, 5], line_boxes[:, 6], line_boxes[:, 7]
    _x1, _y1, _x2, _y2 = (x1 + x4) / 2, (y1 + y2) / 2, (x2 + x3) / 2, (y3 + y4) / 2
    detected_boxes = tf.stack([_y1, _x1, _y2, _x2], axis=1)
    im_shape = tf.cast(tf.concat([im_shape, im_shape], axis=0), tf.float32)
    detected_boxes = detected_boxes / im_shape
    summary_img = tf.image.draw_bounding_boxes(images[0:1], detected_boxes[tf.newaxis,], colors=[[0., 255., 0.]])
    
    # 直接用python画框
    # image = tf.numpy_function(draw_text_lines_py, inp=[images[0], boxes[0], scores[0]], Tout=tf.float32)
    # summary_img = image[tf.newaxis, ...]
    
    if text_type.lower() in ("v", "vertical"):
        summary_img = tf.transpose(summary_img[:, ::-1], perm=(0, 2, 1, 3))
        
    return summary_img
