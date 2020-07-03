# -- coding: utf-8 --
# Author: hushukai

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

from .utils import remove_pad_tf

from config import SEGMENT_BASE_ROOT_DIR


def draw_text_boxes(np_img, text_boxes):
    np_img = np_img.astype(np.uint8)
    h, w = np_img.shape[:2]
    
    PIL_img = Image.fromarray(np_img)
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert(mode="RGB")
    
    draw = ImageDraw.Draw(PIL_img)
    # font_path = os.path.join(CTPN_ROOT_DIR, "font", "FiraMono-Medium.otf")
    # font_object = ImageFont.truetype(font=font_path, size=16)  # 字体, or size=int(3e-2 * h)
    
    for i, text_box in enumerate(text_boxes):
        score = text_box[8]  # 得分
        
        label = '{:.2f}'.format(score)  # 标签
        # label_size = draw.textsize(label, font_object)  # 标签文字
        
        # top, left, bottom, right = text_line
        x1, y1, x2, y2, x3, y3, x4, y4 = np.round(text_box[:8]).astype(np.int32)
        
        text_origin = np.array([x1 + 1, y1 + 1])
        
        # draw.rectangle([left, top, right, bottom], outline=colors[c], width=2)  # 画框
        draw.polygon(xy=[x1, y1, x2, y2, x3, y3, x4, y4], outline="blue")
        # draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=self.colors[c])    # 文字背景
        # draw.text(text_origin, label, fill="blue", font=font_object)  # 文案
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    return np_img


def draw_split_lines(np_img, split_positions, scores=None):
    np_img = np_img.astype(np.uint8)
    h, w = np_img.shape[:2]
    split_positions = split_positions.astype(np.int32)
    
    PIL_img = Image.fromarray(np_img)
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert(mode="RGB")
    
    draw = ImageDraw.Draw(PIL_img)
    font_path = os.path.join(SEGMENT_BASE_ROOT_DIR, "font", "FiraMono-Medium.otf")
    font_object = ImageFont.truetype(font=font_path, size=16)  # 字体, or size=int(3e-2 * h)
    
    for i, (x1, x2) in enumerate(split_positions):
        y1, y2 = 0, h-1
        draw.line([x1, y1, x2, y2], fill="green", width=2)  # 画框
        
        if scores is not None:
            score = scores[i]  # 得分
            label = '{:.2f}'.format(score)  # 标签
            text_origin = np.array([x1 + 1, y1 + 1])
            # label_size = draw.textsize(label, font_object)  # 标签文字
            # draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=self.colors[c])    # 文字背景
            draw.text(text_origin, label, fill="green", font=font_object)  # 文案

    np_img = np.array(PIL_img, dtype=np.uint8)
    return np_img


def draw_split_lines_py(np_img, split_positions, scores):
    raw_type = np_img.dtype
    np_img = draw_split_lines(np_img, split_positions, scores)
    np_img = np_img.astype(raw_type)
    return np_img


def images_to_summary_tf(inputs, segment_task="book_page", text_type="horizontal"):
    """只汇总批的第一张图片"""
    images, split_positions, scores = inputs

    split_positions = remove_pad_tf(split_positions[0])
    scores = remove_pad_tf(scores[0])[:, 0]
    
    # 直接用python画框
    image = tf.numpy_function(draw_split_lines_py, inp=[images[0], split_positions, scores], Tout=tf.float32)
    summary_img = image[tf.newaxis, ...]
    
    text_type = text_type[0].lower()
    if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
        summary_img = tf.transpose(summary_img[:, ::-1], perm=(0, 2, 1, 3))
    
    return summary_img
