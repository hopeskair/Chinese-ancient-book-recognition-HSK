# -- coding: utf-8 --
# Miscellaneous utility functions

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import colorsys
import os

from ..config import YOLO3_ROOT_DIR


def get_anchors(anchors_path):
    '''loads the anchors from file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_diff_colors(classes):
    # 不同类别的框使用不同的颜色
    hsv_tuples = [(i/len(classes), 1.0, 1.0) for i in range(len(classes))]  # 生成不同颜色
    rgb_tuples = list(map(lambda hsv: colorsys.hsv_to_rgb(*hsv), hsv_tuples))
    colors = list(map(lambda rgb: (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)), rgb_tuples))
    np.random.shuffle(colors)
    return colors


def check_boxes(boxes, raw_img_size):
    assert boxes.dtype == "float32"
    h, w = raw_img_size
    
    boxes[:, 0][boxes[:, 0]<0] = 0      # y1
    boxes[:, 1][boxes[:, 1]<0] = 0      # x1
    boxes[:, 2][boxes[:, 2]>h-1] = h-1  # y2
    boxes[:, 3][boxes[:, 3]>w-1] = w-1  # x2
    
    return boxes


def draw_boxes(np_img, boxes, scores, classes):
    assert len(boxes) == len(scores) and len(boxes) == len(classes)
    colors = get_diff_colors(classes)  # 不同类别的框使用不同的颜色
    np_img = np_img.astype(np.uint8)
    h, w = np_img.shape[:2]
    
    PIL_img = Image.fromarray(np_img)
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert(mode="RGB")

    draw = ImageDraw.Draw(PIL_img)
    font_path = os.path.join(YOLO3_ROOT_DIR, "font", "FiraMono-Medium.otf")
    font_object = ImageFont.truetype(font=font_path, size=int(3e-2 * h))  # 字体
    for i, c in enumerate(classes):
        class_name = classes[c] # 类别
        box = boxes[i]          # 框
        score = scores[i]       # 得分
    
        label = '{} {:.2f}'.format(class_name, score)   # 标签
        # label_size = draw.textsize(label, font_object)  # 标签文字
    
        top, left, bottom, right = box
        top = np.round(top).astype('int32')
        left = np.round(left).astype('int32')
        bottom = np.round(bottom).astype('int32')
        right = np.round(right).astype('int32')
    
        text_origin = np.array([left + 1, top + 1])
    
        draw.rectangle([left, top, right, bottom], outline=colors[c], width=2)  # 画框
        # draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=self.colors[c])    # 文字背景
        draw.text(text_origin, label, fill=colors[c], font=font_object)  # 文案
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    return np_img