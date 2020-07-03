# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import sys
import json
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import ONE_TEXT_LINE_IMGS_H, ONE_TEXT_LINE_TAGS_FILE_H
from config import ONE_TEXT_LINE_IMGS_V, ONE_TEXT_LINE_TAGS_FILE_V
from config import ONE_TEXT_LINE_TFRECORDS_H, ONE_TEXT_LINE_TFRECORDS_V
from config import TWO_TEXT_LINE_IMGS_H, TWO_TEXT_LINE_TAGS_FILE_H
from config import TWO_TEXT_LINE_IMGS_V, TWO_TEXT_LINE_TAGS_FILE_V
from config import TWO_TEXT_LINE_TFRECORDS_H, TWO_TEXT_LINE_TFRECORDS_V
from config import MIX_TEXT_LINE_IMGS_H, MIX_TEXT_LINE_TAGS_FILE_H
from config import MIX_TEXT_LINE_IMGS_V, MIX_TEXT_LINE_TAGS_FILE_V
from config import MIX_TEXT_LINE_TFRECORDS_H, MIX_TEXT_LINE_TFRECORDS_V
from config import FONT_FILE_DIR, EXTERNEL_IMAGES_DIR, MAX_ROTATE_ANGLE
from util import CHAR2ID_DICT, IGNORABLE_CHARS, IMPORTANT_CHARS

from util import check_or_makedirs
from data_generator.img_utils import rotate_PIL_image
from data_generator.img_utils import find_min_bound_box
from data_generator.img_utils import adjust_img_and_put_into_background
from data_generator.img_utils import reverse_image_color
from data_generator.img_utils import generate_bigger_image_by_font
from data_generator.img_utils import load_external_image_bigger
from data_generator.generate_chinese_images import get_external_image_paths


def check_text_type(text_type):
    if text_type.lower() in ("h", "horizontal"):
        text_type = "h"
    elif text_type.lower() in ("v", "vertical"):
        text_type = "v"
    else:
        ValueError("Optional text_types: 'h', 'horizontal', 'v', 'vertical'.")
    return text_type


def check_text_line_shape(shape, text_type):
    text_h, text_w = shape
    if text_type == "h":
        assert text_h <= text_w, "Horizontal text must meet height <= width."
    if text_type == "v":
        assert text_h >= text_w, "Vertical text must meet height >= width."


def generate_one_text_line_imgs(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_imgs_dir, text_line_tags_file = ONE_TEXT_LINE_IMGS_H, ONE_TEXT_LINE_TAGS_FILE_H
    if text_type == "v":
        text_line_imgs_dir, text_line_tags_file = ONE_TEXT_LINE_IMGS_V, ONE_TEXT_LINE_TAGS_FILE_V
    
    check_or_makedirs(text_line_imgs_dir)
    
    _shape = text_shape
    with open(text_line_tags_file, "w", encoding="utf-8") as fw:
        for i in range(obj_num):
            if text_shape is None and text_type == "h":
                _shape = (random.randint(36, 64), random.randint(540, 960))
            if text_shape is None and text_type == "v":
                _shape = (random.randint(540, 960), random.randint(36, 64))

            PIL_text, char_and_box_list, split_pos_list = create_one_text_line(_shape, text_type=text_type)
            image_tags = {"char_and_box_list": char_and_box_list, "split_pos_list": split_pos_list}
            
            img_name = "text_line_%d.jpg" % i
            save_path = os.path.join(text_line_imgs_dir, img_name)
            PIL_text.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")
            
            if i % 50 == 0:
                print("Process bar: %.2f%%" % (i*100/obj_num))
                sys.stdout.flush()


def generate_one_text_line_tfrecords(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_tfrecords_dir = ONE_TEXT_LINE_TFRECORDS_H
    if text_type == "v":
        text_line_tfrecords_dir = ONE_TEXT_LINE_TFRECORDS_V
    
    check_or_makedirs(text_line_tfrecords_dir)

    # 可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(text_line_tfrecords_dir, "text_lines_%d.tfrecords" % i))
         for i in range(20)]
    
    # 保存生成的文本图片
    _shape = text_shape
    for i in range(obj_num):
        writer = random.choice(writers_list)

        if text_shape is None and text_type == "h":
            _shape = (random.randint(36, 64), random.randint(540, 960))
        if text_shape is None and text_type == "v":
            _shape = (random.randint(540, 960), random.randint(36, 64))
        
        PIL_text, char_and_box_list, split_pos_list = create_one_text_line(_shape, text_type=text_type)
        
        bytes_image = PIL_text.tobytes()  # 将图片转化为原生bytes
        bytes_chars = "".join([chinese_char for chinese_char, gt_box in char_and_box_list]).encode("utf-8")
        labels = np.array([CHAR2ID_DICT[char] for char, gt_box in char_and_box_list], dtype=np.int32).tobytes()
        gt_boxes = np.array([gt_box for chinese_char, gt_box in char_and_box_list], dtype=np.int32).tobytes()
        split_positions = np.array(split_pos_list, dtype=np.int32).tobytes()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.width])),
                    'bytes_chars': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_chars])),
                    'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
                    'gt_boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_boxes])),
                    'split_positions': tf.train.Feature(bytes_list=tf.train.BytesList(value=[split_positions]))
                }))
        writer.write(example.SerializeToString())

        if i % 50 == 0:
            print("Process bar: %.2f%%" % (i*100/obj_num))
            sys.stdout.flush()

    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def generate_two_text_line_imgs(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_imgs_dir, text_line_tags_file = TWO_TEXT_LINE_IMGS_H, TWO_TEXT_LINE_TAGS_FILE_H
    if text_type == "v":
        text_line_imgs_dir, text_line_tags_file = TWO_TEXT_LINE_IMGS_V, TWO_TEXT_LINE_TAGS_FILE_V
    
    check_or_makedirs(text_line_imgs_dir)
    
    _shape = text_shape
    with open(text_line_tags_file, "w", encoding="utf-8") as fw:
        for i in range(obj_num):
            if text_shape is None and text_type == "h":
                _shape = (random.randint(54, 108), random.randint(108, 960)) # 双行文本数据无需太长
            if text_shape is None and text_type == "v":
                _shape = (random.randint(108, 960), random.randint(54, 108)) # 双行文本数据无需太长
            
            # 训练双行文本的切分，既需要生成双行数据，也需要生成单行数据（不切分的情况）
            PIL_text, split_pos_list = create_two_text_line(_shape, text_type=text_type)
            image_tags = {"split_pos_list": split_pos_list}
            
            img_name = "text_line_%d.jpg" % i
            save_path = os.path.join(text_line_imgs_dir, img_name)
            PIL_text.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")
            
            if i % 50 == 0:
                print("Process bar: %.2f%%" % (i * 100 / obj_num))
                sys.stdout.flush()


def generate_two_text_line_tfrecords(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_tfrecords_dir = TWO_TEXT_LINE_TFRECORDS_H
    if text_type == "v":
        text_line_tfrecords_dir = TWO_TEXT_LINE_TFRECORDS_V
    
    check_or_makedirs(text_line_tfrecords_dir)
    
    # 可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(text_line_tfrecords_dir, "text_lines_%d.tfrecords" % i))
         for i in range(20)]
    
    # 保存生成的文本图片
    _shape = text_shape
    for i in range(obj_num):
        writer = random.choice(writers_list)

        if text_shape is None and text_type == "h":
            _shape = (random.randint(54, 108), random.randint(108, 960))  # 双行文本数据无需太长
        if text_shape is None and text_type == "v":
            _shape = (random.randint(108, 960), random.randint(54, 108))  # 双行文本数据无需太长

        # 训练双行文本的切分，既需要生成双行数据，也需要生成单行数据（不切分的情况）
        PIL_text, split_pos_list = create_two_text_line(_shape, text_type=text_type)
        
        bytes_image = PIL_text.tobytes()  # 将图片转化为原生bytes
        split_positions = np.array(split_pos_list, dtype=np.int32).tobytes()
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.width])),
                    'split_positions': tf.train.Feature(bytes_list=tf.train.BytesList(value=[split_positions]))
                }))
        writer.write(example.SerializeToString())
        
        if i % 50 == 0:
            print("Process bar: %.2f%%" % (i * 100 / obj_num))
            sys.stdout.flush()
    
    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def generate_mix_text_line_imgs(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_imgs_dir, text_line_tags_file = MIX_TEXT_LINE_IMGS_H, MIX_TEXT_LINE_TAGS_FILE_H
    if text_type == "v":
        text_line_imgs_dir, text_line_tags_file = MIX_TEXT_LINE_IMGS_V, MIX_TEXT_LINE_TAGS_FILE_V
    
    check_or_makedirs(text_line_imgs_dir)
    
    _shape = text_shape
    with open(text_line_tags_file, "w", encoding="utf-8") as fw:
        for i in range(obj_num):
            if text_shape is None and text_type == "h":
                _shape = (random.randint(54, 108), random.randint(720, 1280))
            if text_shape is None and text_type == "v":
                _shape = (random.randint(720, 1280), random.randint(54, 108))

            PIL_text, _, split_pos_list = create_mix_text_line(_shape, text_type=text_type)
            image_tags = {"split_pos_list": split_pos_list}
            
            img_name = "text_line_%d.jpg" % i
            save_path = os.path.join(text_line_imgs_dir, img_name)
            PIL_text.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")
            
            if i % 50 == 0:
                print("Process bar: %.2f%%" % (i * 100 / obj_num))
                sys.stdout.flush()


def generate_mix_text_line_tfrecords(obj_num=100, text_type="horizontal", text_shape=None):
    text_type = check_text_type(text_type)
    
    if text_type == "h":
        text_line_tfrecords_dir = MIX_TEXT_LINE_TFRECORDS_H
    if text_type == "v":
        text_line_tfrecords_dir = MIX_TEXT_LINE_TFRECORDS_V
    
    check_or_makedirs(text_line_tfrecords_dir)
    
    # 可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(text_line_tfrecords_dir, "text_lines_%d.tfrecords" % i))
         for i in range(20)]
    
    # 保存生成的文本图片
    _shape = text_shape
    for i in range(obj_num):
        writer = random.choice(writers_list)
        
        if text_shape is None and text_type == "h":
            _shape = (random.randint(54, 108), random.randint(720, 1280))
        if text_shape is None and text_type == "v":
            _shape = (random.randint(720, 1280), random.randint(54, 108))

        PIL_text, _, split_pos_list = create_mix_text_line(_shape, text_type=text_type)
        
        bytes_image = PIL_text.tobytes()  # 将图片转化为原生bytes
        split_positions = np.array(split_pos_list, dtype=np.int32).tobytes()
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.width])),
                    'split_positions': tf.train.Feature(bytes_list=tf.train.BytesList(value=[split_positions]))
                }))
        writer.write(example.SerializeToString())
        
        if i % 50 == 0:
            print("Process bar: %.2f%%" % (i * 100 / obj_num))
            sys.stdout.flush()
    
    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def create_one_text_line(shape=(36, 720), text_type="horizontal"):
    text_type = check_text_type(text_type)
    check_text_line_shape(shape, text_type)
    text_h, text_w = shape
    
    # 生成黑色背景
    np_text = np.zeros(shape=(text_h, text_w), dtype=np.uint8)

    # 横向排列
    if text_type == "h":
        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.0, 0.05), random.uniform(0.0, 0.2))  # (高方向, 宽方向)

        # 生成一行汉字
        y1, y2 = 0, text_h -1
        x = 0
        _, _, char_and_box_list, split_pos = generate_one_row_chars(x, y1, y2, text_w, np_text, char_spacing)

    # 纵向排列
    else:
        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.0, 0.05))  # (高方向, 宽方向)

        # 生成一列汉字
        x1, x2 = 0, text_w - 1
        y = 0
        _, _, char_and_box_list, split_pos = generate_one_col_chars(x1, x2, y, text_h, np_text, char_spacing)
    
    np_text = reverse_image_color(np_img=np_text)
    PIL_text = Image.fromarray(np_text)

    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_text.show()

    return PIL_text, char_and_box_list, split_pos


def create_two_text_line(shape=(64, 1280), text_type="horizontal"):
    """训练双行文本的切分，既需要生成双行数据，也需要生成单行数据（不切分的情况）"""
    
    text_type = check_text_type(text_type)
    check_text_line_shape(shape, text_type)
    text_h, text_w = shape
    
    # 生成黑色背景
    np_text = np.zeros(shape=(text_h, text_w), dtype=np.uint8)
    
    # 横向排列
    if text_type == "h":
        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.0, 0.05), random.uniform(0.0, 0.2))  # (高方向, 宽方向)
        
        y1, y2 = 0, text_h - 1
        x = 0
        if random.random() < 0.7:
            # 生成两行汉字
            _, text1_bbox, text2_bbox, split_pos = generate_two_rows_chars(x, y1, y2, text_w, np_text, char_spacing)
        else:
            # 生成单行汉字
            _, text_bbox, _, _ = generate_one_row_chars(x, y1, y2, text_w, np_text, char_spacing)
            split_pos = [text_bbox[1], text_bbox[3]]
    
    # 纵向排列
    else:
        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.0, 0.05))  # (高方向, 宽方向)
        
        x1, x2 = 0, text_w - 1
        y = 0
        if random.random() < 0.7:
            # 生成两列汉字
            _, text1_bbox, text2_bbox, split_pos = generate_two_cols_chars(x1, x2, y, text_h, np_text, char_spacing)
        else:
            # 生成单列汉字
            _, text_bbox, _, _ = generate_one_col_chars(x1, x2, y, text_h, np_text, char_spacing)
            split_pos = [text_bbox[0], text_bbox[2]]
    
    np_text = reverse_image_color(np_img=np_text)
    PIL_text = Image.fromarray(np_text)
    
    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_text.show()
    return PIL_text, split_pos


def create_mix_text_line(shape=(64, 1280), text_type="horizontal"):
    text_type = check_text_type(text_type)
    check_text_line_shape(shape, text_type)
    text_h, text_w = shape
    
    # 生成黑色背景
    np_text = np.zeros(shape=(text_h, text_w), dtype=np.uint8)
    
    # 横向排列
    if text_type == "h":
        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.0, 0.05), random.uniform(0.0, 0.2))  # (高方向, 宽方向)
        
        # 生成单双行
        y1, y2 = 0, text_h - 1
        x = 0
        _, text_bbox_list, split_pos = generate_mix_rows_chars(x, y1, y2, text_w, np_text, char_spacing)
    
    # 纵向排列
    else:
        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.0, 0.05))  # (高方向, 宽方向)
        
        # 生成单双列
        x1, x2 = 0, text_w - 1
        y = 0
        _, text_bbox_list, split_pos = generate_mix_cols_chars(x1, x2, y, text_h, np_text, char_spacing)
    
    np_text = reverse_image_color(np_img=np_text)
    PIL_text = Image.fromarray(np_text)
    
    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_text.show()
    
    return PIL_text, text_bbox_list, split_pos


def generate_one_row_chars(x, y1, y2, length, np_background, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    row_end = x + length - 1
    row_height = y2 - y1 + 1
    while length >= row_height:
        chinese_char, bounding_box, x_tail = \
            generate_char_img_into_unclosed_box(np_background, x1=x, y1=y1, x2=None, y2=y2, char_spacing=char_spacing)
        
        char_and_box_list.append((chinese_char, bounding_box))
        added_length = x_tail - x
        length -= added_length
        x = x_tail
    
    # 获取文本行的bounding-box
    head_x1, head_y1, _, _ = char_and_box_list[0][1]
    _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)
    
    # 获取字符之间的划分位置
    char_spacing_w = round(row_height * char_spacing[1])
    split_pos = [head_x1,]
    for i in range(len(char_and_box_list)-1):
        x_cent = (char_and_box_list[i][1][2] + char_and_box_list[i+1][1][0]) // 2
        split_pos.append(x_cent)
    split_pos.append(tail_x2)

    return x, text_bbox, char_and_box_list, split_pos


def generate_two_rows_chars(x, y1, y2, length, np_background, char_spacing):
    row_height = y2 - y1 + 1
    mid_y = y1 + round(row_height / 2)
    
    x_1, text1_bbox, _, _ = generate_one_row_chars(x, y1, mid_y, length, np_background, char_spacing)
    x_2, text2_bbox, _, _ = generate_one_row_chars(x, mid_y+1, y2, length, np_background, char_spacing)

    # 获取文本行之间的划分位置
    center_val = (text1_bbox[3] + text2_bbox[1]) // 2
    char_spacing_h = round(row_height * char_spacing[0])
    split_pos = [text1_bbox[1], center_val, text2_bbox[3]]
    
    return max(x_1, x_2), text1_bbox, text2_bbox, split_pos


def generate_one_col_chars(x1, x2, y, length, np_background, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    col_end = y + length - 1
    col_width = x2 - x1 + 1
    while length >= col_width:
        chinese_char, bounding_box, y_tail = \
            generate_char_img_into_unclosed_box(np_background, x1=x1, y1=y, x2=x2, y2=None, char_spacing=char_spacing)
        
        char_and_box_list.append((chinese_char, bounding_box))
        added_length = y_tail - y
        length -= added_length
        y = y_tail

    # 获取文本行的bounding-box
    head_x1, head_y1, _, _ = char_and_box_list[0][1]
    _, _, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)

    # 获取字符之间的划分位置
    char_spacing_h = round(col_width * char_spacing[0])
    split_pos = [head_y1, ]
    for i in range(len(char_and_box_list) - 1):
        x_cent = (char_and_box_list[i][1][3] + char_and_box_list[i + 1][1][1]) // 2
        split_pos.append(x_cent)
    split_pos.append(tail_y2)

    return y, text_bbox, char_and_box_list, split_pos


def generate_two_cols_chars(x1, x2, y, length, np_background, char_spacing):
    col_width = x2 - x1 + 1
    mid_x = x1 + round(col_width / 2)
    
    y_1, text1_bbox, _, _ = generate_one_col_chars(x1, mid_x, y, length, np_background, char_spacing)
    y_2, text2_bbox, _, _ = generate_one_col_chars(mid_x+1, x2, y, length, np_background, char_spacing)

    # 获取文本行之间的划分位置
    center_val = (text1_bbox[2] + text2_bbox[0]) // 2
    char_spacing_w = round(col_width * char_spacing[1])
    split_pos = [text1_bbox[0], center_val, text2_bbox[2]]

    return max(y_1, y_2), text1_bbox, text2_bbox, split_pos


def generate_mix_rows_chars(x, y1, y2, row_length, np_background, char_spacing):
    row_height =  y2 - y1 + 1
    x_start = x
    
    text_bbox_list = []
    head_tail_list = []
    flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
    remaining_len = row_length
    while remaining_len >= row_height:
        # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
        length = random.randint(row_height, remaining_len)
        flag += 1
        if flag % 2 == 1:
            x, text_bbox, _, _ = generate_one_row_chars(x, y1, y2, length, np_background, char_spacing)
            text_bbox_list.append(text_bbox)
            head_tail_list.append((text_bbox[0], text_bbox[2]))
        else:
            x, text1_bbox, text2_bbox, _ = generate_two_rows_chars(x, y1, y2, length, np_background, char_spacing)
            text_bbox_list.extend([text1_bbox, text2_bbox])
            head_tail_list.append((min(text1_bbox[0], text2_bbox[0]), max(text1_bbox[2], text2_bbox[2])))
        remaining_len = row_length - (x - x_start)
    
    # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...
    
    # 获取单双行的划分位置
    char_spacing_w = round(row_height * char_spacing[1])
    head_x1, tail_x2 = head_tail_list[0][0], head_tail_list[-1][1]
    split_pos = [head_x1,]
    for i in range(len(head_tail_list)-1):
        x_cent = (head_tail_list[i][1] + head_tail_list[i+1][0]) // 2
        split_pos.append(x_cent)
    split_pos.append(tail_x2)
    
    return x, text_bbox_list, split_pos


def generate_mix_cols_chars(x1, x2, y, col_length, np_background, char_spacing):
    col_width = x2 - x1 + 1
    y_start = y
    
    text_bbox_list = []
    head_tail_list = []
    flag = 0 if random.random() < 0.6 else 1  # 以单行字串还是双行字串开始
    remaining_len = col_length
    while remaining_len >= col_width:
        # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
        length = random.randint(col_width, remaining_len)
        flag += 1
        if flag % 2 == 1:
            y, text_bbox, _, _ = generate_one_col_chars(x1, x2, y, length, np_background, char_spacing)
            text_bbox_list.append(text_bbox)
            head_tail_list.append((text_bbox[1], text_bbox[3]))
        else:
            y, text1_bbox, text2_bbox, _ = generate_two_cols_chars(x1, x2, y, length, np_background, char_spacing)
            text_bbox_list.extend([text1_bbox, text2_bbox])
            head_tail_list.append((min(text1_bbox[1], text2_bbox[1]), max(text1_bbox[3], text2_bbox[3])))
        remaining_len = col_length - (y - y_start)

    # pure_two_lines = True if len(text_bbox_list) == 2 else False    # 1,2,1,2,... or 2,1,2,1,...
    
    # 获取单双行的划分位置
    char_spacing_h = round(col_width * char_spacing[0])
    head_y1, tail_y2 = head_tail_list[0][0], head_tail_list[-1][1]
    split_pos = [head_y1,]
    for i in range(len(head_tail_list) - 1):
        y_cent = (head_tail_list[i][1] + head_tail_list[i + 1][0]) // 2
        split_pos.append(y_cent)
    split_pos.append(tail_y2)
    
    return y, text_bbox_list, split_pos


def generate_char_img_into_unclosed_box(np_background, x1, y1, x2=None, y2=None, char_spacing=(0.05, 0.05)):
    if x2 is None and y2 is None:
        raise ValueError("There is one and only one None in (x2, y2).")
    if x2 is not None and y2 is not None:
        raise ValueError("There is one and only one None in (x2, y2).")
    
    # 图片为黑底白字
    chinese_char, PIL_char_img = next(Char_Image_Generator)
    
    # 随机决定是否对汉字图片进行旋转，以及旋转的角度
    if random.random() < 0.35:
        PIL_char_img = rotate_PIL_image(PIL_char_img, rotate_angle=random.randint(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE))

    # 转为numpy格式
    np_char_img = np.array(PIL_char_img, dtype=np.uint8)

    if chinese_char in IMPORTANT_CHARS:
        pass
    else:
        # 查找字体的最小包含矩形
        left, right, top, low = find_min_bound_box(np_char_img)
        np_char_img = np_char_img[top:low + 1, left:right + 1]

    char_img_height, char_img_width = np_char_img.shape[:2]

    if x2 is None:  # 文本横向排列
        row_h = y2 - y1 + 1
        char_spacing_h = round(row_h * char_spacing[0])
        char_spacing_w = round(row_h * char_spacing[1])
        box_x1 = x1 + char_spacing_w
        box_y1 = y1 + char_spacing_h
        box_y2 = y2 - char_spacing_h
        box_h = box_y2 - box_y1 + 1

        if char_img_height * 1.4 < char_img_width:
            # 对于“一”这种高度很小、宽度很大的字，应该生成正方形的字图片
            box_w = box_h
            np_char_img = adjust_img_and_put_into_background(np_char_img, background_size=box_h)
        else:
            # 对于宽高相差不大的字，高度撑满，宽度随意
            box_w = round(char_img_width * box_h / char_img_height)
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        box_x2 = box_x1 + box_w - 1
    
    else:  # y2 is None, 文本纵向排列
        col_w = x2 - x1 + 1
        char_spacing_h = round(col_w * char_spacing[0])
        char_spacing_w = round(col_w * char_spacing[1])
        box_x1 = x1 + char_spacing_w
        box_x2 = x2 - char_spacing_w
        box_y1 = y1 + char_spacing_h
        box_w = box_x2 - box_x1 + 1
        
        if char_img_width * 1.4 < char_img_height:
            # 对于“卜”这种高度很大、宽度很小的字，应该生成正方形的字图片
            box_h = box_w
            np_char_img = adjust_img_and_put_into_background(np_char_img, background_size=box_w)
        else:
            # 对于宽高相差不大的字，宽度撑满，高度随意
            box_h = round(char_img_height * box_w / char_img_width)
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        box_y2 = box_y1 + box_h - 1

    # 将生成的汉字图片放入背景图片
    try:
        np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = np_char_img
    except ValueError as e:
        # print('Exception:', e)
        # print("The size of char_img is larger than the length of (y1, x1) to edge. Now, resize char_img ...")
        if x2 is None:
            box_x2 = np_background.shape[1] - 1
            box_w = box_x2 - box_x1 + 1
        else:
            box_y2 = np_background.shape[0] - 1
            box_h = box_y2 - box_y1 + 1
        np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        np_background[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = np_char_img

    # 包围汉字的最小box作为bounding-box
    # bounding_box = (box_x1, box_y1, box_x2, box_y2)
    
    # 随机选定汉字图片的bounding-box
    bbox_x1 = random.randint(x1, box_x1)
    bbox_y1 = random.randint(y1, box_y1)
    bbox_x2 = min(random.randint(box_x2, box_x2+char_spacing_w), np_background.shape[1]-1)
    bbox_y2 = min(random.randint(box_y2, box_y2+char_spacing_h), np_background.shape[0]-1)
    bounding_box =(bbox_x1, bbox_y1, bbox_x2, bbox_y2)
    
    char_box_tail = box_x2+1 if x2 is None else box_y2+1

    return chinese_char, bounding_box, char_box_tail


# 对字体图像做等比例缩放
def resize_img_by_opencv(np_img, obj_size):
    cur_height, cur_width = np_img.shape[:2]
    obj_width, obj_height = obj_size

    # cv.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # dsize为目标图像大小，(fx, fy)为(横, 纵)方向的缩放比例，参数dsize和参数(fx, fy)不必同时传值
    # interpolation为插值方法，共有5种：INTER_NEAREST 最近邻插值法，INTER_LINEAR 双线性插值法(默认)，
    # INTER_AREA 基于局部像素的重采样，INTER_CUBIC 基于4x4像素邻域的3次插值法，INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值
    # 如果是缩小图片，效果最好的是INTER_AREA；如果是放大图片，效果最好的是INTER_CUBIC(slow)或INTER_LINEAR(faster but still looks OK)
    if obj_height == cur_height and obj_width == cur_width:
        return np_img
    elif obj_height + obj_width < cur_height + cur_width:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    resized_np_img = cv2.resize(np_img, dsize=(obj_width, obj_height), interpolation=interpolation)

    return resized_np_img


def chinese_char_img_generator_using_font(img_size=64):
    print("Get all_chinese_list ...")
    all_chinese_list = list(CHAR2ID_DICT.keys())

    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]

    PIL_images_list = []
    while True:
        random.shuffle(font_file_list)
        total = len(font_file_list)
        count = 0
        for font_file in font_file_list:
            count += 1
            print("Char image generator: %d of %d" % (count, total))
            
            random.shuffle(all_chinese_list)
            for chinese_char in all_chinese_list:
                if chinese_char in IGNORABLE_CHARS:
                    continue
                
                # 生成字体图片
                bigger_PIL_img = generate_bigger_image_by_font(chinese_char, font_file, img_size)
                # 检查生成的灰度图像是否可用，黑底白字
                image_data = list(bigger_PIL_img.getdata())
                if sum(image_data) < 10:
                    continue
                
                if chinese_char in IMPORTANT_CHARS:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(6, 10)
                else:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(3, 6)

                if len(PIL_images_list) > 10000:
                    random.shuffle(PIL_images_list)
                    for i in range(5000):
                        # 生成一对(chinese_char，bigger_PIL_img)
                        yield PIL_images_list.pop()


def chinese_char_img_generator_using_image():
    print("Get all calligraphy categories ...")
    calligraphy_categories_list = [os.path.join(EXTERNEL_IMAGES_DIR, content)
                                   for content in os.listdir(EXTERNEL_IMAGES_DIR)
                                   if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]

    PIL_images_list = []
    while True:
        total = len(calligraphy_categories_list)
        count = 0
        for font_text_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):
            count += 1
            print("Char image generator: %d of %d" % (count, total))
            
            for image_path in image_paths_list:
                chinese_char = os.path.basename(image_path)[0]
                if chinese_char in IGNORABLE_CHARS:
                    continue

                # 加载外部图片，将图片调整为正方形
                # 为了保证图片旋转时不丢失信息，生成的图片应该比本来的图片稍微bigger
                # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
                try:
                    bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
                except OSError:
                    # print("The image %s result in OSError !" % image_path)
                    continue

                if chinese_char in IMPORTANT_CHARS:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(6, 10)
                else:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(3, 6)

                if len(PIL_images_list) > 10000:
                    random.shuffle(PIL_images_list)
                    for i in range(5000):
                        # 生成一对(chinese_char，bigger_PIL_img)
                        yield PIL_images_list.pop()


Char_Image_Generator = chinese_char_img_generator_using_font()
# Char_Image_Generator = chinese_char_img_generator_using_image()


""" ****************** 检查生成的tfrecords文件是否可用 ******************* """


def display_tfrecords(tfrecords_file):
    data_set = tf.data.TFRecordDataset([tfrecords_file])
    
    def parse_func(serialized_example):
        return tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'bytes_chars': tf.io.FixedLenFeature([], tf.string),
                'labels': tf.io.FixedLenFeature([], tf.string),
                'gt_boxes': tf.io.FixedLenFeature([], tf.string),
                "split_positions": tf.io.FixedLenFeature([], tf.string)
            })
    
    data_set = data_set.map(parse_func)
    
    for features in data_set.take(1):
        img_h = features['img_height']
        img_w = features['img_width']
        image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
        image = tf.reshape(image_raw, shape=[img_h, img_w])
        PIL_img = Image.fromarray(image.numpy())
        # PIL_img.show()
        
        chars = features["bytes_chars"].numpy().decode("utf-8")
        print(chars)
        
        labels = tf.io.decode_raw(features["labels"], tf.int32).numpy()
        print(labels)
        
        gt_boxes = tf.io.decode_raw(features["gt_boxes"], tf.int32)
        gt_boxes = tf.reshape(gt_boxes, shape=(-1, 4)).numpy()
        print(gt_boxes)


if __name__ == '__main__':
    # generate_one_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_one_text_line_imgs(obj_num=100, text_type="vertical")
    # generate_one_text_line_tfrecords(obj_num=100, text_type="horizontal")
    # generate_one_text_line_tfrecords(obj_num=100, text_type="vertical")
    #
    # generate_two_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_two_text_line_imgs(obj_num=100, text_type="vertical")
    # generate_two_text_line_tfrecords(obj_num=100, text_type="horizontal")
    # generate_two_text_line_tfrecords(obj_num=100, text_type="vertical")
    #
    # generate_mix_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_mix_text_line_imgs(obj_num=100, text_type="vertical")
    # generate_mix_text_line_tfrecords(obj_num=100, text_type="horizontal")
    # generate_mix_text_line_tfrecords(obj_num=100, text_type="vertical")
    
    # display_tfrecords(os.path.join(ONE_TEXT_LINE_TFRECORDS_H, "text_lines_0.tfrecords"))
    
    print("Done !")
