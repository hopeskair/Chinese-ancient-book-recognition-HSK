# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import sys
import json
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from config import BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
from config import BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
from config import BOOK_PAGE_TFRECORDS_H, BOOK_PAGE_TFRECORDS_V

from util import check_or_makedirs
from data_generator.img_utils import reverse_image_color
from data_generator.generate_text_lines import generate_one_row_chars, generate_two_rows_chars
from data_generator.generate_text_lines import generate_one_col_chars, generate_two_cols_chars


def generate_book_pages(obj_num=10, type="horizontal", page_shape=None):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    if type == "h":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_H, BOOK_PAGE_TAGS_FILE_H
    if type == "v":
        book_page_imgs_dir, book_page_tags_file = BOOK_PAGE_IMGS_V, BOOK_PAGE_TAGS_FILE_V
        
    check_or_makedirs(book_page_imgs_dir)

    with open(book_page_tags_file, "w", encoding="utf-8") as fw:
        for i in range(obj_num):
            if page_shape is None and type == "h":
                page_shape = (random.randint(480, 720), random.randint(720, 1080))
            if page_shape is None and type == "v":
                page_shape = (random.randint(720, 1080), random.randint(480, 720))

            PIL_page, text_bbox_list = create_book_page(page_shape, type=type)
            PIL_page = PIL_page.convert("L")
            
            img_name = "book_page_%d.jpg" % i
            save_path = os.path.join(book_page_imgs_dir, img_name)
            PIL_page.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(text_bbox_list) + "\n")

            if i % 50 == 0:
                print("Process bar: %.2f%%" % (i*100/obj_num))
                sys.stdout.flush()


def generate_book_page_tfrecords(obj_num=10, type="horizontal", init_num=0, page_shape=None):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    if type == "h":
        book_page_tfrecords_dir = BOOK_PAGE_TFRECORDS_H
    if type == "v":
        book_page_tfrecords_dir = BOOK_PAGE_TFRECORDS_V
    
    check_or_makedirs(book_page_tfrecords_dir)
    
    # 我们可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(book_page_tfrecords_dir, "book_pages_%d.tfrecords" % i))
         for i in range(init_num, init_num+20)]
    
    # 保存生成的书页图片
    for i in range(obj_num):
        writer = random.choice(writers_list)
        if page_shape is None and type == "h":
            page_shape = (random.randint(480, 720), random.randint(720, 1080))
        if page_shape is None and type == "v":
            page_shape = (random.randint(720, 1080), random.randint(480, 720))

        PIL_page, text_bbox_list = create_book_page(page_shape, type=type)
        
        bytes_image = PIL_page.tobytes()  # 将图片转化为原生bytes
        text_boxes = np.array([text_box for text_box in text_bbox_list], dtype=np.int32).tobytes()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_page.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_page.width])),
                    'text_boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text_boxes]))
                }))
        writer.write(example.SerializeToString())

        if i % 50 == 0:
            print("Process bar: %.2f%%" % (i*100/obj_num))
            sys.stdout.flush()

    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def create_book_page(shape=(960, 540), type="horizontal"):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    # 黑色背景书页
    np_page = np.zeros(shape=shape, dtype=np.uint8)
    page_height, page_width = shape
    
    # 随机确定是否画边框线及行线
    draw = None
    if random.random() < 0.6:
        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)
    
    # 随机确定书页边框
    margin_w = round(random.uniform(0.01, 0.05) * page_width)
    margin_h = round(random.uniform(0.01, 0.05) * page_height)
    margin_line_thickness = random.randint(2, 6)
    line_thickness = round(margin_line_thickness / 2)
    if draw is not None:
        # 点的坐标格式为(x, y)，不是(y, x)
        draw.rectangle([(margin_w, margin_h), (page_width-1- margin_w, page_height-1- margin_h)],
                       fill=None, outline="white", width=margin_line_thickness)
    
    # 记录下文本行的bounding-box
    text_bbox_list = []
    
    if type == "h":  # 横向排列
        
        # 随机确定文本的行数
        rows_num = random.randint(6, 10)
        row_h = (page_height - 2 * margin_h) / rows_num

        # y-coordinate划分行
        ys = [margin_h + round(i * row_h) for i in range(rows_num)] + [page_height-1-margin_h]
        
        # 画行线，第一条线和最后一条线是边框线，不需要画
        if draw is not None:
            for y in ys[1:-1]:
                draw.line([(margin_w, y), (page_width-1-margin_w, y)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)
        
        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.05, 0.15), random.uniform(0.0, 0.2))  # (高方向, 宽方向)
        
        # 逐行生成汉字
        for i in range(len(ys) - 1):
            y1, y2 = ys[i]+1, ys[i+1]-1
            x = margin_w + int(random.uniform(0.5, 1) * margin_line_thickness)
            char_height = y2 - y1 + 1
            row_length = page_width - x - margin_w
            flag = 0 if random.random() < 0.6 else 1  # 单行字串还是双行字串
            while row_length >= char_height:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(char_height, row_length)
                flag += 1
                if flag%2 == 1:
                    x, _, text_bbox = generate_one_row_chars(x, y1, y2, length, np_page, char_spacing)
                    text_bbox_list.append(text_bbox)
                else:
                    x, text1_bbox, text2_bbox = generate_two_rows_chars(x, y1, y2, length, np_page, char_spacing)
                    text_bbox_list.extend([text1_bbox, text2_bbox])
                row_length = page_width - x - margin_w
    
    else:  # 纵向排列

        # 随机决定文本的列数
        cols_num = random.randint(7, 9)
        col_w = (page_width - 2 * margin_w) / cols_num

        # x-coordinate划分列
        xs = [margin_w + round(i * col_w) for i in range(cols_num)] + [page_width-1-margin_w, ]

        # 画列线，第一条线和最后一条线是边缘线，不需要画
        if draw is not None:
            for x in xs[1:-1]:
                draw.line([(x, margin_h), (x, page_height-1-margin_h)], fill="white", width=line_thickness)
            np_page = np.array(PIL_page, dtype=np.uint8)

        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.05, 0.15))  # (高方向, 宽方向)

        # 逐列生成汉字，最右边为第一列
        for i in range(len(xs) - 1, 0, -1):
            x1, x2 = xs[i-1]+1, xs[i]-1
            y = margin_h + int(random.uniform(0.5, 1) * margin_line_thickness)
            char_width = x2 - x1 + 1
            col_length = page_height - y - margin_h
            flag = 0 if random.random() < 0.6 else 1  # 单行字串还是双行字串
            while col_length >= char_width:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(char_width, col_length)
                flag += 1
                if flag%2 == 1:
                    y, _, text_bbox = generate_one_col_chars(x1, x2, y, length, np_page, char_spacing)
                    text_bbox_list.append(text_bbox)
                else:
                    y, text1_bbox, text2_bbox = generate_two_cols_chars(x1, x2, y, length, np_page, char_spacing)
                    text_bbox_list.extend([text1_bbox, text2_bbox])
                col_length = page_height - y - margin_h
                
    # 将黑底白字转换为白底黑字
    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    # print(text_bbox_list)
    # print(len(text_bbox_list))
    # PIL_page.show()

    return PIL_page, text_bbox_list


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
                'text_boxes': tf.io.FixedLenFeature([], tf.string)
            })
    
    data_set = data_set.map(parse_func)
    
    # for features in data_set.take(1):
    #     img_h = features['img_height']
    #     img_w = features['img_width']
    #     image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
    #
    #     image = tf.reshape(image_raw, shape=[img_h, img_w])
    #     PIL_img = Image.fromarray(image.numpy())
    #     PIL_img.show()
    #
    #     text_boxes = tf.io.decode_raw(features["text_boxes"], tf.int32)
    #     text_boxes = tf.reshape(text_boxes, shape=(-1, 4)).numpy()
    #     print(text_boxes)
    
    def restore_func(features):
        img_h = features['img_height']
        img_w = features['img_width']
        image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
        image = tf.reshape(image_raw, shape=[img_h, img_w])
        
        text_boxes = tf.io.decode_raw(features["text_boxes"], tf.int32)
        text_boxes = tf.reshape(text_boxes, shape=(-1, 4))
        return image, text_boxes
    
    data_set = data_set.map(restore_func)
    
    for i, (image, text_boxes) in enumerate(data_set.take(2)):
        image, text_boxes = image.numpy(), text_boxes.numpy()
        PIL_img = Image.fromarray(image)
        # PIL_img.show()
        print(text_boxes)
        

if __name__ == '__main__':
    # generate_book_pages(obj_num=100, type="horizontal", page_shape=(416, 416))
    # generate_book_pages(obj_num=100, type="vertical", page_shape=(416, 416))
    # generate_book_page_tfrecords(obj_num=100, type="horizontal")
    # generate_book_page_tfrecords(obj_num=8000, type="vertical")
    
    display_tfrecords(os.path.join(BOOK_PAGE_TFRECORDS_V, "book_pages_0.tfrecords"))
    
    print("Done !")
