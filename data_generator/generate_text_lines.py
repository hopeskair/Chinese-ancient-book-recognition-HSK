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

from config import TEXT_LINE_IMGS_H, TEXT_LINE_TAGS_FILE_H
from config import TEXT_LINE_IMGS_V, TEXT_LINE_TAGS_FILE_V
from config import TEXT_LINE_TFRECORDS_H, TEXT_LINE_TFRECORDS_V
from config import FONT_FILE_DIR, EXTERNEL_IMAGES_DIR, MAX_ROTATE_ANGLE
from utils import CHAR2ID_DICT, IGNORABLE_CHARS, IMPORTANT_CHARS

from utils import check_or_makedirs
from data_generator.img_utils import rotate_PIL_image
from data_generator.img_utils import find_min_bound_box
from data_generator.img_utils import adjust_img_and_put_into_background
from data_generator.img_utils import reverse_image_color
from data_generator.img_utils import generate_bigger_image_by_font
from data_generator.img_utils import load_external_image_bigger
from data_generator.generate_chinese_images import get_external_image_paths


def generate_text_line_imgs(obj_num=100, type="horizontal", text_shape=None):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    if type == "h":
        text_line_imgs_dir, text_line_tags_file = TEXT_LINE_IMGS_H, TEXT_LINE_TAGS_FILE_H
    if type == "v":
        text_line_imgs_dir, text_line_tags_file = TEXT_LINE_IMGS_V, TEXT_LINE_TAGS_FILE_V
    
    check_or_makedirs(text_line_imgs_dir)
    
    with open(text_line_tags_file, "w", encoding="utf-8") as fw:
        for i in range(obj_num):
            if text_shape is None and type == "h":
                text_shape = (random.randint(36, 96), random.randint(360, 960))
            if text_shape is None and type == "v":
                text_shape = (random.randint(360, 960), random.randint(36, 96))
            
            PIL_text, chinese_char_and_box_list = create_text_line(text_shape, type=type)
            
            img_name = "book_pages_%d.jpg" % i
            save_path = os.path.join(text_line_imgs_dir, img_name)
            PIL_text.save(save_path, format="jpeg")
            fw.write(img_name + "\t" + json.dumps(chinese_char_and_box_list) + "\n")

            if i % 50 == 0:
                print("Process bar: %.2f%%" % (i*100/obj_num))
                sys.stdout.flush()


def generate_text_line_tfrecords(obj_num=100, type="horizontal", text_shape=None):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    if type == "h":
        text_line_tfrecords_dir = TEXT_LINE_TFRECORDS_H
    if type == "v":
        text_line_tfrecords_dir = TEXT_LINE_TFRECORDS_V
    
    check_or_makedirs(text_line_tfrecords_dir)

    # 可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(text_line_tfrecords_dir, "text_lines_%d.tfrecords" % i))
         for i in range(20)]

    # 保存生成的文本图片
    for i in range(obj_num):
        writer = random.choice(writers_list)

        if text_shape is None and type == "h":
            text_shape = (random.randint(36, 96), random.randint(360, 960))
        if text_shape is None and type == "v":
            text_shape = (random.randint(360, 960), random.randint(36, 96))

        PIL_text, chinese_char_and_box_list = create_text_line(text_shape, type=type)

        bytes_image = PIL_text.tobytes()  # 将图片转化为原生bytes
        bytes_chars = "".join([chinese_char for chinese_char, gt_box in chinese_char_and_box_list]).encode("utf-8")
        labels = np.array([CHAR2ID_DICT[char] for char, gt_box in chinese_char_and_box_list], dtype=np.int32).tobytes()
        gt_boxes = np.array([gt_box for chinese_char, gt_box in chinese_char_and_box_list], dtype=np.int32).tobytes()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_text.width])),
                    'bytes_chars': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_chars])),
                    'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels])),
                    'gt_boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_boxes]))
                }))
        writer.write(example.SerializeToString())

        if i % 50 == 0:
            print("Process bar: %.2f%%" % (i*100/obj_num))
            sys.stdout.flush()

    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def create_text_line(shape=(96, 480), type="horizontal"):
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    text_h, text_w = shape
    if type == "h":
        assert text_h <= text_w, "Horizontal text must meet height <= width."
    if type == "v":
        assert text_h >= text_w, "Vertical text must meet height >= width."
    
    # 生成黑色背景
    np_text = np.zeros(shape=(text_h, text_w), dtype=np.uint8)

    # 横向排列
    if type == "h":
        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.0, 0.05), random.uniform(0.0, 0.2))  # (高方向, 宽方向)

        # 生成一行汉字
        y1, y2 = 0, text_h -1
        x = 0
        _, chinese_char_and_box_list, _ = generate_one_row_chars(x, y1, y2, text_w, np_text, char_spacing)

    # 纵向排列
    else:
        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.0, 0.2), random.uniform(0.0, 0.05))  # (高方向, 宽方向)

        # 生成一列汉字
        x1, x2 = 0, text_w - 1
        y = 0
        _, chinese_char_and_box_list, _ = generate_one_col_chars(x1, x2, y, text_h, np_text, char_spacing)
    
    np_text = reverse_image_color(np_img=np_text)
    PIL_text = Image.fromarray(np_text)

    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_text.show()

    return PIL_text, chinese_char_and_box_list


def generate_one_row_chars(x, y1, y2, length, np_background, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    row_height = y2 - y1 + 1
    while length >= row_height:
        chinese_char, bounding_box, x_tail = \
            generate_char_img_into_unclosed_box(np_background, x1=x, y1=y1, x2=None, y2=y2, char_spacing=char_spacing)

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = x_tail - x
        length -= added_length
        x = x_tail
    
    # 获取文本行的bounding-box
    head_x1, head_y1, head_x2, head_y2 = char_and_box_list[0][1]
    tail_x1, tail_y1, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)

    return x, char_and_box_list, text_bbox


def generate_two_rows_chars(x, y1, y2, length, np_background, char_spacing):
    row_height = y2 - y1 + 1
    mid_y = y1 + round(row_height / 2)

    x_1, _, text1_bbox = generate_one_row_chars(x, y1, mid_y, length, np_background, char_spacing)
    x_2, _, text2_bbox = generate_one_row_chars(x, mid_y+1, y2, length, np_background, char_spacing)

    return max(x_1, x_2), text1_bbox, text2_bbox


def generate_one_col_chars(x1, x2, y, length, np_background, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    col_width = x2 - x1 + 1
    while length >= col_width:
        chinese_char, bounding_box, y_tail = \
            generate_char_img_into_unclosed_box(np_background, x1=x1, y1=y, x2=x2, y2=None, char_spacing=char_spacing)

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = y_tail - y
        length -= added_length
        y = y_tail

    # 获取文本行的bounding-box
    head_x1, head_y1, head_x2, head_y2 = char_and_box_list[0][1]
    tail_x1, tail_y1, tail_x2, tail_y2 = char_and_box_list[-1][1]
    text_bbox = (head_x1, head_y1, tail_x2, tail_y2)

    return y, char_and_box_list, text_bbox


def generate_two_cols_chars(x1, x2, y, length, np_background, char_spacing):
    col_width = x2 - x1 + 1
    mid_x = x1 + round(col_width / 2)

    y_1, _, text1_box = generate_one_col_chars(x1, mid_x, y, length, np_background, char_spacing)
    y_2, _, text2_box = generate_one_col_chars(mid_x+1, x2, y, length, np_background, char_spacing)

    return max(y_1, y_2), text1_box, text2_box


def generate_char_img_into_unclosed_box(np_background, x1, y1, x2=None, y2=None, char_spacing=(0.05, 0.05)):
    if x2 is None and y2 is None:
        raise ValueError("There is one and only one None in (x2, y2).")
    if x2 is not None and y2 is not None:
        raise ValueError("There is one and only one None in (x2, y2).")
    
    # 图片为黑底白字
    chinese_char, PIL_char_img = next(Char_Image_Generator)
    
    # 随机决定是否对汉字图片进行旋转，以及旋转的角度
    if random.random() < 0.2:
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
        print('Exception:', e)
        print("The size of char_img is larger than the length of (y1, x1) to background_img's edge.")
        print("Now, try another char_img ...")
        return generate_char_img_into_unclosed_box(np_background, x1, y1, x2, y2, char_spacing)

    # 包围汉字的最小box作为bounding-box
    # bounding_box = (box_x1, box_y1, box_x2, box_y2)
    
    # 随机选定汉字图片的bounding-box
    bbox_x1 = random.randint(x1, box_x1)
    bbox_y1 = random.randint(y1, box_y1)
    bbox_x2 = random.randint(box_x2, box_x2+char_spacing_w)
    bbox_y2 = random.randint(box_y2, box_y2+char_spacing_h)
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


def chinese_char_img_generator_using_font(img_size=96):
    print("Get all_chinese_list ...")
    all_chinese_list = list(CHAR2ID_DICT.keys())

    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)]

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
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(5, 8)
                else:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(2, 5)

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
        for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):
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
                'gt_boxes': tf.io.FixedLenFeature([], tf.string)
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
    # generate_text_line_imgs(obj_num=100, type="horizontal")
    # generate_text_line_imgs(obj_num=100, type="vertical")
    # generate_text_line_tfrecords(obj_num=100, type="horizontal")
    # generate_text_line_tfrecords(obj_num=100, type="vertical")
    
    # display_tfrecords(os.path.join(TEXT_LINE_TFRECORDS_H, "text_lines_0.tfrecords"))
    
    print("Done !")
