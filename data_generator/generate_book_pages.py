# -*- encoding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
import json

from chinese_image_processing import rotate_PIL_image
from chinese_image_processing import find_min_bound_box
from chinese_image_processing import adjust_img_and_put_into_Background
from chinese_image_processing import reverse_image_color
from chinese_image_processing import generate_bigger_image_by_font
from generate_chinese_image import get_external_image_paths
from chinese_image_processing import load_external_image_bigger
import basic_tools
neglected_chars_list = basic_tools.neglected_chinese_chars()
important_chars_list = basic_tools.important_chars_to_repeat()

from basic_setting import Font_File_Dir, Non_Chars_Images_Dir
from basic_setting import Chinese_Char_Detection_Root_Dir

book_page_data_set = os.path.join(Chinese_Char_Detection_Root_Dir, "data_set")
book_page_train_set = os.path.join(book_page_data_set, "train_set")
book_page_test_set = os.path.join(book_page_data_set, "test_set")
book_pages_train_info_file = os.path.join(book_page_data_set, "book_pages_train_info.txt")
book_pages_test_info_file = os.path.join(book_page_data_set, "book_pages_test_info.txt")
book_page_train_tfrecords_dir = os.path.join(book_page_data_set, "train_tfrecords")
book_page_test_tfrecords_dir = os.path.join(book_page_data_set, "test_tfrecords")


def generate_and_save_book_pages(pages_num=100):
    if not os.path.exists(book_page_train_set):
        os.makedirs(book_page_train_set)
    if not os.path.exists(book_page_test_set):
        os.makedirs(book_page_test_set)

    with open(book_pages_train_info_file, "w", encoding="utf-8") as f_train:
        with open(book_pages_test_info_file, "w", encoding="utf-8") as f_test:
            for i in range(pages_num):
                # PIL_page, chinese_char_and_box_list = create_book_page()
                PIL_page, chinese_char_and_box_list = create_line_text_image()

                # 让train_set和test_set的比例约为 4:1
                if random.random() < 0.85:
                    img_name = "train_book_page_%d.jpg" % i
                    save_path = os.path.join(book_page_train_set, img_name)
                    fw = f_train
                else:
                    img_name = "test_book_page_%d.jpg" % i
                    save_path = os.path.join(book_page_test_set, img_name)
                    fw = f_test
                PIL_page.save(save_path, format="jpeg")
                fw.write(save_path + "\t" + json.dumps(chinese_char_and_box_list) + "\n")

                if i % 10 == 0:
                    print("Rate of Process: %.2f%%" % (i * 100 / pages_num))
                    sys.stdout.flush()


def generate_book_pages_tfrecords_for_Train(pages_num=100):
    if not os.path.exists(book_page_train_tfrecords_dir):
        os.makedirs(book_page_train_tfrecords_dir)
    if not os.path.exists(book_page_test_tfrecords_dir):
        os.makedirs(book_page_test_tfrecords_dir)

    # 我们可以把生成的图片直接存入tfrecords文件
    # 而不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    # 将书页图片分多个tfrecords文件存放
    writers_list = \
        [tf.python_io.TFRecordWriter(
            os.path.join(book_page_train_tfrecords_dir, "text_line_train_set_by_font_%d.tfrecords" % i))
         for i in range(15)] + \
        [tf.python_io.TFRecordWriter(
            os.path.join(book_page_test_tfrecords_dir, "text_line_test_set_by_font_%d.tfrecords" % i))
         for i in range(2)]

    # 保存生成的书页图片
    for i in range(pages_num):
        # train_set和test_set的比例约为 5:1
        writer = random.choice(writers_list)

        # PIL_page, chinese_char_and_box_list = create_book_page()
        PIL_page, chinese_char_and_box_list = create_line_text_image()

        bytes_image = PIL_page.tobytes()  # 将图片转化为原生bytes
        gt_boxes_np_array = np.array([gt_box for chinese_char, gt_box in chinese_char_and_box_list],
                                     dtype=np.int32).tobytes()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_page.height])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_page.width])),
                    'gt_boxes_np_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_boxes_np_array]))
                }))
        writer.write(example.SerializeToString())

        if i % 10 == 0:
            print("Rate of Process: %.2f%%" % (i * 100 / pages_num))
            sys.stdout.flush()

    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]
    return


def create_book_page():
    # 随机生成一张黑色背景书页
    page_height = random.randint(560, 1080)
    page_width = random.randint(560, 1080)
    np_page = np.zeros(shape=(page_height, page_width), dtype=np.uint8)

    PIL_page = Image.fromarray(np_page)
    draw = ImageDraw.Draw(PIL_page)

    # 随机生成书页边框
    margin_w = round(random.uniform(0.01, 0.05) * page_width)
    margin_h = round(random.uniform(0.01, 0.05) * page_height)
    margin_line_thickness = random.randint(2, 8)
    # 点的坐标格式为(x, y)，不是(y, x)
    draw.rectangle([(margin_w, margin_h), (page_width - margin_w, page_height - margin_h)],
                   fill=None, outline="white", width=margin_line_thickness)

    # 记录下生成的汉字及其bounding-box
    chinese_char_and_box_list = []

    # 随机决定文本是横向排列，还是纵向排列
    if random.random() < 0.4:
        # 横向排列

        # 随机决定文本的行数
        rows_num = random.randint(6, 10)
        row_spacing = (page_height - 2 * margin_h) / rows_num

        # y-coordinate划分行
        ys = [margin_h + round(i * row_spacing) for i in range(rows_num)] + [page_height - margin_h, ]

        # 画行线，第一条线和最后一条线是边缘线，不需要画
        line_thickness = random.randint(1, margin_line_thickness)
        for y in ys[1:-1]:
            draw.line([(margin_w, y), (page_width - margin_w, y)], fill="white", width=line_thickness)

        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.02, 0.1), random.uniform(0.05, 0.15))  # (宽方向, 高方向)

        # 转为numpy格式
        np_page = np.array(PIL_page, dtype=np.uint8)

        # 逐行生成汉字
        for i in range(len(ys) - 1):
            y1, y2 = ys[i], ys[i + 1]
            x = margin_w + int(random.uniform(0.5, 1) * margin_line_thickness)
            char_height = y2 - y1
            row_length = page_width - margin_w - x
            while row_length >= char_height:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(char_height, row_length)

                # 随机决定字串是单行字串还是双行字串
                if random.random() < 0.6:
                    x, char_and_box_list = generate_one_row_chars(x, y1, y2, length, np_page, char_spacing)
                else:
                    x, char_and_box_list = generate_two_rows_chars(x, y1, y2, length, np_page, char_spacing)
                row_length = page_width - margin_w - x
                chinese_char_and_box_list.extend(char_and_box_list)
    else:
        # 纵向排列

        # 随机决定文本的列数
        cols_num = random.randint(6, 10)
        col_spacing = (page_width - 2 * margin_w) / cols_num

        # x-coordinate划分列
        xs = [margin_w + round(i * col_spacing) for i in range(cols_num)] + [page_width - margin_w, ]

        # 画列线，第一条线和最后一条线是边缘线，不需要画
        line_thickness = random.randint(1, margin_line_thickness)
        for x in xs[1:-1]:
            draw.line([(x, margin_h), (x, page_height - margin_h)], fill="white", width=line_thickness)

        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.05, 0.15), random.uniform(0.02, 0.1))  # (宽方向, 高方向)

        # 转为numpy格式
        np_page = np.array(PIL_page, dtype=np.uint8)

        # 逐列生成汉字，最右边为第一列
        for i in range(len(xs) - 1, 0, -1):
            x1, x2 = xs[i - 1], xs[i]
            y = margin_h + int(random.uniform(0.5, 1) * margin_line_thickness)
            char_width = x2 - x1
            col_length = page_height - margin_h - y
            while col_length >= char_width:
                # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
                length = random.randint(char_width, col_length)

                # 随机决定字串是单列字串还是双列字串
                if random.random() < 0.6:
                    y, char_and_box_list = generate_one_col_chars(x1, x2, y, length, np_page, char_spacing)
                else:
                    y, char_and_box_list = generate_two_cols_chars(x1, x2, y, length, np_page, char_spacing)
                col_length = page_height - margin_h - y
                chinese_char_and_box_list.extend(char_and_box_list)

    # 将黑底白字转换为白底黑字
    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_page.show()

    return PIL_page, chinese_char_and_box_list


def create_line_text_image():
    # 随机决定文本行是否使用边框线（分隔线）
    separator = True if random.random() < 0.4 else False

    # 记录下生成的汉字及其bounding-box
    chinese_char_and_box_list = []

    # 随机决定文本是横向排列，还是纵向排列
    if random.random() < 0.4:
        # 横向排列

        # 随机生成一张黑色背景的文本行
        page_height = random.randint(64, 108)
        page_width = random.randint(560, 1080)
        np_page = np.zeros(shape=(page_height, page_width), dtype=np.uint8)

        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

        # 随机生成边框
        margin_w = random.randint(0, 20)
        margin_h = random.randint(0, 3)
        margin_line_thickness = random.randint(1, 5)

        if separator:
            draw.rectangle([(margin_w, margin_h), (page_width - margin_w, page_height - margin_h)],
                           fill=None, outline="white", width=margin_line_thickness)

        # 随机决定字符间距占行距的比例
        char_spacing = (random.uniform(0.02, 0.1), random.uniform(0.05, 0.15))  # (宽方向, 高方向)

        # 转为numpy格式
        np_page = np.array(PIL_page, dtype=np.uint8)

        # 生成一行汉字
        y1, y2 = margin_h, page_height - margin_h
        x = margin_w + int(random.uniform(0.5, 1) * margin_line_thickness)
        char_height = y2 - y1
        row_length = page_width - margin_w - x
        while row_length >= char_height:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(char_height, row_length)

            # 随机决定字串是单行字串还是双行字串
            if random.random() < 0.6:
                x, char_and_box_list = generate_one_row_chars(x, y1, y2, length, np_page, char_spacing)
            else:
                x, char_and_box_list = generate_two_rows_chars(x, y1, y2, length, np_page, char_spacing)
            row_length = page_width - margin_w - x
            chinese_char_and_box_list.extend(char_and_box_list)

    else:
        # 纵向排列

        # 随机生成一张黑色背景的文本行
        page_height = random.randint(560, 1080)
        page_width = random.randint(64, 108)
        np_page = np.zeros(shape=(page_height, page_width), dtype=np.uint8)

        PIL_page = Image.fromarray(np_page)
        draw = ImageDraw.Draw(PIL_page)

        # 随机生成边框
        margin_w = random.randint(0, 3)
        margin_h = random.randint(0, 20)
        margin_line_thickness = random.randint(1, 5)

        if separator:
            draw.rectangle([(margin_w, margin_h), (page_width - margin_w, page_height - margin_h)],
                           fill=None, outline="white", width=margin_line_thickness)

        # 随机决定字符间距占列距的比例
        char_spacing = (random.uniform(0.05, 0.15), random.uniform(0.02, 0.1))  # (宽方向, 高方向)

        # 转为numpy格式
        np_page = np.array(PIL_page, dtype=np.uint8)

        # 生成一列汉字
        x1, x2 = margin_w, page_width - margin_w
        y = margin_h + int(random.uniform(0.5, 1) * margin_line_thickness)
        char_width = x2 - x1
        col_length = page_height - margin_h - y
        while col_length >= char_width:
            # 随机决定接下来的字串长度（这是大约数值，实际可能比它小,也可能比它大）
            length = random.randint(char_width, col_length)

            # 随机决定字串是单列字串还是双列字串
            if random.random() < 0.6:
                y, char_and_box_list = generate_one_col_chars(x1, x2, y, length, np_page, char_spacing)
            else:
                y, char_and_box_list = generate_two_cols_chars(x1, x2, y, length, np_page, char_spacing)
            col_length = page_height - margin_h - y
            chinese_char_and_box_list.extend(char_and_box_list)

    np_page = reverse_image_color(np_img=np_page)
    PIL_page = Image.fromarray(np_page)

    # print(chinese_char_and_box_list)
    # print(len(chinese_char_and_box_list))
    # PIL_page.show()

    return PIL_page, chinese_char_and_box_list


def generate_one_row_chars(x, y1, y2, length, np_page, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    row_height = y2 - y1
    while length >= row_height:
        chinese_char, bounding_box, char_box_tail = \
            generate_char_img_into_unclosed_box(np_page, x1=x, y1=y1, x2=None, y2=y2, char_spacing=char_spacing)

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = char_box_tail - x
        length -= added_length
        x = char_box_tail

    return x, char_and_box_list


def generate_two_rows_chars(x, y1, y2, length, np_page, char_spacing):
    row_height = y2 - y1
    mid_y = y1 + round(row_height / 2)

    x_1, char_and_box_list_1 = generate_one_row_chars(x, y1, mid_y, length, np_page, char_spacing)
    x_2, char_and_box_list_2 = generate_one_row_chars(x, mid_y, y2, length, np_page, char_spacing)

    return max(x_1, x_2), char_and_box_list_1 + char_and_box_list_2


def generate_one_col_chars(x1, x2, y, length, np_page, char_spacing):
    # 记录下生成的汉字及其bounding-box
    char_and_box_list = []

    col_width = x2 - x1
    while length >= col_width:
        chinese_char, bounding_box, char_box_tail = \
            generate_char_img_into_unclosed_box(np_page, x1=x1, y1=y, x2=x2, y2=None, char_spacing=char_spacing)

        char_and_box_list.append((chinese_char, bounding_box))
        added_length = char_box_tail - y
        length -= added_length
        y = char_box_tail

    return y, char_and_box_list


def generate_two_cols_chars(x1, x2, y, length, np_page, char_spacing):
    col_width = x2 - x1
    mid_x = x1 + round(col_width / 2)

    y_1, char_and_box_list_1 = generate_one_col_chars(x1, mid_x, y, length, np_page, char_spacing)
    y_2, char_and_box_list_2 = generate_one_col_chars(mid_x, x2, y, length, np_page, char_spacing)

    return max(y_1, y_2), char_and_box_list_1 + char_and_box_list_2


def generate_char_img_into_unclosed_box(np_page, x1, y1, x2=None, y2=None, char_spacing=(0, 0)):
    if x2 is None and y2 is None:
        raise ValueError("x2 is None and y2 is None")
    if x2 is not None and y2 is not None:
        raise ValueError("x2 is not None and y2 is not None")

    chinese_char, PIL_char_img = next(Char_Image_Generator)

    # 随机决定是否对汉字图片进行轻微旋转，以及旋转的角度
    if random.random() < 0.2:
        PIL_char_img = rotate_PIL_image(PIL_char_img, rotate_angle=random.randint(-5, 5))

    # 转为numpy格式
    np_char_img = np.array(PIL_char_img, dtype=np.uint8)

    if chinese_char in important_chars_list:
        pass
    else:
        # 查找字体的最小包含矩形
        left, right, top, low = find_min_bound_box(np_char_img)
        np_char_img = np_char_img[top:low + 1, left:right + 1]

    char_img_height, char_img_width = np_char_img.shape[:2]

    if x2 is None:
        # 文本横向排列
        row_spacing = y2 - y1
        char_spacing_w = round(row_spacing * char_spacing[0])
        char_spacing_h = round(row_spacing * char_spacing[1])
        box_x1 = x1 + char_spacing_w
        box_y1 = y1 + char_spacing_h
        box_y2 = y2 - char_spacing_h
        box_h = box_y2 - box_y1 + 1

        if char_img_height * 1.4 < char_img_width:
            # 对于“一”这种高度很小、宽度很大的字，应该生成正方形的字图片
            box_w = box_h
            np_char_img = adjust_img_and_put_into_Background(np_char_img, background_size=box_h)
        else:
            # 对于宽高相差不大的字，高度撑满，宽度随意
            box_w = round(char_img_width * box_h / char_img_height)
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        box_x2 = box_x1 + box_w - 1
    else:
        # y2 is None, 文本纵向排列
        col_spacing = x2 - x1
        char_spacing_w = round(col_spacing * char_spacing[0])
        char_spacing_h = round(col_spacing * char_spacing[1])
        box_x1 = x1 + char_spacing_w
        box_x2 = x2 - char_spacing_w
        box_y1 = y1 + char_spacing_h
        box_w = box_x2 - box_x1 + 1

        if char_img_width * 1.4 < char_img_height:
            # 对于“卜”这种高度很大、宽度很小的字，应该生成正方形的字图片
            box_h = box_w
            np_char_img = adjust_img_and_put_into_Background(np_char_img, background_size=box_w)
        else:
            # 对于宽高相差不大的字，宽度撑满，高度随意
            box_h = round(char_img_height * box_w / char_img_width)
            np_char_img = resize_img_by_opencv(np_char_img, obj_size=(box_w, box_h))
        box_y2 = box_y1 + box_h - 1

    # 将生成的汉字图片放入书页图片
    try:
        np_page[box_y1:box_y2 + 1, box_x1:box_x2 + 1] = np_char_img
    except ValueError as e:
        print('Exception:', e)
        print("Caused by the size of char-image being larger than the length of box's (y1, x1) to book-page's edge.")
        print("Now, try another char-image ...")
        return generate_char_img_into_unclosed_box(np_page, x1, y1, x2, y2, char_spacing)

    # 随机选定汉字图片的bounding-box
    # bbox_x1 = random.randint(x1, box_x1)
    # bbox_y1 = random.randint(y1, box_y1)
    # bbox_x2 = random.randint(box_x2, box_x2+char_img_width)
    # bbox_y2 = random.randint(box_y2, box_y2+char_spacing_h)
    # bounding_box =(bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    # 包围汉字的最小box作为bounding-box
    bounding_box = (box_x1, box_y1, box_x2, box_y2)

    char_box_tail = box_x2 if x2 is None else box_y2

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


def non_chinese_char_images():
    non_char_imgs_list = []
    for file_name in os.listdir(Non_Chars_Images_Dir):
        if file_name.endswith(".gif") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            assert file_name[0] == "＄"

            image_path = os.path.join(Non_Chars_Images_Dir, file_name)
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                # print("The image %s result in OSError !" % image_path)
                continue
            non_char_imgs_list.append(bigger_PIL_img)

    return non_char_imgs_list


def chinese_char_img_generator_by_font(image_size=100):
    print("Get all_chinese_list ...")
    chinese_to_label_dict = basic_tools.chinese_to_label()
    all_chinese_list = list(chinese_to_label_dict.keys())
    if "＄" in all_chinese_list:
        all_chinese_list.remove("＄")

    non_char_imgs_list = non_chinese_char_images()

    print("Get font_file_list ...")
    font_file_list = [os.path.join(Font_File_Dir, font_name) for font_name in os.listdir(Font_File_Dir)]

    PIL_images_list = []
    while True:
        random.shuffle(font_file_list)
        count = 0
        for font_file in font_file_list:
            count += 1
            print(count)
            random.shuffle(all_chinese_list)
            for chinese_char in all_chinese_list:
                if chinese_char in neglected_chars_list:
                    continue

                # 生成字体图片
                bigger_PIL_img = generate_bigger_image_by_font(chinese_char, font_file, image_size)
                # 检查生成的灰度图像是否可用，黑底白字
                image_data = list(bigger_PIL_img.getdata())
                if sum(image_data) < 10:
                    continue

                if chinese_char in important_chars_list:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(6, 12)
                else:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(2, 5)

                if len(PIL_images_list) > 10000:
                    PIL_images_list += [("＄", bigger_PIL_img) for bigger_PIL_img in non_char_imgs_list]
                    random.shuffle(PIL_images_list)
                    for i in range(3000):
                        # 生成一对(chinese_char，bigger_PIL_img)
                        yield PIL_images_list.pop()


def chinese_char_img_generator_by_image():
    non_char_imgs_list = non_chinese_char_images()

    PIL_images_list = []
    while True:
        count = 0
        for font_type, image_paths_list in get_external_image_paths():
            count += 1
            print(count)
            for image_path in image_paths_list:
                chinese_char = os.path.basename(image_path)[0]

                if chinese_char in neglected_chars_list:
                    continue

                # 加载外部图片，将图片调整为正方形
                # 为了保证图片旋转时不丢失信息，生成的图片应该比本来的图片稍微bigger
                # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
                try:
                    bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
                except OSError:
                    # print("The image %s result in OSError !" % image_path)
                    continue

                if chinese_char in important_chars_list:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(6, 12)
                else:
                    PIL_images_list += [(chinese_char, bigger_PIL_img)] * random.randint(3, 6)

                if len(PIL_images_list) > 10000:
                    PIL_images_list += [("＄", bigger_PIL_img) for bigger_PIL_img in non_char_imgs_list]
                    random.shuffle(PIL_images_list)
                    for i in range(3000):
                        # 生成一对(chinese_char，bigger_PIL_img)
                        yield PIL_images_list.pop()


Char_Image_Generator = chinese_char_img_generator_by_font()
# Char_Image_Generator = chinese_char_img_generator_by_image()


if __name__ == '__main__':
    # create_book_page()
    # create_line_text_image()
    # generate_and_save_book_pages(pages_num=500)
    generate_book_pages_tfrecords_for_Train(pages_num=80000)
    print("Done !")
