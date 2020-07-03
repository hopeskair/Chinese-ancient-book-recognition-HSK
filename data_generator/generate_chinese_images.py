# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import sys
import shutil
import random
from PIL import Image
import numpy as np
import tensorflow as tf

from util import CHAR2ID_DICT, BLANK_CHAR, TRADITION_CHARS
from config import CHAR_IMG_SIZE, NUM_IMAGES_PER_FONT
from config import FONT_FILE_DIR, EXTERNEL_IMAGES_DIR
from config import CHAR_IMGS_DIR, CHAR_TFRECORDS_DIR

from util import check_or_makedirs, remove_then_makedirs
from data_generator.img_utils import get_standard_image, get_augmented_image
from data_generator.img_utils import generate_bigger_image_by_font, load_external_image_bigger


""" ************************ 矢量字体生成训练图片 *************************** """


def generate_all_chinese_images_bigger(font_file, image_size=int(CHAR_IMG_SIZE*1.2)):
    all_chinese_list = list(CHAR2ID_DICT.keys())
    if BLANK_CHAR in all_chinese_list:
        all_chinese_list.remove(BLANK_CHAR)

    font_name = os.path.basename(font_file)
    if "繁" in font_name and "简" not in font_name:
        _chinese_chars = TRADITION_CHARS
    else:
        _chinese_chars = "".join(all_chinese_list)
    
    # _chinese_chars = "安愛蒼碧冒葡囊蒡夏啻坌"
    
    for chinese_char in _chinese_chars:
        try:  # 生成字体图片
            bigger_PIL_img = generate_bigger_image_by_font(chinese_char, font_file, image_size)
        except OSError:
            print("OSError: invalid outline, %s, %s"%(font_file, chinese_char))
            continue

        yield chinese_char, bigger_PIL_img


# 生成样例图片，以便检查图片的有效性
def generate_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=False):
    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]
    # font_file_list = [os.path.join(FONT_FINISHED_DIR, "chinese_fonts_暂时移出/康熙字典体完整版本.otf")]
    
    chinese_char_num = len(CHAR2ID_DICT)
    total_num = len(font_file_list) * chinese_char_num
    count = 0
    for font_file in font_file_list:    # 外层循环是字体
        font_name = os.path.basename(font_file)
        font_type = font_name.split(".")[0]
        
        # 创建保存该字体图片的目录
        font_img_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(font_img_dir)
        
        for chinese_char, bigger_PIL_img in generate_all_chinese_images_bigger(font_file, image_size=int(obj_size*1.2)):    # 内层循环是字
            # 检查生成的灰度图像是否可用，黑底白字
            image_data = list(bigger_PIL_img.getdata())
            if sum(image_data) < 10:
                continue
            
            if not augmentation:
                PIL_img = get_standard_image(bigger_PIL_img, obj_size, reverse_color=True)
            else:
                PIL_img = get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)
            
            # 保存生成的字体图片
            image_name = chinese_char + ".jpg"
            save_path = os.path.join(font_img_dir, image_name)
            PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count*100/total_num))
                sys.stdout.flush()
    return


def generate_chinese_images(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]
    
    print("Begin to generate images ...")
    chinese_char_num = len(CHAR2ID_DICT)
    total_num = len(font_file_list) * chinese_char_num
    count = 0
    for font_file in font_file_list:  # 外层循环是字体
        font_name = os.path.basename(font_file)
        font_type = font_name.split(".")[0]

        # 创建保存该字体图片的目录
        save_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(save_dir)

        for chinese_char, bigger_PIL_img in generate_all_chinese_images_bigger(font_file, image_size=int(obj_size*1.2)):  # 内层循环是字
            # 检查生成的灰度图像是否可用，黑底白字
            image_data = list(bigger_PIL_img.getdata())
            if sum(image_data) < 10:
               continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)
                 for i in range(num_imgs_per_font)]
            
            # 保存生成的字体图片
            for index, PIL_img in enumerate(PIL_img_list):
                image_name = chinese_char + "_" + str(index) + ".jpg"
                save_path = os.path.join(save_dir, image_name)
                PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count*100/total_num))
                sys.stdout.flush()


def generate_tfrecords(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get font_file_list ...")
    font_file_list = [os.path.join(FONT_FILE_DIR, font_name) for font_name in os.listdir(FONT_FILE_DIR)
                      if font_name.lower()[-4:] in (".otf", ".ttf", ".ttc", ".fon")]
    
    # 创建保存tfrecords文件的目录
    check_or_makedirs(CHAR_TFRECORDS_DIR)

    # 可以把生成的图片直接存入tfrecords文件
    # 不必将生成的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    # 通常是用某种字体对一个字生成很多个增强的图片，这些图片最好是分开存放
    # 若直接把同一字体同一个字的多张图片连续放到同一个tfrecords里，那么训练batch的多样性不好
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(CHAR_TFRECORDS_DIR, "chinese_imgs_%d_from_font.tfrecords" % i))
         for i in range(100)]
    
    print("Begin to generate images ...")
    chinese_char_num = len(CHAR2ID_DICT)
    total_num = len(font_file_list) * chinese_char_num
    count = 0
    for font_file in font_file_list:  # 外层循环是字体
        
        for chinese_char, bigger_PIL_img in generate_all_chinese_images_bigger(font_file, image_size=int(obj_size * 1.2)):  # 内层循环是字
            # 检查生成的灰度图像是否可用，黑底白字
            image_data = list(bigger_PIL_img.getdata())
            if sum(image_data) < 10:
                continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)
                 for i in range(num_imgs_per_font)]

            # 保存生成的字体图片
            for PIL_img in PIL_img_list:
                writer = random.choice(writers_list)

                bytes_image = PIL_img.tobytes()  # 将图片转化为原生bytes
                bytes_char = chinese_char.encode('utf-8')
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                            'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_img.height])),
                            'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[PIL_img.width])),
                            'bytes_char': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_char]))
                        }))
                writer.write(example.SerializeToString())

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count*100/total_num))
                sys.stdout.flush()

    # 关闭所有的tfrecords写者
    [writer.close() for writer in writers_list]


""" ************************* 已有图片转化得到训练图片 ************************ """


def get_external_image_paths(root_dir):
    
    for root, dirs, files_list in os.walk(root_dir):
        if len(files_list)>0:
            image_paths_list = []
            for file_name in files_list:
                if file_name.endswith(".gif") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = os.path.join(root, file_name)
                    image_paths_list.append(image_path)
            if len(image_paths_list)>0:
                font_type = os.path.basename(root)
                random.shuffle(image_paths_list)
                yield font_type, image_paths_list


# 生成样例图片，以便检查图片的有效性
def convert_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=True):
    print("Get total images num ...")
    font_images_num_list = [len(os.listdir(os.path.join(EXTERNEL_IMAGES_DIR, content)))
                            for content in os.listdir(EXTERNEL_IMAGES_DIR)
                            if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]

    print("Begin to convert images ...")
    total_num = sum(font_images_num_list)
    count = 0
    for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):
        # 创建保存该字体图片的目录
        font_img_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(font_img_dir)
        
        for image_path in image_paths_list:
            # 加载外部图片，将图片调整为正方形
            # 为了保证图片旋转时不丢失信息，生成的图片比本来的图片稍微bigger
            # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                print("The image %s result in OSError !"%image_path )
                continue

            if not augmentation:
                PIL_img = get_standard_image(bigger_PIL_img, obj_size, reverse_color=True)
            else:
                PIL_img = get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)

            # 保存生成的字体图片
            image_name = os.path.basename(image_path).split(".")[0] + ".jpg"
            save_path = os.path.join(font_img_dir, image_name)
            PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count*100/total_num))
                sys.stdout.flush()


def convert_chinese_images(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get total images num ...")
    font_images_num_list = [len(os.listdir(os.path.join(EXTERNEL_IMAGES_DIR, content)))
                            for content in os.listdir(EXTERNEL_IMAGES_DIR)
                            if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]

    print("Begin to convert images ...")
    total_num = sum(font_images_num_list)
    count = 0
    for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):

        # 创建保存该字体图片的目录
        save_dir = os.path.join(CHAR_IMGS_DIR, font_type)
        remove_then_makedirs(save_dir)
        
        for image_path in image_paths_list:
            # 加载外部图片，将图片调整为正方形
            # 为了保证图片旋转时不丢失信息，生成的图片比本来的图片稍微bigger
            # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                print("The image %s result in OSError !"%image_path )
                continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)
                 for i in range(num_imgs_per_font)]
            
            # 保存生成的字体图片
            for index, PIL_img in enumerate(PIL_img_list):
                image_name = os.path.basename(image_path).split(".")[0] + "_" + str(index) + ".jpg"
                save_path = os.path.join(save_dir, image_name)
                PIL_img.save(save_path, format="jpeg")

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count * 100 / total_num))
                sys.stdout.flush()


def convert_tfrecords(obj_size=CHAR_IMG_SIZE, num_imgs_per_font=NUM_IMAGES_PER_FONT):
    print("Get total images num ...")
    font_images_num_list = [len(os.listdir(os.path.join(EXTERNEL_IMAGES_DIR, content)))
                            for content in os.listdir(EXTERNEL_IMAGES_DIR)
                            if os.path.isdir(os.path.join(EXTERNEL_IMAGES_DIR, content))]
    
    # 创建保存tfrecords文件的目录
    check_or_makedirs(CHAR_TFRECORDS_DIR)
    
    # 可以把变换的图片直接存入tfrecords文件
    # 不必将变换的图片先保存到磁盘，再从磁盘读取出来保存到tfrecords文件，这样效率太低
    # 通常是用一种字体的一个字图片增强出很多个图片，这些图片最好是分开存放
    # 若直接把同一字体同一个字图片增强出的多张图片连续放到同一个tfrecords里，那么每一个训练batch的多样性就不好
    writers_list = \
        [tf.io.TFRecordWriter(os.path.join(CHAR_TFRECORDS_DIR, "chinese_imgs_%d_from_img.tfrecords" % i))
         for i in range(100)]

    print("Begin to convert images ...")
    total_num = sum(font_images_num_list)
    count = 0
    for font_type, image_paths_list in get_external_image_paths(root_dir=EXTERNEL_IMAGES_DIR):

        for image_path in image_paths_list:
            chinese_char = os.path.basename(image_path)[0]

            # 加载外部图片，将图片调整为正方形
            # 为了保证图片旋转时不丢失信息，生成的图片比本来的图片稍微bigger
            # 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
            try:
                bigger_PIL_img = load_external_image_bigger(image_path, white_background=True, reverse_color=True)
            except OSError:
                print("The image %s result in OSError !" % image_path)
                continue

            PIL_img_list = \
                [get_augmented_image(bigger_PIL_img, obj_size, rotation=True, dilate=False, erode=True, reverse_color=True)
                 for i in range(num_imgs_per_font)]

            # 保存生成的字体图片
            for index, PIL_img in enumerate(PIL_img_list):
                # train_set和test_set的比例约为 5:1
                writer = random.choice(writers_list)

                bytes_image = PIL_img.tobytes()  # 将图片转化为原生bytes
                bytes_char = chinese_char.encode('utf-8')
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'bytes_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image])),
                            'bytes_char': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_char])),
                        }))
                writer.write(example.SerializeToString())

            # 当前进度
            count += 1
            if count % 200 == 0:
                print("Progress bar: %.2f%%" % (count*100/total_num))
                sys.stdout.flush()

    # 关闭所有的 tfrecords writer
    [writer.close() for writer in writers_list]


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
                'bytes_char': tf.io.FixedLenFeature([], tf.string)
            })

    data_set = data_set.map(parse_func)
    
    for features in data_set.take(1):
        bytes_char = features['bytes_char']
        bytes_image = features['bytes_image']                               # 第一种处理方法
        raw_image = tf.io.decode_raw(bytes_image, tf.uint8)                 # 第二种处理方法
        image = tf.reshape(raw_image, shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE)) # 第三种处理方法
        
        # *************************
        print(bytes_char)
        chinese_char = bytes_char.numpy().decode("utf-8")
        label_id = CHAR2ID_DICT(chinese_char)

        print(bytes_image)  # 第一种处理方法
        # PIL_img = Image.frombytes(mode="L", size=(CHAR_IMG_SIZE, CHAR_IMG_SIZE), data=bytes_image.numpy())

        print(raw_image)    # 第二种处理方法
        # np_img = raw_image.numpy().reshape((CHAR_IMG_SIZE, CHAR_IMG_SIZE))
        # PIL_img = Image.fromarray(np_img)
        
        print(image)        # 第三种处理方法
        np_img = image.numpy()
        PIL_img = Image.fromarray(np_img)
        PIL_img.show()


if __name__ == '__main__':
    generate_chinese_images_to_check(obj_size=200, augmentation=False)
    # generate_chinese_images(num_imgs_per_font=3)
    # generate_tfrecords(num_imgs_per_font=NUM_IMAGES_PER_FONT)
    #
    # convert_chinese_images_to_check(obj_size=CHAR_IMG_SIZE, augmentation=True)
    # convert_chinese_images(num_imgs_per_font=3)
    # convert_tfrecords(num_imgs_per_font=10)
    #
    # display_tfrecords(tfrecords_file=os.path.join(CHAR_TFRECORDS_DIR, "chinese_imgs_0_from_font.tfrecords"))
    # display_tfrecords(tfrecords_file=os.path.join(CHAR_TFRECORDS_DIR, "chinese_imgs_0_from_img.tfrecords"))

    print("Done!")
    