# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json
from PIL import Image
from skimage import color, transform
import numpy as np
import tensorflow as tf

from segment_base.utils import threadsafe_generator
from segment_base.utils import get_segment_task_params

from config import SEGMENT_MAX_INCLINATION
from config import SEGMENT_TASK_ID, SEGMENT_ID_TO_TASK


def image_preprocess_tf(images, stage="train"):
    convert_imgs = tf.image.per_image_standardization(images)
    if stage == "train":
        convert_imgs = tf.cond(tf.random.uniform([]) < 0.5, lambda: convert_imgs, lambda: 1. - convert_imgs)
    return convert_imgs


def gauss_noise(np_img):
    np_img = np_img.astype(np.float32)  # 此步是为了避免像素点小于0，大于255的情况
    
    noise = np.random.normal(0, 30, size=np_img.shape)
    np_img = np_img + noise
    np_img[np_img < 0] = 0
    np_img[np_img > 255] = 255
    np_img = np_img.astype(np.uint8)
    
    return np_img


def salt_and_pepper_noise(np_img, proportion=0.05):
    h, w = np_img.shape[:2]
    num = int(h * w * proportion) // 2  # 添加椒盐噪声的个数
    
    # 椒噪声
    hs = np.random.randint(0, h - 1, size=[num], dtype=np.int64)
    ws = np.random.randint(0, w - 1, size=[num], dtype=np.int64)
    np_img[hs, ws] = 0
    
    # 盐噪声
    hs = np.random.randint(0, h - 1, size=[num], dtype=np.int64)
    ws = np.random.randint(0, w - 1, size=[num], dtype=np.int64)
    np_img[hs, ws] = 255
    
    return np_img


def gauss_noise_tf(img_tensor):
    img_tensor = tf.cast(img_tensor, dtype=tf.float32)  # 此步是为了避免像素点小于0，大于255的情况
    
    noise = tf.random.normal(tf.shape(img_tensor), mean=0.0, stddev=30.0)
    img_tensor = img_tensor + noise
    img_tensor = tf.where(img_tensor < 0., 0., img_tensor)
    img_tensor = tf.where(img_tensor > 255., 255., img_tensor)
    img_tensor = tf.cast(img_tensor, dtype=tf.uint8)
    
    return img_tensor


def salt_and_pepper_noise_tf(img_tensor, proportion=0.05):
    img_shape = tf.shape(img_tensor, out_type=tf.int64)
    float_shape = tf.cast(img_shape, tf.float32)
    h, w = img_shape[0], img_shape[1]
    fh, fw = float_shape[0], float_shape[1]
    num = tf.cast(fh * fw * proportion, tf.int32) // 2  # 添加椒盐噪声的个数
    
    # 椒噪声
    hs = tf.random.uniform([num], 0, h - 1, dtype=tf.int64)
    ws = tf.random.uniform([num], 0, w - 1, dtype=tf.int64)
    hs_ws = tf.stack([hs, ws], axis=1)
    noise = tf.zeros(shape=[num, 3], dtype=img_tensor.dtype)
    img_tensor = tf.tensor_scatter_nd_add(img_tensor, indices=hs_ws, updates=noise)
    
    # 盐噪声
    hs = tf.random.uniform([num], 0, h - 1, dtype=tf.int64)
    ws = tf.random.uniform([num], 0, w - 1, dtype=tf.int64)
    hs_ws = tf.stack([hs, ws], axis=1)
    noise = tf.zeros(shape=[num, 3], dtype=img_tensor.dtype) + 255
    img_tensor = tf.tensor_scatter_nd_add(img_tensor, indices=hs_ws, updates=noise)
    
    return img_tensor


def imgs_augmentation(np_img):
    np_img = gauss_noise(np_img)  # 高斯噪声
    np_img = salt_and_pepper_noise(np_img)  # 椒盐噪声
    return np_img


def imgs_augmentation_tf(img_tensors):
    img_tensors = gauss_noise_tf(img_tensors)  # 高斯噪声
    img_tensors = salt_and_pepper_noise_tf(img_tensors)  # 椒盐噪声
    
    # augment images, 这些方法对uint8, float32类型均有效
    img_tensors = tf.image.random_brightness(img_tensors, max_delta=0.5)
    img_tensors = tf.image.random_contrast(img_tensors, lower=0.3, upper=1.)
    img_tensors = tf.image.random_hue(img_tensors, max_delta=0.5)
    img_tensors = tf.image.random_saturation(img_tensors, lower=0., upper=2)
    return img_tensors


def rotate_90_degrees(np_img=None, split_positions=None):
    if np_img is not None:
        axes = (1, 0) if len(np_img.shape) == 2 else (1, 0, 2)
        np_img = np_img.transpose(axes)[::-1, ...]
    
    if split_positions is not None:
        # assert split_positions.shape[-1] == 2
        split_positions = split_positions[:, ::-1]
    
    return np_img, split_positions


def rotate_90_degrees_tf(img_tensor=None, split_positions=None):
    if img_tensor is not None:
        perm = (1, 0) if len(img_tensor.shape) == 2 else (1, 0, 2)
        img_tensor = tf.transpose(img_tensor, perm=perm)[::-1]
    
    if split_positions is not None:
        split_positions = split_positions[:, ::-1]
    
    return img_tensor, split_positions


def restore_original_angle(np_img=None, pred_split_positions=None):
    if np_img is not None:
        axes = (1, 0) if len(np_img.shape) == 2 else (1, 0, 2)
        np_img = np_img[::-1].transpose(axes)
    
    if pred_split_positions is not None:
        # assert pred_split_positions.shape[-1] == 2
        pred_split_positions = pred_split_positions[:, ::-1]
    
    return np_img, pred_split_positions


def restore_original_angle_tf(img_tensor=None, pred_split_positions=None):
    if img_tensor is not None:
        perm = (1, 0) if len(img_tensor.shape) == 2 else (1, 0, 2)
        img_tensor = tf.transpose(img_tensor[::-1], perm=perm)
    
    if pred_split_positions is not None:
        pred_split_positions = pred_split_positions[:, ::-1]
    
    return img_tensor, pred_split_positions


def tilt_text_img_py(np_img, split_positions, segment_task):
    raw_shape = np_img.shape
    raw_h, raw_w = raw_shape[:2]

    segment_task = SEGMENT_ID_TO_TASK[int(segment_task)] if not isinstance(segment_task, str) else segment_task
    incline = np.random.randint(0, SEGMENT_MAX_INCLINATION[segment_task])  # inclination 倾斜度
    
    pad_width = ((0, 0), (incline, incline)) if len(raw_shape) == 2 else ((0, 0), (incline, incline), (0, 0))
    np_img = np.pad(np_img, pad_width, mode="constant", constant_values=255)  # 白底黑字
    
    # quad_coordinates: 8-tuple (x0, y0, x1, y1, x2, y2, y3, y3), they are
    # upper left, lower left, lower right, and upper right corner of the source quadrilateral.
    quad_coordinates_rightward = (0, 0, incline, raw_h - 1, 2 * incline + raw_w - 1, raw_h - 1, incline + raw_w - 1, 0)
    quad_coordinates_leftward = (incline, 0, 0, raw_h - 1, incline + raw_w - 1, raw_h - 1, 2 * incline + raw_w - 1, 0)
    
    if np.random.rand() < 0.5:
        quad_coordinates = quad_coordinates_rightward
        split_positions = split_positions + np.array([incline, 0], dtype=np.float32)
    else:
        quad_coordinates = quad_coordinates_leftward
        split_positions = split_positions + np.array([0, incline], dtype=np.float32)
    
    PIL_size = (raw_w + incline, raw_h)
    PIL_img = Image.fromarray(np_img)
    PIL_img = PIL_img.transform(PIL_size, method=Image.QUAD, data=quad_coordinates)
    np_img = np.array(PIL_img)
    
    return [np_img, split_positions]


def adjust_img_to_fixed_shape(np_img, split_positions=None, fixed_shape=(560, None), feat_stride=16, segment_task="book_page", text_type="horizontal"):
    # rotate 90 degrees rightward.
    text_type = text_type[0].lower()
    if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
        np_img, split_positions = rotate_90_degrees(np_img, split_positions)
    
    # to rgb
    if len(np_img.shape) == 2 or np_img.shape[-1] != 3:
        np_img = color.grey2rgb(np_img)
    
    # scale image to fixed shape
    raw_h, raw_w = np_img.shape[:2]
    fixed_h, fixed_w = fixed_shape
    if segment_task in ("book_page", "mix_line", "text_line"):
        scale_ratio = fixed_h / raw_h       # fixed_h是16的倍数, fixed_w为None
        fixed_w = int(raw_w * scale_ratio)  # 在打包batch时，将调整为16的倍数
    else:
        scale_ratio = fixed_w / raw_w       # double_line情况, fixed_h为None, fixed_w是16的倍数
        fixed_h = int(raw_h * scale_ratio)
        fixed_h += -fixed_h % feat_stride   # 调整为16的倍数
    np_img = transform.resize(np_img, output_shape=(fixed_h, fixed_w))  # float32
    np_img = np_img.astype(np.uint8)
    
    if split_positions is not None:
        split_positions = split_positions * scale_ratio
        np_img, split_positions = tilt_text_img_py(np_img, split_positions, segment_task)   # 加入文本倾斜
    
    return np_img, split_positions, scale_ratio


def adjust_img_to_fixed_shape_tf(img_tensor, split_positions, fixed_shape=(560, None), feat_stride=16, segment_task="book_page", text_type="horizontal"):
    # rotate 90 degrees rightward.
    text_type = text_type[0].lower()
    if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
        img_tensor, split_positions = rotate_90_degrees_tf(img_tensor, split_positions)
    
    # to rgb
    channels = tf.shape(img_tensor)[-1]
    img_tensor = tf.cond(channels == 3, lambda: img_tensor, lambda: tf.image.grayscale_to_rgb(img_tensor))
    
    # scale image to fixed size
    raw_shape = tf.cast(tf.shape(img_tensor)[:2], tf.float32)
    fixed_h, fixed_w = fixed_shape
    if segment_task in ("book_page", "mix_line", "text_line"):
        scale_ratio = fixed_h / raw_shape[0]                    # fixed_h是16的倍数, fixed_w为None
        fixed_w = tf.cast(raw_shape[1] * scale_ratio, tf.int32) # 在打包batch后，将调整为16的倍数
    else:
        scale_ratio = fixed_w / raw_shape[1]                    # double_line情况, fixed_h为None, fixed_w是16的倍数
        fixed_h = tf.cast(raw_shape[0] * scale_ratio, tf.int32)
        fixed_h += -fixed_h % feat_stride                       # 调整为16的倍数
    img_tensor = tf.image.resize(img_tensor, size=[fixed_h, fixed_w])  # float32
    img_tensor = tf.cast(img_tensor, dtype=tf.uint8)
    split_positions = split_positions * scale_ratio
    
    # 加入文本倾斜
    segment_task = tf.constant(SEGMENT_TASK_ID[segment_task], tf.int32)
    img_tensor, split_positions = tf.numpy_function(tilt_text_img_py,
                                                    inp=[img_tensor, split_positions, segment_task],
                                                    Tout=[tf.uint8, tf.float32])
    
    return img_tensor, split_positions


def get_image_and_split_pos(annotation_line, segment_task="book_page"):
    line = annotation_line.split("\t")
    PIL_img = Image.open(line[0])
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    split_pos = json.loads(line[1])["split_pos_list"]
    split_pos = np.array(split_pos, dtype=np.float32)
    split_pos.sort()    # 排序
    if segment_task in ("double_line",):
        split_pos = split_pos[1: -1] # 首尾的切分线不需要学习
    split_pos = np.tile(split_pos[:, np.newaxis], reps=(1, 2))
    
    return np_img, split_pos


def pack_a_batch(imgs_list, split_pos_list, feat_stride=16, background="white"):
    # assert len(imgs_list) == len(split_pos_list)
    batch_size = len(imgs_list)
    
    num_split = max([len(split_pos) for split_pos in split_pos_list])
    split_positions = -1 * np.ones(shape=(batch_size, num_split, 2), dtype=np.float32)  # padding -1
    
    fixed_h = imgs_list[0].shape[0]
    imgs_w = [np_img.shape[1] for np_img in imgs_list]
    max_w = max([w for w in imgs_w])
    max_w += -max_w % feat_stride
    batch_imgs = np.empty(shape=(batch_size, fixed_h, max_w, 3), dtype=np.float32)
    real_images_width = np.array(imgs_w, dtype=np.int32)
    if background == "white":
        batch_imgs.fill(255)
    elif background == "black":
        batch_imgs.fill(0)
    else:
        ValueError("Optional image background: 'white', 'black'.")
    
    for i, np_img in enumerate(imgs_list):
        img_h, img_w = np_img.shape[:2]
        batch_imgs[i, :img_h, :img_w] = np_img
        batch_imgs[i] = imgs_augmentation(np_img=batch_imgs[i]) # image augmentation
        num = len(split_pos_list[i])
        split_positions[i, :num, :] = split_pos_list[i]
    
    return batch_imgs, real_images_width, split_positions


@threadsafe_generator
def data_generator_with_images(annotation_lines,
                               batch_size,
                               fixed_shape,
                               feat_stride=16,
                               segment_task="book_page",
                               text_type="horizontal"):
    
    n = len(annotation_lines)
    i = 0
    while True:
        images_list = []
        split_pos_list = []
        for _ in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            np_img, split_positions = get_image_and_split_pos(annotation_lines[i], segment_task)
            np_img, split_positions, _ = adjust_img_to_fixed_shape(np_img, split_positions, fixed_shape, feat_stride, segment_task, text_type)
            images_list.append(np_img)
            split_pos_list.append(split_positions)
            i = (i + 1) % n

        batch_imgs, real_images_width, split_positions = pack_a_batch(images_list, split_pos_list, feat_stride, background="white")
        inputs_dict = {"batch_images": batch_imgs, "real_images_width": real_images_width, "split_lines_pos": split_positions}
        
        yield inputs_dict


def parse_fn(serialized_example, segment_task="book_page"):
    if segment_task == "book_page":
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'text_boxes': tf.io.FixedLenFeature([], tf.string),
                'split_positions': tf.io.FixedLenFeature([], tf.string)
            })
    elif segment_task == "text_line":
        features = tf.io.parse_single_example(
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
    else:
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                "split_positions": tf.io.FixedLenFeature([], tf.string)
            })
    return features


def data_generator_with_tfrecords(tfrecords_files,
                                  batch_size,
                                  fixed_shape,
                                  feat_stride=16,
                                  segment_task="book_page",
                                  text_type="horizontal"):
    
    data_set = tf.data.TFRecordDataset(tfrecords_files).repeat()
    
    def _parse_fn(serialized_example):
        features = parse_fn(serialized_example, segment_task)
        
        img_h = features['img_height']
        img_w = features['img_width']
        image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
        img_tensor = tf.reshape(image_raw, shape=[img_h, img_w, 1])
        
        split_positions = tf.io.decode_raw(features["split_positions"], tf.int32)
        split_positions = tf.reshape(split_positions, shape=(-1,))
        order_indices = tf.argsort(split_positions, axis=0)         # 排序
        split_positions = tf.gather(split_positions, order_indices) # 排序
        if segment_task in ("double_line",):
            split_positions = split_positions[1: -1]    # 首尾的切分线不需要学习
        split_positions = tf.tile(split_positions[:, tf.newaxis], multiples=[1, 2])
        split_positions = tf.cast(split_positions, tf.float32)

        img_tensor, split_positions = adjust_img_to_fixed_shape_tf(img_tensor, split_positions, fixed_shape, feat_stride, segment_task, text_type)
        img_width = tf.cast(tf.shape(img_tensor)[1], tf.float32)
        
        return img_tensor, img_width, split_positions
    
    def _image_augmentation(batch_imgs, real_images_width, split_positions):
        curr_width = tf.shape(batch_imgs)[2]
        num_padding = -curr_width % feat_stride
        batch_imgs = tf.pad(batch_imgs, [[0,0], [0,0], [0,num_padding], [0,0]], mode="CONSTANT", constant_values=255)
        batch_imgs = tf.map_fn(fn=lambda x: imgs_augmentation_tf(x), elems=batch_imgs, dtype=tf.uint8)
        batch_imgs = tf.cast(batch_imgs, tf.float32)
        inputs_dict = {"batch_images": batch_imgs, "real_images_width": real_images_width, "split_lines_pos": split_positions}
        return  inputs_dict
    
    fixed_h = fixed_shape[0]    # None if segment_task is double_line.
    padded_shapes = (tf.TensorShape([fixed_h, None, 3]), tf.TensorShape([]), tf.TensorShape([None, 2]))
    padding_values = (tf.constant(255, tf.uint8), tf.constant(-1, tf.float32), tf.constant(-1, tf.float32))
    data_set = (data_set
                .map(_parse_fn, tf.data.experimental.AUTOTUNE)
                .shuffle(200)
                .padded_batch(batch_size, padded_shapes, padding_values, drop_remainder=True)
                .map(_image_augmentation, tf.data.experimental.AUTOTUNE)
                .prefetch(200)
    )
    
    return data_set


def data_generator(data_file, src_type="images", segment_task="book_page", text_type="horizontal", validation_split=0.1):
    """data generator for fit_generator"""
    batch_size, fixed_shape, feat_stride = get_segment_task_params(segment_task)
    
    with open(data_file, "r", encoding="utf8") as fr:
        lines = [line.strip() for line in fr.readlines()]
    num_train = int(len(lines) * (1 - validation_split))
    train_lines = lines[:num_train]
    validation_lines = lines[num_train:]
    np.random.shuffle(train_lines)
    
    if src_type == "images":
        training_generator = data_generator_with_images(train_lines, batch_size, fixed_shape, feat_stride, segment_task, text_type)
        validation_generator = data_generator_with_images(validation_lines, batch_size, fixed_shape, feat_stride, segment_task, text_type)
    elif src_type == "tfrecords":
        training_generator = data_generator_with_tfrecords(train_lines, batch_size, fixed_shape, feat_stride, segment_task, text_type)
        validation_generator = data_generator_with_tfrecords(validation_lines, batch_size, fixed_shape, feat_stride, segment_task, text_type)
    else:
        ValueError("Optional src type: 'images', 'tfrecords'.")
        
    return training_generator, validation_generator
    

if __name__ == "__main__":
    from config import SEGMENT_BOOK_PAGE_TFRECORDS_H
    
    training_generator, validation_generator = \
        data_generator(data_file=SEGMENT_BOOK_PAGE_TFRECORDS_H,
                       src_type="tfrecords",
                       segment_task="book_page",
                       text_type="horizontal")
    
    for inputs_dict in training_generator:
        print(inputs_dict)
    
    print("Done !")