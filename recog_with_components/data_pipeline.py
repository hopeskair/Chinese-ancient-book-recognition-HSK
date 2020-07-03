# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import functools
import threading
import numpy as np
import tensorflow as tf
from PIL import Image

from config import CHAR_RECOG_BATCH_SIZE, CHAR_IMG_SIZE
from config import CHAR_RECOG_FEAT_STRIDE, COMPO_SEQ_LENGTH
from config import CHAR_STRUC_TO_ID, ID_TO_CHAR_STRUC
from util import CHAR_TO_COMPO_SEQ


class ThreadSafeGenerator:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    
    def __iter__(self):
        return self
    
    def __next__(self):  # python3
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    
    @functools.wraps(f)
    def g(*args, **kwargs):
        return ThreadSafeGenerator(f(*args, **kwargs))
    
    return g


def image_preprocess_tf(images, stage="test"):
    if stage == "train":
        images = tf.cond(tf.random.uniform([]) < 0.5, lambda: images, lambda: 255. - images)
    convert_imgs = tf.image.per_image_standardization(images)
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
    img_tensor = tf.tensor_scatter_nd_update(img_tensor, indices=hs_ws, updates=noise)
    
    # 盐噪声
    hs = tf.random.uniform([num], 0, h - 1, dtype=tf.int64)
    ws = tf.random.uniform([num], 0, w - 1, dtype=tf.int64)
    hs_ws = tf.stack([hs, ws], axis=1)
    noise = tf.zeros(shape=[num, 3], dtype=img_tensor.dtype) + 255
    img_tensor = tf.tensor_scatter_nd_update(img_tensor, indices=hs_ws, updates=noise)
    
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
    img_tensors = tf.image.random_contrast(img_tensors, lower=0.5, upper=1.)
    img_tensors = tf.image.random_hue(img_tensors, max_delta=0.5)
    img_tensors = tf.image.random_saturation(img_tensors, lower=0., upper=2.)
    return img_tensors


def adjust_img_to_fixed_shape(PIL_img=None, np_img=None, random_crop=False, fixed_shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE)):
    if PIL_img is None:
        PIL_img = Image.fromarray(np_img)
    
    # to rgb
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert("RGB")
    
    # random crop image
    if random_crop:
        raw_w, raw_h = PIL_img.size
        crop_w, crop_h = np.random.uniform(0.88, 1.0, size=[2])
        left, upper = np.random.uniform(0.0, 1 - crop_w), np.random.uniform(0.0, 1 - crop_h)
        right, lower = left + crop_w, upper + crop_h
        crop_box = np.array([left, upper, right, lower]) * np.array([raw_w, raw_h, raw_w, raw_h])
        crop_box = crop_box.astype(np.int32)
        PIL_img = PIL_img.crop(crop_box)
    
    # resize
    raw_w, raw_h = PIL_img.size
    obj_h, obj_w = fixed_shape
    if raw_w != obj_w or raw_h != obj_h:
        scale_ratio = min(obj_h / raw_h, obj_w / raw_w)
        new_w, new_h = int(raw_w * scale_ratio), int(raw_h * scale_ratio)
        PIL_img = PIL_img.resize(size=(new_w, new_h))
        x1, y1 = (obj_w - new_w) // 2, (obj_h - raw_h) // 2
        PIL_background = Image.new("RGB", size=(obj_w, obj_h), color="white")
        PIL_background.paste(PIL_img, box=(x1, y1))
        PIL_img = PIL_background
    
    return PIL_img


def get_img_then_augment(annotation_line, fixed_shape):
    img_path = annotation_line.split("\t")[0]
    chinese_char = os.path.basename(img_path)[0]
    
    PIL_img = Image.open(img_path)
    PIL_img = adjust_img_to_fixed_shape(PIL_img=PIL_img, random_crop=True, fixed_shape=fixed_shape)
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    np_img = imgs_augmentation(np_img=np_img)  # image augmentation
    
    return np_img, chinese_char


@threadsafe_generator
def data_generator_with_images(annotation_lines, batch_size, fixed_shape):
    n = len(annotation_lines)
    i = 0
    while True:
        
        batch_images = np.zeros(shape=(batch_size, *fixed_shape, 3), dtype=np.uint8) + 255  # 白底
        char_struc = np.empty(shape=(batch_size,), dtype=np.int32)
        components_seq = np.zeros(shape=(batch_size, COMPO_SEQ_LENGTH), dtype=np.int32)
        
        for batch_id in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            
            np_img, chinese_char = get_img_then_augment(annotation_lines[i], fixed_shape)
            i = (i + 1) % n

            compo_seq_str = CHAR_TO_COMPO_SEQ[chinese_char]
            struc_type, compo_seq = compo_seq_str[0], compo_seq_str[1:]
            cid_seq = [int(cid) for cid in compo_seq.split(",")]
            cid_seq_len = len(cid_seq)
            
            batch_images[batch_id] = np_img
            char_struc[batch_id] = CHAR_STRUC_TO_ID[struc_type]
            components_seq[batch_id, 0: cid_seq_len] = cid_seq
        
        batch_images = batch_images.astype(np.float32)
        inputs_dict = {"batch_images": batch_images,
                       "char_struc": char_struc,
                       "components_seq": components_seq}
        
        yield inputs_dict


def adjust_images_then_augment_tf(img_tensor, fixed_shape):
    # to rgb
    img_tensor = tf.image.grayscale_to_rgb(img_tensor)
    
    # random crop image
    raw_shape = tf.cast(tf.shape(img_tensor)[:2], tf.float32)
    raw_h, raw_w = raw_shape[0], raw_shape[1]
    crop_h = tf.cast(raw_h * tf.random.uniform([], 0.88, 1.), tf.int32)
    crop_w = tf.cast(raw_w * tf.random.uniform([], 0.88, 1.), tf.int32)
    img_tensor = tf.image.random_crop(img_tensor, size=[crop_h, crop_w, 3])
    
    # scale image to fixed size
    raw_shape = tf.cast(tf.shape(img_tensor)[:2], tf.float32)
    obj_shape = tf.cast(fixed_shape, tf.float32)
    scale_ratio = tf.reduce_min(obj_shape / raw_shape)
    new_shape = tf.cast(raw_shape * scale_ratio, tf.int32)
    img_tensor = tf.image.resize(img_tensor, size=new_shape)  # float32
    img_tensor = tf.cast(img_tensor, dtype=tf.uint8)
    
    delta_h, delta_w = fixed_shape[0] - new_shape[0], fixed_shape[1] - new_shape[1]
    h1, w1 = delta_h // 2, delta_w // 2
    h2, w2 = delta_h - h1, delta_w - w1
    img_tensor = tf.pad(img_tensor, [[h1, h2], [w1, w2], [0, 0]], mode="CONSTANT", constant_values=255)
    
    # image augmentation
    img_tensor = imgs_augmentation_tf(img_tensor)
    
    return img_tensor


def look_up_dict_py(char_utf8):
    # char_utf8 = char_utf8.tolist()   # scalar
    chinese_char = char_utf8.decode("utf-8")
    
    compo_seq_str = CHAR_TO_COMPO_SEQ[chinese_char]
    struc_type, compo_seq = compo_seq_str[0], compo_seq_str[1:]
    cid_seq = [int(cid) for cid in compo_seq.split(",")]
    cid_seq_len = len(cid_seq)
    
    struc_id = np.array(CHAR_STRUC_TO_ID[struc_type], dtype=np.int32)
    compo_seq = np.zeros(shape=[COMPO_SEQ_LENGTH,], dtype=np.int32)
    compo_seq[0:cid_seq_len] = cid_seq
    
    return struc_id, compo_seq


def data_generator_with_tfrecords(tfrecords_files, batch_size, fixed_shape):
    data_set = tf.data.TFRecordDataset(tfrecords_files).repeat()
    
    def parse_fn(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'bytes_char': tf.io.FixedLenFeature([], tf.string)
            })
        
        img_h = features['img_height']
        img_w = features['img_width']
        image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
        img_tensor = tf.reshape(image_raw, shape=[img_h, img_w, 1])
        img_tensor = adjust_images_then_augment_tf(img_tensor, fixed_shape)
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # chinese char id
        char_utf8_tf = features['bytes_char']
        char_struc_id, compo_seq = tf.numpy_function(look_up_dict_py, inp=[char_utf8_tf], Tout=[tf.int32, tf.int32])
        return img_tensor, char_struc_id, compo_seq
    
    def set_dataset_outputs(batch_images, char_struc_ids, components_seq):
        # Using set_shape() to avoid errors caused by Keras failing to infer the shape of outputs.
        batch_images.set_shape([batch_size, *fixed_shape, 3])
        char_struc_ids.set_shape([batch_size])
        components_seq.set_shape([batch_size, COMPO_SEQ_LENGTH])
        return {"batch_images": batch_images, "char_struc": char_struc_ids, "components_seq": components_seq}

    data_set = (data_set
                .map(parse_fn, tf.data.experimental.AUTOTUNE)
                .shuffle(2048)
                .batch(batch_size)
                .map(set_dataset_outputs)
                .prefetch(200))
    
    return data_set


def data_generator(data_file, src_type="images", validation_split=0.1):
    """data generator for fit_generator"""
    fixed_shape, batch_size = (CHAR_IMG_SIZE, CHAR_IMG_SIZE), CHAR_RECOG_BATCH_SIZE
    
    with open(data_file, "r", encoding="utf8") as fr:
        lines = [line.strip() for line in fr.readlines()]
    num_train = int(len(lines) * (1 - validation_split))
    train_lines = lines[:num_train]
    validation_lines = lines[num_train:]
    
    if src_type == "images":
        training_generator = data_generator_with_images(train_lines, batch_size, fixed_shape)
        validation_generator = data_generator_with_images(validation_lines, batch_size, fixed_shape)
    elif src_type == "tfrecords":
        training_generator = data_generator_with_tfrecords(train_lines, batch_size, fixed_shape)
        validation_generator = data_generator_with_tfrecords(validation_lines, batch_size, fixed_shape)
    else:
        ValueError("Optional src type: 'images', 'tfrecords'.")
    
    return training_generator, validation_generator


if __name__ == '__main__':
    print("Done !")
