# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import functools
import threading
import numpy as np
import tensorflow as tf
from PIL import Image

from config import CHAR_RECOG_BATCH_SIZE, CHAR_IMG_SIZE
from util import CHAR2ID_DICT_TASK2 as CHAR2ID_DICT
from util import NUM_COMPO, ID2COMPO_INDICES


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


def adjust_img_to_fixed_shape(PIL_img=None, np_img=None, img_shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE)):
    if PIL_img is None:
        PIL_img = Image.fromarray(np_img)
    
    # to rgb
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert("RGB")
    
    # resize
    raw_w, raw_h = PIL_img.size
    obj_h, obj_w = img_shape
    if raw_w != obj_w or raw_h != obj_h:
        scale_ratio = min(obj_h / raw_h, obj_w / raw_w)
        new_w, new_h = int(raw_w * scale_ratio), int(raw_h * scale_ratio)
        PIL_img = PIL_img.resize(size=(new_w, new_h))
        x1, y1 = (obj_w - new_w) // 2, (obj_h - raw_h) // 2
        PIL_img = Image.new("RGB", size=(obj_w, obj_h), color="white").paste(PIL_img, box=(x1, y1))
    
    return PIL_img


def get_img_then_augment(annotation_line, img_shape):
    img_path = annotation_line.split("\t")[0]
    chinese_char = os.path.basename(img_path)[0]
    
    PIL_img = Image.open(img_path)
    PIL_img = adjust_img_to_fixed_shape(PIL_img, img_shape)
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    np_img = imgs_augmentation(np_img=np_img)  # image augmentation
    
    return np_img, chinese_char


@threadsafe_generator
def data_generator_with_images(annotation_lines, batch_size, img_shape):
    n = len(annotation_lines)
    i = 0
    while True:
        images_list = []
        char_ids_list = []
        compo_embeddings = np.zeros(shape=(batch_size, NUM_COMPO), dtype=np.int8)
        for batch_id in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            i = (i + 1) % n
            
            np_img, chinese_char = get_img_then_augment(annotation_lines[i], img_shape)
            char_id = CHAR2ID_DICT(chinese_char)
            
            images_list.append(np_img)
            char_ids_list.append(char_id)
            
            compo_indices = ID2COMPO_INDICES[char_id]
            batch_ids = [batch_id,] * len(compo_indices)
            compo_embeddings[batch_ids, compo_indices] = 1
        
        batch_images = np.stack(images_list, axis=0).astype(np.float32)
        chinese_char_ids = np.array(char_ids_list, dtype=np.int32)
        
        inputs_dict = {"batch_images": batch_images,
                       "chinese_char_ids": chinese_char_ids,
                       "compo_embeddings": compo_embeddings}
        
        yield inputs_dict
        

def look_up_dict_py(char_utf8_np):
    char_utf8 = char_utf8_np.tolist()   # scalar
    chinese_char = char_utf8.decode("utf-8")
    
    char_id = CHAR2ID_DICT[chinese_char]
    compo_indices = ID2COMPO_INDICES[char_id]
    
    char_id = np.array(char_id, dtype=np.int32)
    compo_embedding = np.zeros(shape=[NUM_COMPO,], dtype=np.int8)
    compo_embedding[compo_indices] = 1
    
    return char_id, compo_embedding
    

def data_generator_with_tfrecords(tfrecords_files, batch_size, img_shape):
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

        # to rgb
        img_tensor = tf.image.grayscale_to_rgb(img_tensor)

        # scale image to fixed size
        raw_shape = tf.cast([img_h, img_w], tf.float32)
        obj_shape = tf.cast(img_shape, tf.float32)
        scale_ratio = tf.reduce_min(obj_shape / raw_shape)
        new_shape = tf.cast(obj_shape * scale_ratio, tf.int32)
        img_tensor = tf.image.resize(img_tensor, size=new_shape)  # float32
        img_tensor = tf.cast(img_tensor, dtype=tf.uint8)
        
        delta_h, delta_w = img_shape[0] - new_shape[0], img_shape[1] - new_shape[1]
        h1, w1 = delta_h // 2, delta_w // 2
        h2, w2 = delta_h - h1, delta_w - w1
        img_tensor = tf.pad(img_tensor, [[h1,h2], [w1,w2], [0,0]], mode="CONSTANT", constant_values=255)
        img_tensor = imgs_augmentation_tf(img_tensor)
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # chinese char id
        char_utf8_tf = features['bytes_char']
        chinese_char_id, compo_embedding = tf.numpy_function(look_up_dict_py, inp=char_utf8_tf, dout=[tf.int32, tf.int8])
        
        return {"batch_images": img_tensor, "chinese_char_ids": chinese_char_id, "compo_embeddings": compo_embedding}
    
    data_set = data_set.map(parse_fn, tf.data.experimental.AUTOTUNE).shuffle(2048).batch(batch_size).prefetch(200)
    
    return data_set


def data_generator(data_file, src_type="images", validation_split=0.1):
    """data generator for fit_generator"""
    img_shape, batch_size = (CHAR_IMG_SIZE, CHAR_IMG_SIZE), CHAR_RECOG_BATCH_SIZE
    
    with open(data_file, "r", encoding="utf8") as fr:
        lines = [line.strip() for line in fr.readlines()]
    num_train = int(len(lines) * (1 - validation_split))
    train_lines = lines[:num_train]
    validation_lines = lines[num_train:]
    np.random.shuffle(train_lines)
    
    if src_type == "images":
        training_generator = data_generator_with_images(train_lines, batch_size, img_shape)
        validation_generator = data_generator_with_images(validation_lines, batch_size, img_shape)
    elif src_type == "tfrecords":
        training_generator = data_generator_with_tfrecords(train_lines, batch_size, img_shape)
        validation_generator = data_generator_with_tfrecords(validation_lines, batch_size, img_shape)
    else:
        ValueError("Optional src type: 'images', 'tfrecords'.")
    
    return training_generator, validation_generator


if __name__ == '__main__':
    print("Done !")
