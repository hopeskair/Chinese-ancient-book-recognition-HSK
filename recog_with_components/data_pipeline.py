# -*- encoding: utf-8 -*-
# Author: hushukai

import numpy as np
import tensorflow as tf


def image_preprocess_tf(images, stage="test"):
    convert_imgs = tf.image.per_image_standardization(images)
    if stage == "train":
        convert_imgs = tf.cond(tf.random.uniform([]) < 0.5, lambda: convert_imgs, lambda: -convert_imgs)
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


if __name__ == '__main__':
    print("Done !")
