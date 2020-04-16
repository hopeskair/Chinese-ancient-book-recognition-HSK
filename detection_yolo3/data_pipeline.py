# -*- encoding: utf-8 -*-
# Author: hushukai

from PIL import Image
import numpy as np
import tensorflow as tf

from detection_yolo3.model import preprocess_true_boxes


def pack_book_pages(imgs_list, boxes_list, background="white"):
    assert len(imgs_list) == len(boxes_list)
    batch_size = len(imgs_list)
    
    num_boxes = max([len(boxes) for boxes in boxes_list])
    batch_boxes = np.zeros(shape=(batch_size, num_boxes, 5), dtype=np.float32)
    
    img_shape = [np_img.shape[:2] for np_img in imgs_list]
    max_h = max([h for (h, w) in img_shape])
    max_w = max([w for (h, w) in img_shape])
    max_h += -max_h % 32
    max_w += -max_w % 32
    batch_imgs = np.empty(shape=(batch_size, max_h, max_w), dtype=np.float32)
    if background == "white":
        batch_imgs.fill(255)
    elif background == "black":
        batch_imgs.fill(0)
    else:
        ValueError("Optional image background: 'white', 'black'.")
    
    for i, np_img in enumerate(imgs_list):
        img_h, img_w = np_img.shape[:2]
        batch_imgs[i, :img_h, :img_w] = np_img
        num = len(boxes_list[i])
        batch_boxes[i, :num, :] = boxes_list[i]
    batch_imgs = np.expand_dims(batch_imgs, axis=-1)
    
    return batch_imgs, batch_boxes


def get_image_and_boxes(annotation_line):
    line = annotation_line.strip().split()
    PIL_img = Image.open(line[0])
    
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert("RGB")
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    boxes = [list(map(int, box.split(','))) for box in line[1:]]
    boxes = np.array(boxes, dtype=np.float32)
    
    return np_img, boxes


def data_generator_with_images(annotation_lines, batch_size, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    
    i = 0
    while True:
        imgs_list = []
        boxes_list = []
        for _ in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            np_img, boxes = get_image_and_boxes(annotation_lines[i])
            imgs_list.append(np_img)
            boxes_list.append(boxes)
            i = (i + 1) % n
        batch_imgs, batch_boxes = pack_book_pages(imgs_list, boxes_list, background="white")
        y_true = preprocess_true_boxes(batch_boxes, batch_imgs.shape[1:3], anchors, num_classes)
        
        yield [batch_imgs] + y_true, np.zeros(batch_size)


def data_generator_with_tfrecords(tfrecords_files, batch_size, anchors, num_classes):
    data_set = tf.data.TFRecordDataset(tfrecords_files).repeat()
    
    def parse_func(serialized_example):
        return tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'text_boxes': tf.io.FixedLenFeature([], tf.string)
            })
    
    data_set = data_set.map(parse_func).shuffle(100).batch(batch_size)
    
    for batch_features in data_set:
        imgs_list = []
        boxes_list = []
        for features in batch_features:
            img_h = features['img_height']
            img_w = features['img_width']
            image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
            image = tf.reshape(image_raw, shape=[img_h, img_w])
            imgs_list.append(image.numpy())
            
            text_boxes = tf.io.decode_raw(features["text_boxes"], tf.int32)
            text_boxes = tf.reshape(text_boxes, shape=(-1, 5))
            boxes_list.append(text_boxes.numpy())
        
        batch_imgs, batch_boxes = pack_book_pages(imgs_list, boxes_list, background="white")
        y_true = preprocess_true_boxes(batch_boxes, batch_imgs.shape[1:3], anchors, num_classes)
        
        yield [batch_imgs] + y_true, np.zeros(batch_size)


def data_generator(data_file, batch_size, anchors, num_classes, src_type="images", validation_split=0.1):
    """data generator for fit_generator"""
    with open(data_file, "r", encoding="utf8") as fr:
        lines = fr.readlines()
    np.random.shuffle(lines)
    num_train = int(len(lines)*(1-validation_split))
    
    if src_type == "images":
        training_generator = data_generator_with_images(lines[:num_train], batch_size, anchors, num_classes)
        validation_generator = data_generator_with_images(lines[num_train:], batch_size, anchors, num_classes)
    elif src_type == "tfrecords":
        training_generator = data_generator_with_tfrecords(lines[:num_train], batch_size, anchors, num_classes)
        validation_generator = data_generator_with_tfrecords(lines[num_train:], batch_size, anchors, num_classes)
    else:
        ValueError("Optional src type: 'images', 'tfrecords'.")
        
    return training_generator, validation_generator