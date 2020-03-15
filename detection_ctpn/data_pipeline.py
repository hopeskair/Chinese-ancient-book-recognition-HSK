# -*- encoding: utf-8 -*-
# Author: hushukai

from PIL import Image
import numpy as np
import tensorflow as tf

from .utils.gt_utils import gen_gt_from_quadrilaterals
from config import CTPN_NET_STRIDE


def get_image_and_boxes(annotation_line):
    line = annotation_line.strip().split()
    PIL_img = Image.open(line[0])
    
    if PIL_img.mode != "L":
        PIL_img = PIL_img.convert("L")
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    boxes = [list(map(int, box.split(','))) for box in line[1:]]
    boxes = np.array(boxes, dtype=np.float32)
    
    return np_img, boxes


def change_text_vertical_to_horisontal(np_img, boxes=None):
    h, w = np_img.shape[:2]
    np_img = np_img.transpose((1,0))[::-1, ...]
    
    if boxes:
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1_tr, y1_tr, x2_tr, y2_tr = y1, x1, y2, x2
        x1_flip, y1_flip, x2_flip, y2_flip = x1_tr, w-1-y1_tr, x2_tr, w-1-y2_tr # upside-down
        x1_result, y1_result, x2_result, y2_result = x1_flip, y2_flip, x2_flip, y1_flip
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1_result, y1_result, x2_result, y2_result
    
    return np_img, boxes


def restore_text_horizontal_to_vertical(horizontal_text_img):
    shape = horizontal_text_img.shape[:2]
    h, w = shape[:2]
    if len(shape) == 2:
        np_img = horizontal_text_img.transpose((1, 0))[::-1, ...]
    elif len(shape):
        np_img = horizontal_text_img.transpose((1, 0, 2))[::-1, ...]
    return np_img


def restore_boxes_horizontal_to_vertical(boxes, raw_img_shape):
    h, w = raw_img_shape
    
    x1_result, y1_result, x2_result, y2_result = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1_flip, y1_flip, x2_flip, y2_flip = x1_result, y2_result, x2_result, y1_result
    x1_tr, y1_tr, x2_tr, y2_tr = x1_flip, w-1-y1_flip, x2_flip, w-1-y2_flip
    x1, y1, x2, y2 = y1_tr, x1_tr, y2_tr, x2_tr
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1, y1, x2, y2
    
    return boxes


def pack_book_pages(imgs_list, boxes_list, background="white"):
    assert len(imgs_list) == len(boxes_list)
    batch_size = len(imgs_list)
    
    num_boxes = max([len(boxes) for boxes in boxes_list])
    batch_boxes = np.zeros(shape=(batch_size, num_boxes, 6), dtype=np.float32)
    # x1, y1, x2, y2, class_id, padding_flag
    
    img_shape = [np_img.shape[:2] for np_img in imgs_list]
    max_h = max([h for (h, w) in img_shape])
    max_w = max([w for (h, w) in img_shape])
    max_h += -max_h % 16
    max_w += -max_w % 16
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
        batch_boxes[i, :num, :5] = boxes_list[i]    # x1, y1, x2, y2, class_id
        batch_boxes[i, :num, 5] = 1                 # padding_flag
    batch_imgs = np.expand_dims(batch_imgs, axis=-1)
    
    return batch_imgs, batch_boxes


def split_boxes_to_fixed_width(boxes, class_id, im_shape):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    gt_quadrilaterals = np.stack([x1,y1,x2,y1,x2,y2,x1,y2], axis=0)
    gt_boxes, cls_ids = gen_gt_from_quadrilaterals(gt_quadrilaterals,
                                                   gt_class_ids=np.array([class_id]*boxes.shape[0]),
                                                   image_shape=im_shape,
                                                   width_stride=CTPN_NET_STRIDE)
    gt_boxes = np.concatenate([gt_boxes, class_id[:,np.newaxis]], axis=1)
    return gt_boxes


def data_generator_with_images(annotation_lines, batch_size, text_type="horizontal"):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    
    i = 0
    while True:
        imgs_list = []
        boxes_list = []
        for _ in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            np_img, boxes = get_image_and_boxes(annotation_lines[i])
            if text_type.lower() in ("v", "vertical"):
                np_img, boxes = change_text_vertical_to_horisontal(np_img, boxes)
            boxes = split_boxes_to_fixed_width(boxes, class_id=1.0, im_shape=np_img.shape[:2])
            imgs_list.append(np_img)
            boxes_list.append(boxes)
            i = (i + 1) % n
        
        batch_imgs, batch_boxes = pack_book_pages(imgs_list, boxes_list, background="white")
        inputs_dict = {"batch_images": batch_imgs, "batch_boxes": batch_boxes}
        
        yield inputs_dict, None


def data_generator_with_tfrecords(tfrecords_files, batch_size, text_type="horizontal"):
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
            np_img = image.numpy()
            
            text_boxes = tf.io.decode_raw(features["text_boxes"], tf.int32)
            text_boxes = tf.reshape(text_boxes, shape=(-1, 5))
            boxes = text_boxes.numpy()
            boxes = split_boxes_to_fixed_width(boxes, class_id=1, im_shape=np_img.shape[:2])
            
            if text_type.lower() in ("v", "vertical"):
                np_img, boxes = change_text_vertical_to_horisontal(np_img, boxes)
            imgs_list.append(np_img)
            boxes_list.append(boxes)
        
        batch_imgs, batch_boxes = pack_book_pages(imgs_list, boxes_list, background="white")
        inputs_dict = {"batch_images": batch_imgs, "batch_boxes": batch_boxes}
        
        yield inputs_dict, None


def data_generator(data_file, batch_size, src_type="images", text_type="horizontal", validation_split=0.1):
    """data generator for fit_generator"""
    with open(data_file, "r", encoding="utf8") as fr:
        lines = fr.readlines()
    np.random.shuffle(lines)
    num_train = int(len(lines)*(1-validation_split))
    
    if src_type == "images":
        training_generator = data_generator_with_images(lines[:num_train], batch_size, text_type)
        validation_generator = data_generator_with_images(lines[num_train:], batch_size, text_type)
    elif src_type == "tfrecords":
        training_generator = data_generator_with_tfrecords(lines[:num_train], batch_size, text_type)
        validation_generator = data_generator_with_tfrecords(lines[num_train:], batch_size, text_type)
    else:
        ValueError("Optional src type: 'images', 'tfrecords'.")
        
    return training_generator, validation_generator