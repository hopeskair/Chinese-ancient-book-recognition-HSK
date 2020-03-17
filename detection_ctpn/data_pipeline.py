# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from PIL import Image
import numpy as np
import tensorflow as tf

from detection_ctpn.utils import tf_utils
from detection_ctpn.utils import gt_utils
from config import CTPN_NET_STRIDE, BOOK_PAGE_FIXED_SIZE, BOOK_PAGE_MAX_GT_BOXES


def get_image_and_boxes(annotation_line):
    line = annotation_line.split()
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
    
    if boxes is not None:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1_tr, y1_tr, x2_tr, y2_tr = y1, x1, y2, x2
        x1_flip, y1_flip, x2_flip, y2_flip = x1_tr, w - 1 - y1_tr, x2_tr, w - 1 - y2_tr  # upside-down
        x1_result, y1_result, x2_result, y2_result = x1_flip, y2_flip, x2_flip, y1_flip
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x1_result, y1_result, x2_result, y2_result
    
    return np_img, boxes


def change_text_vertical_to_horisontal_tf(image, boxes=None):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    h, w = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
    perm = (1,0) if len(image.shape) == 2 else (1,0,2)
    image = tf.transpose(image, perm=perm)
    
    if boxes is not None:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1_tr, y1_tr, x2_tr, y2_tr = y1, x1, y2, x2
        x1_flip, y1_flip, x2_flip, y2_flip = x1_tr, w - 1 - y1_tr, x2_tr, w - 1 - y2_tr  # upside-down
        x1_result, y1_result, x2_result, y2_result = x1_flip, y2_flip, x2_flip, y1_flip
        boxes = tf.stack([x1_result, y1_result, x2_result, y2_result], axis=1)
    
    return image, boxes


def restore_text_horizontal_to_vertical(horizontal_text_img):
    shape = horizontal_text_img.shape[:2]
    h, w = shape[:2]
    if len(shape) == 2:
        np_img = horizontal_text_img.transpose((1, 0))[::-1, ...]
    elif len(shape):
        np_img = horizontal_text_img.transpose((1, 0, 2))[::-1, ...]
    return np_img


def restore_boxes_horizontal_to_vertical(boxes, raw_img_shape):
    h, w = raw_img_shape[:2]
    
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
        batch_boxes[i, :num, 5] = 1                 # padding_flag, fg:1, bg:0
    batch_imgs = np.expand_dims(batch_imgs, axis=-1)
    
    return batch_imgs, batch_boxes


def split_boxes_to_fixed_width(boxes, cls_id, im_shape):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    gt_quadrilaterals = np.stack([x1,y1,x2,y1,x2,y2,x1,y2], axis=1)
    gt_boxes, cls_ids = gt_utils.gen_gt_from_quadrilaterals(gt_quadrilaterals,
                                                            class_ids=np.array([cls_id]*boxes.shape[0]),
                                                            image_shape=im_shape,
                                                            width_stride=CTPN_NET_STRIDE)
    gt_boxes = np.concatenate([gt_boxes, cls_ids[:,np.newaxis]], axis=1)
    return gt_boxes


def split_boxes_to_fixed_width_tf(boxes, cls_id, im_shape):
    num_boxes = tf.shape(boxes)[0]
    class_ids = tf.ones(shape=[num_boxes,], dtype=tf.float32) * cls_id
    gt_boxes, class_ids = gt_utils.gen_gt_from_boxes_tf(raw_boxes=boxes,
                                                        class_ids=class_ids,
                                                        im_shape=im_shape,
                                                        width_stride=CTPN_NET_STRIDE)
    return gt_boxes, class_ids


def adjust_img_and_splite_boxes_tf(image, boxes, cls_id,
                                   text_type="horizontal",
                                   fixed_size=BOOK_PAGE_FIXED_SIZE,
                                   max_boxes=BOOK_PAGE_MAX_GT_BOXES):
    # vertical to horizontal
    if text_type.lower() in ("v", "vertical"):
        image, boxes = change_text_vertical_to_horisontal_tf(image, boxes)
    
    # scale image to fixed size
    fixed_size = tf.constant([fixed_size[0], fixed_size[1]], dtype=tf.float32)  # 16的倍数
    raw_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale_ratio = tf.reduce_min(fixed_size / raw_shape)
    new_size = tf.cast(raw_shape * scale_ratio, dtype=tf.int32)
    image = tf.image.resize(image, size=new_size)
    boxes = boxes * scale_ratio
    delta = tf.cast(fixed_size, tf.int32) - new_size
    dh, dw = delta[0], delta[1]
    image = tf.pad(image, paddings=[[0, dh], [0, dw], [0, 0]], mode='CONSTANT', constant_values=255) # fixed_size, 白底黑字
    
    # split_boxes_to_fixed_width
    gt_boxes, class_ids = split_boxes_to_fixed_width_tf(boxes, cls_id, im_shape=tf.shape(image)[:2])
    
    gt_boxes = tf.concat([gt_boxes, class_ids[:, tf.newaxis]], axis=1)
    gt_boxes = tf_utils.pad_to_fixed_size(gt_boxes, max_boxes)  # add a padding_flag
    
    # augment image
    image = tf.cast(image, tf.uint8)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=2.)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=40, max_jpeg_quality=90)
    
    return image, gt_boxes


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
            boxes = split_boxes_to_fixed_width(boxes, class_id=1, im_shape=np_img.shape[:2])
            imgs_list.append(np_img)
            boxes_list.append(boxes)
            i = (i + 1) % n
        
        batch_imgs, batch_boxes = pack_book_pages(imgs_list, boxes_list, background="white")
        inputs_dict = {"batch_images": batch_imgs, "batch_boxes": batch_boxes}
        
        yield inputs_dict, None


def data_generator_with_tfrecords(tfrecords_files, batch_size, text_type="horizontal"):
    data_set = tf.data.TFRecordDataset(tfrecords_files).repeat()
    
    def parse_fn(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'bytes_image': tf.io.FixedLenFeature([], tf.string),
                'img_height': tf.io.FixedLenFeature([], tf.int64),
                'img_width': tf.io.FixedLenFeature([], tf.int64),
                'text_boxes': tf.io.FixedLenFeature([], tf.string)
            })

        img_h = features['img_height']
        img_w = features['img_width']
        image_raw = tf.io.decode_raw(features["bytes_image"], tf.uint8)
        image = tf.reshape(image_raw, shape=[img_h, img_w, 1])
        text_boxes = tf.io.decode_raw(features["text_boxes"], tf.int32)
        text_boxes = tf.reshape(text_boxes, shape=(-1, 4))
        text_boxes = tf.cast(text_boxes, tf.float32)
        
        image, gt_boxes = adjust_img_and_splite_boxes_tf(image, text_boxes, cls_id=1, text_type=text_type)
        image = tf.cast(image, tf.float32)
        
        return {"batch_images": image, "batch_boxes": gt_boxes}
    
    data_set = data_set.map(parse_fn, tf.data.experimental.AUTOTUNE).shuffle(100).batch(batch_size).prefetch(100)
    
    return data_set
        


def data_generator(data_file, batch_size, src_type="images", text_type="horizontal", validation_split=0.1):
    """data generator for fit_generator"""
    with open(data_file, "r", encoding="utf8") as fr:
        lines = [line.strip() for line in fr.readlines()]
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
    

if __name__ == "__main__":
    from config import BOOK_PAGE_TFRECORDS_V
    training_generator, validation_generator = \
        data_generator(data_file=os.path.join(BOOK_PAGE_TFRECORDS_V, "..", "book_pages_tags_ctpn.txt"),
                       batch_size=1,
                       src_type="tfrecords",
                       text_type="vertical")
    
    for data in training_generator:
        print(data)
    
    print("Done !")