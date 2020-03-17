# -*- coding: utf-8 -*-
"""
   File Name：     gt_utils
   Description :   gt 四边形分割为固定宽度的系列gt boxes
   Author :       mick.yi
   date：          2019/3/18
"""
import numpy as np
import tensorflow as tf


def linear_fit_y(xs, ys, x_list):
    """
    线性函数拟合两点(x1,y1),(x2,y2)；并求得x_list在的取值
    :param xs:  [x1,x2]
    :param ys:  [y1,y2]
    :param x_list: x轴坐标点,numpy数组 [n]
    :return:
    """
    if xs[0] == xs[1]:  # 垂直线
        return np.ones_like(x_list) * np.mean(ys)
    elif ys[0] == ys[1]:  # 水平线
        return np.ones_like(x_list) * ys[0]
    else:
        fn = np.poly1d(np.polyfit(xs, ys, 1))  # 一元线性函数
        return fn(x_list)


def get_min_max_y(quadrilateral, xs):
    """
    获取指定x值坐标点集合四边形上的y轴最小值和最大值
    :param quadrilateral: 四边形坐标；x1,y1,x2,y2,x3,y3,x4,y4
    :param xs: x轴坐标点,numpy数组 [n]
    :return:  x轴坐标点在四边形上的最小值和最大值
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quadrilateral.tolist()
    y_val_1 = linear_fit_y(np.array([x1, x2]), np.array([y1, y2]), xs)
    y_val_2 = linear_fit_y(np.array([x2, x3]), np.array([y2, y3]), xs)
    y_val_3 = linear_fit_y(np.array([x3, x4]), np.array([y3, y4]), xs)
    y_val_4 = linear_fit_y(np.array([x4, x1]), np.array([y4, y1]), xs)
    y_val_min = []
    y_val_max = []
    for i in range(len(xs)):
        y_val = []
        if min(x1, x2) <= xs[i] <= max(x1, x2):
            y_val.append(y_val_1[i])
        if min(x2, x3) <= xs[i] <= max(x2, x3):
            y_val.append(y_val_2[i])
        if min(x3, x4) <= xs[i] <= max(x3, x4):
            y_val.append(y_val_3[i])
        if min(x4, x1) <= xs[i] <= max(x4, x1):
            y_val.append(y_val_4[i])
        # print("y_val:{}".format(y_val))
        y_val_min.append(min(y_val))
        y_val_max.append(max(y_val))

    return np.array(y_val_min), np.array(y_val_max)


def get_xs_in_range(x_array, x_min, x_max):
    """
    获取分割坐标点
    :param x_array: 宽度方向分割坐标点数组；0~image_width,间隔16 ；如:[0,16,32,...608]
    :param x_min: 四边形x最小值
    :param x_max: 四边形x最大值
    :return:
    """
    indices = np.logical_and(x_array >= x_min, x_array <= x_max)
    xs = x_array[indices]
    # 处理两端的值
    if xs.shape[0] == 0 or xs[0] > x_min:
        xs = np.insert(xs, 0, x_min)
    if xs.shape[0] == 0 or xs[-1] < x_max:
        xs = np.append(xs, x_max)
    return xs


def gen_gt_from_quadrilaterals(gt_quadrilaterals, class_ids, image_shape, width_stride=16, box_min_size=3):
    """从gt四边形生成，宽度固定的gt boxes
    Parameter:
        gt_quadrilaterals: GT四边形坐标,[m,(x1,y1,x2,y2,x3,y3,x4,y4)]
        gt_class_ids: GT四边形类别，一般就是1, [m]
        image_shape:
        width_stride: 分割的步长，一般16
        box_min_size: 分割后GT boxes的最小尺寸
        
    Return:
        gt_boxes：[n,(x1,y1,x2,y2)]
        gt_class_ids: [n]
    """
    h, w = list(image_shape)[:2]
    x_array = np.arange(0, w+1, width_stride, np.float32)  # 固定宽度间隔的x坐标点
    # 每个四边形x 最小值和最大值
    x_min_np = np.min(gt_quadrilaterals[:, ::2], axis=1)
    x_max_np = np.max(gt_quadrilaterals[:, ::2], axis=1)
    gt_boxes = []
    gt_class_ids = []
    for i in np.arange(len(gt_quadrilaterals)):
        xs = get_xs_in_range(x_array, x_min_np[i], x_max_np[i])  # 获取四边形内的x中坐标点
        ys_min, ys_max = get_min_max_y(gt_quadrilaterals[i], xs)
        # print("xs:{}".format(xs))
        # 为每个四边形生成固定宽度的gt
        for j in range(len(xs) - 1):
            x1, x2 = xs[j], xs[j + 1]
            y1, y2 = np.min(ys_min[j:j + 2]), np.max(ys_max[j:j + 2])
            gt_boxes.append([x1, y1, x2, y2])
            gt_class_ids.append(class_ids[i])
    gt_boxes = np.reshape(np.array(gt_boxes), (-1, 4))
    gt_class_ids = np.reshape(np.array(gt_class_ids), (-1,))
    # 过滤高度太小的边框
    height = gt_boxes[:, 2] - gt_boxes[:, 0]
    width = gt_boxes[:, 3] - gt_boxes[:, 1]
    indices = np.where(np.logical_and(height >= 8, width >= box_min_size))
    return gt_boxes[indices], gt_class_ids[indices]


def gen_gt_from_boxes_tf(raw_boxes, class_ids, im_shape, width_stride=16, box_min_size=3):
    im_shape = tf.cast(im_shape, tf.float32)
    h, w = im_shape[0], im_shape[1]
    x_array = tf.range(0, w + 1, delta=width_stride, dtype=tf.float32)  # 固定宽度间隔的x坐标点

    x1, y1, x2, y2 = raw_boxes[:,0], raw_boxes[:,1], raw_boxes[:,2], raw_boxes[:,3]
    
    _x1, _x2 = tf.expand_dims(x1, axis=-1), tf.expand_dims(x2, axis=-1) # (m, 1)
    _x_array = tf.expand_dims(x_array, axis=0)                          # (1, n)
    
    indices = tf.where(tf.logical_and(_x_array > _x1, _x_array < _x2))
    xs = tf.gather(x_array, indices[:, 1])
    x1 = tf.gather(x1, indices[:, 0])
    y1 = tf.gather(y1, indices[:, 0])
    x2 = tf.gather(x2, indices[:, 0])
    y2 = tf.gather(y2, indices[:, 0])
    cls_ids = tf.gather(class_ids, indices[:, 0])

    new_x1 = tf.where(xs - x1 > width_stride, xs - width_stride, x1)
    new_x2 = xs
    gt_boxes = tf.stack([new_x1, y1, new_x2, y2], axis=1)
    
    indices_for_right = tf.where(x2-xs<=16)
    r_x1 = tf.gather(xs, indices_for_right[:, 0])
    r_x2 = tf.gather(x2, indices_for_right[:, 0])
    r_y1 = tf.gather(y1, indices_for_right[:, 0])
    r_y2 = tf.gather(y2, indices_for_right[:, 0])
    r_cls_ids = tf.gather(cls_ids, indices_for_right[:, 0])
    r_boxes = tf.stack([r_x1, r_y1, r_x2, r_y2], axis=1)
    
    gt_boxes = tf.concat([gt_boxes, r_boxes], axis=0)
    cls_ids = tf.concat([cls_ids, r_cls_ids], axis=0)
    
    gt_indices = tf.where(gt_boxes[:, 2] - gt_boxes[:, 0] > box_min_size)
    gt_boxes = tf.gather(gt_boxes, gt_indices[:, 0])
    cls_ids = tf.gather(cls_ids, gt_indices[:, 0])
    
    return gt_boxes, cls_ids