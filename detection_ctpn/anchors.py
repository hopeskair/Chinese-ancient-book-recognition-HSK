# -*- coding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
import numpy as np


def get_base_anchors(width, heights):
    """base anchors"""
    num_anchors = len(heights)
    w = np.array([width] * num_anchors, dtype=np.float32)
    h = np.array(heights, dtype=np.float32)
    base_anchors = np.stack([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis=1)  # x1,y1,x2,y2
    return base_anchors


def generate_anchors_then_filter(feat_shape, feat_stride, anchor_width, anchor_heights):
    base_anchors = get_base_anchors(anchor_width, anchor_heights)
    
    feat_h, feat_w = feat_shape[:2]
    grid_ctr_x = (tf.range(feat_w, dtype=tf.float32) + 0.5) * feat_stride
    grid_ctr_y = (tf.range(feat_h, dtype=tf.float32) + 0.5) * feat_stride

    grid_ctr_x, grid_ctr_y = tf.meshgrid(grid_ctr_x, grid_ctr_y)
    grid_ctr = tf.stack([grid_ctr_x, grid_ctr_y, grid_ctr_x, grid_ctr_y], axis=-1)
    grid_ctr = tf.expand_dims(grid_ctr, axis=-2)
    
    anchors = grid_ctr + base_anchors
    anchors = tf.reshape(anchors, shape=[-1, 4])
    
    inp_h, inp_w = feat_stride[:2] * feat_stride

    valid_tag = tf.logical_and(
        tf.logical_and(anchors[:,0] >= 0, anchors[:,1] >= 0),
        tf.logical_and(anchors[:,2] < inp_w, anchors[:,3] < inp_h))
    
    valid_anchors = tf.boolean_mask(anchors, valid_tag)
    valid_indices = tf.where(valid_tag)[:,0]
    
    return valid_anchors, valid_indices
