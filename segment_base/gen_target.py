# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers


class SegmentTarget(layers.Layer):
    def __init__(self, feat_stride=16, pos_weight=2., neg_weight=1., pad_weight=0.2, **kwargs):
        self.feat_stride = feat_stride
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.pad_weight = pad_weight
        super(SegmentTarget, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        split_line_pos, feat_width, real_features_width = inputs  # 要求划分位置是有序的，从小到大; padding value -1
        
        batch_size = split_line_pos.shape[0]
        split_line_x1, split_line_x2 = split_line_pos[..., 0], split_line_pos[..., 1]
        split_line_center = (split_line_x1 + split_line_x2) * 0.5
        
        x_interval_num = tf.floor(split_line_center / self.feat_stride)
        
        # 如果多条划分线落在同一区间，该区间只负责预测第一条划分线，其它的忽略；这种情况几乎不可能发生
        _first_col = tf.constant(-1., shape=[batch_size, 1])
        prev_interval_num = tf.concat([_first_col, x_interval_num[:, :-1]], axis=1)
        x_interval_num = tf.where(x_interval_num == prev_interval_num, -1., x_interval_num)
        
        split_line_indices = tf.where(x_interval_num >= 0)
        
        x_interval_num = tf.gather_nd(x_interval_num, split_line_indices)
        split_line_pos = tf.gather_nd(split_line_pos, split_line_indices)
        
        batch_indices = split_line_indices[:, 0]
        x_interval_num = tf.cast(x_interval_num, tf.int64)
        target_indices = tf.stack([batch_indices, x_interval_num], axis=1)
        pre_cls_ids = tf.ones_like(target_indices[:, 0], dtype=tf.float32)

        interval_cls_ids = tf.scatter_nd(indices=target_indices, updates=pre_cls_ids, shape=[batch_size, feat_width])   # 0, 1
        interval_split_line = tf.scatter_nd(indices=target_indices, updates=split_line_pos, shape=[batch_size, feat_width, 2])
        
        interval_center = (tf.range(0, feat_width, dtype=tf.float32) + 0.5) * self.feat_stride
        interval_center = tf.tile(interval_center[:, tf.newaxis], multiples=[1, 2])
        interval_center = interval_center[tf.newaxis, ...]  # shape (1, feat_width, 2)
        
        split_line_delta = (interval_split_line - interval_center) / self.feat_stride

        feat_region = tf.expand_dims(tf.range(0, feat_width, dtype=tf.int32), axis=0)
        real_features_width = tf.expand_dims(tf.cast(real_features_width, tf.int32), axis=1)
        inside_weights = tf.where(feat_region <= real_features_width, self.neg_weight, self.pad_weight)
        inside_weights = tf.where(interval_cls_ids == 1, self.pos_weight, inside_weights)
        
        return interval_cls_ids, inside_weights, split_line_delta
        
        
if __name__ == '__main__':
    print("Done !")
