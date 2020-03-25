# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .utils import pad_to_fixed_size_tf, remove_pad_tf


def nms(split_positions, scores, img_width, score_thresh=0.7, distance_thresh=16, max_outputs=50):
    """Non-Maximum-Suppression"""
    img_width = tf.cast(img_width, tf.float32)
    indices = tf.where(tf.logical_and(
        tf.logical_and(split_positions[..., 0] >= 0,
                       split_positions[..., 0] <= img_width - 1),
        tf.logical_and(split_positions[..., 1] >= 0,
                       split_positions[..., 1] <= img_width - 1)))[:, 0]
    scores = tf.gather(scores, indices)
    split_positions = tf.gather(split_positions, indices)
    
    indices = tf.where(scores >= score_thresh)[:, 0]
    scores = tf.gather(scores, indices)
    split_positions = tf.gather(split_positions, indices)
    
    ordered_indices = tf.argsort(scores)[::-1]
    ordered_scores = tf.gather(scores, ordered_indices)
    ordered_positions = tf.gather(split_positions, ordered_indices)
    
    nms_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    nms_positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def loop_condition(j, ordered_scores, *args):
        return tf.shape(ordered_scores)[0] > 0

    def loop_body(j, ordered_scores, ordered_positions, nms_scores, nms_positions):
        curr_score = ordered_scores[0]
        curr_positions = ordered_positions[0]
        nms_scores = nms_scores.write(j, curr_score)
        nms_positions = nms_positions.write(j, curr_positions)
        
        distances = tf.reduce_mean(ordered_positions[1:], axis=1) - tf.reduce_mean(curr_positions, keepdims=True)
        _indices = tf.where(tf.abs(distances) > distance_thresh)[:, 0] + 1
        
        ordered_scores = tf.gather(ordered_scores, _indices)
        ordered_positions = tf.gather(ordered_positions, _indices)
        return j + 1, ordered_scores, ordered_positions, nms_scores, nms_positions

    _, _, _, nms_scores, nms_positions = tf.while_loop(cond=loop_condition, body=loop_body,
                                                       loop_vars=[0, ordered_scores, ordered_positions, nms_scores, nms_positions])

    nms_scores = nms_scores.stack()
    nms_positions = nms_positions.stack()
    
    nms_scores = pad_to_fixed_size_tf(nms_scores[:, tf.newaxis], max_outputs)
    nms_positions = pad_to_fixed_size_tf(nms_positions, max_outputs)
    
    return [nms_positions, nms_scores]


class ExtractSplitPosition(layers.Layer):
    
    def __init__(self, feat_stride=16, cls_score_thresh=0.7, distance_thresh=16, nms_max_outputs=50, **kwargs):
        self.feat_stride = feat_stride
        self.cls_score_thresh = cls_score_thresh
        self.distance_thresh = distance_thresh
        self.nms_max_outputs = nms_max_outputs
        super(ExtractSplitPosition, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_cls_logit, pred_delta, img_width = inputs
        feat_width = img_width // self.feat_stride

        interval_center = (tf.range(0, feat_width, dtype=tf.float32) + 0.5) * self.feat_stride
        interval_center = tf.tile(interval_center[:, tf.newaxis], multiples=[1, 2])
        interval_center = interval_center[tf.newaxis, ...]  # shape (1, feat_width, 2)
        
        pred_split_positions = pred_delta * self.feat_stride + interval_center
        scores = K.sigmoid(pred_cls_logit)

        # 非极大抑制
        options = {"img_width": img_width,
                   "score_thresh": self.cls_score_thresh,
                   "distance_thresh": self.distance_thresh,
                   "max_outputs": self.nms_max_outputs}
        outputs = tf.map_fn(fn=lambda x: nms(*x, **options),
                            elems=[pred_split_positions, scores],
                            dtype=[tf.float32, tf.float32])
        
        return outputs


if __name__ == '__main__':
    print("Done !")
