# -*- encoding: utf-8 -*-
# Author: hushukai

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from util import NUM_CHARS_TASK2 as NUM_CHARS
from util import NUM_COMPO, COMPO_CO_OCCURRENCE_PROB, CID2CHAR_INDICES


def compo_chinese_mapping_matrix(num_compo=NUM_COMPO, num_chars=NUM_CHARS, cid2char_indices_dict=CID2CHAR_INDICES):
    compo_chinese_matrix = np.zeros(shape=[num_compo, num_chars], dtype=np.int8)
    
    for cid in range(num_compo):
        assert cid in cid2char_indices_dict
        chinese_indices = cid2char_indices_dict[cid]
        cids = [cid,] * len(chinese_indices)
        compo_chinese_matrix[cids, chinese_indices] = 1
    
    return compo_chinese_matrix


def components_to_chinese_char(compo_scores,
                               compo_co_occurrence_prob,
                               compo_chinese_matrix,
                               compo_score_thresh=0.8,
                               score_adjustment_scale=0.1):
    # 调整部件得分
    compo_indices = tf.where(compo_scores > compo_score_thresh)[:, 0]
    
    confidences = tf.gather(compo_scores, compo_indices)
    confidences = tf.expand_dims(confidences, axis=-1)
    co_occurrence_prob = tf.gather(compo_co_occurrence_prob, compo_indices)
    adjusted_scores = co_occurrence_prob * confidences * score_adjustment_scale
    adjusted_scores = tf.reduce_sum(adjusted_scores, axis=0)
    
    compo_scores += adjusted_scores
    max_score_compo_index = tf.argmax(compo_scores)
    max_score = compo_scores[max_score_compo_index]

    # 部件集
    compo_indices = tf.where(compo_scores > compo_score_thresh)[:, 0]
    compo_scores = tf.gather(compo_scores, compo_indices)
    sorted_indices = tf.argsort(compo_scores, direction='DESCENDING')
    compo_scores = tf.gather(compo_scores, sorted_indices)
    compo_indices = tf.gather(compo_indices, sorted_indices)
    num_selected_compo = tf.shape(compo_indices)[0]
    
    # 提取包含部件集的汉字
    hit_chars = compo_chinese_matrix[max_score_compo_index]
    prev_hit_chars = hit_chars
    
    def loop_condition(i, hit_chars, prev_hit_chars_unused):
        num_hit = tf.reduce_sum(tf.cast(hit_chars, tf.int32))   # tf.int8类型加和后可能溢出
        return tf.logical_and(i < num_selected_compo, num_hit > 1)

    def loop_body(i, hit_chars, prev_hit_chars_unused):
        prev_hit_chars = hit_chars
        compo_index = compo_indices[i]
        compo_chars = compo_chinese_matrix[compo_index]
        hit_chars = tf.where(hit_chars + compo_chars == 2, 1, 0)
        return i+1, hit_chars, prev_hit_chars

    i, hit_chars, prev_hit_chars = tf.while_loop(cond=loop_condition,
                                                 body=loop_body,
                                                 loop_vars=[1, hit_chars, prev_hit_chars])  # 从1开始

    num_hit = tf.reduce_sum(tf.cast(hit_chars, tf.int32))
    i, hit_chars = tf.cond(num_hit == 0, lambda :(i-1, prev_hit_chars), lambda :(i, hit_chars))
    
    hit_score = tf.cond(i == 1, lambda :max_score, lambda :tf.reduce_min(compo_scores[:i]))
    hit_indices = tf.where(hit_chars == 1)[:, 0]
    hit_indices = tf.cast(hit_indices, tf.int32)
    
    num_indices = tf.shape(hit_indices)[0]
    hit_indices = tf.cond(num_indices >= 5,
                          lambda :hit_indices[:5],
                          lambda :tf.pad(hit_indices, [[0, 5-num_indices]], mode='CONSTANT', constant_values=-1))
    
    return [hit_indices, hit_score]


class GeneratePrediction(layers.Layer):
    
    def __init__(self, compo_score_thresh=0.8, score_adjustment_scale=0.1, **kwargs):
        self.compo_score_thresh = compo_score_thresh
        self.score_adjustment_scale = score_adjustment_scale
        super(GeneratePrediction, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_class_logits, pred_compo_logits = inputs
        
        class_scores = tf.math.softmax(pred_class_logits, axis=1)
        compo_scores = tf.math.sigmoid(pred_compo_logits)
        
        # 直接预测
        class_scores, class_indices = tf.math.top_k(class_scores, k=5, sorted=True)
        
        # 使用部件预测
        compo_co_occurrence_prob = tf.convert_to_tensor(COMPO_CO_OCCURRENCE_PROB)
        compo_chinese_matrix = tf.convert_to_tensor(compo_chinese_mapping_matrix())
        
        options = {"compo_co_occurence_prob": compo_co_occurrence_prob,
                   "compo_chinese_matric": compo_chinese_matrix,
                   "compo_score_thresh": self.compo_score_thresh,
                   "score_adjustment_scale": self.score_adjustment_scale}
        compo_hit_indices, compo_hit_scores = tf.map_fn(fn=lambda x: components_to_chinese_char(x, **options),
                                                        elems=compo_scores,
                                                        dtype=[tf.int32, tf.float32])
        
        num_compo_hit = tf.reduce_sum(tf.cast(tf.where(compo_hit_indices != -1), tf.int32), axis=1)
        combined_pred1 = tf.where(num_compo_hit == 1, compo_hit_indices[:,0], class_indices[:, 0])
        combined_pred2 = tf.where(tf.logical_and(class_scores[:,0] < 0.85, num_compo_hit == 1), compo_hit_indices[:,0], class_indices[:, 0])
        
        return class_indices, class_scores, compo_hit_indices, compo_hit_scores, combined_pred1, combined_pred2


if __name__ == '__main__':
    print("Done !")
