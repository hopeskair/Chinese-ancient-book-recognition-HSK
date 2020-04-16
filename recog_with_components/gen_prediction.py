# -*- encoding: utf-8 -*-
# Author: hushukai

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from util import NUM_CHARS_TASK2 as NUM_CHARS
from util import NUM_COMPO, COMPO_CO_OCCURRENCE_PROB, CID2CHAR_INDICES, ID2COMPO_INDICES
from util import ID2CHAR_DICT


def compo_chinese_mapping_matrix(num_compo=NUM_COMPO, num_chars=NUM_CHARS, cid2char_indices_dict=CID2CHAR_INDICES):
    compo_chinese_matrix = np.zeros(shape=[num_compo, num_chars], dtype=np.int8)
    
    for cid in range(num_compo):
        assert cid in cid2char_indices_dict
        chinese_indices = cid2char_indices_dict[cid]
        cids = [cid,] * len(chinese_indices)
        compo_chinese_matrix[cids, chinese_indices] = 1
    
    return compo_chinese_matrix


def chinese_compo_mapping_matrix(num_chars=NUM_CHARS, num_compo=NUM_COMPO, id2compo_indices_dict=ID2COMPO_INDICES):
    chinese_compo_matrix = np.zeros(shape=[num_chars, num_compo], dtype=np.int8)
    
    for char_id in range(num_chars):
        assert char_id in id2compo_indices_dict
        compo_indices = id2compo_indices_dict[char_id]
        char_ids = [char_id,] * len(compo_indices)
        chinese_compo_matrix[char_ids, compo_indices] = 1
    
    return chinese_compo_matrix


def components_to_chinese_char(compo_scores,
                               compo_co_occurrence_prob,
                               compo_chinese_matrix,
                               compo_score_thresh=0.7,
                               score_adjustment_scale=0.1):
    # 调整部件得分
    compo_indices = tf.where(compo_scores > 0.88)[:, 0]
    
    confidences = tf.gather(compo_scores, compo_indices)
    confidences = tf.expand_dims(confidences, axis=-1)
    co_occurrence_prob = tf.gather(compo_co_occurrence_prob, compo_indices)
    adjusted_scores = co_occurrence_prob * confidences * score_adjustment_scale
    adjusted_scores = tf.reduce_sum(adjusted_scores, axis=0)
    
    adjusted_scores += compo_scores
    max_score_compo_index = tf.argmax(adjusted_scores)
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
        hit_chars = tf.cast(hit_chars, dtype=prev_hit_chars.dtype)
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
    hit_indices = tf.cond(num_indices >= 10,
                          lambda :hit_indices[:10],
                          lambda :tf.pad(hit_indices, [[0, 10-num_indices]], mode='CONSTANT', constant_values=-1))
    
    return [adjusted_scores, hit_indices, hit_score]


def print_compo_pred_py(chinese_char_id, class_indices, char_indices_pred):
    print(ID2CHAR_DICT[chinese_char_id], [ID2CHAR_DICT[index] for index in class_indices])
    print(ID2CHAR_DICT[chinese_char_id], [ID2CHAR_DICT[index] for index in char_indices_pred])
    return
    

class GeneratePrediction(layers.Layer):
    
    def __init__(self, compo_score_thresh=0.7, score_adjustment_scale=0.1, **kwargs):
        self.compo_score_thresh = compo_score_thresh
        self.score_adjustment_scale = score_adjustment_scale
        super(GeneratePrediction, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_class_logits, pred_compo_logits, chinese_char_ids = inputs
        
        class_scores = tf.math.softmax(pred_class_logits, axis=1)
        compo_scores = tf.math.sigmoid(pred_compo_logits)
        
        # 直接预测
        k = 10
        class_scores, class_indices = tf.math.top_k(class_scores, k=k, sorted=True)
        
        # 使用部件预测
        compo_co_occurrence_prob = tf.convert_to_tensor(COMPO_CO_OCCURRENCE_PROB)
        compo_chinese_matrix = tf.convert_to_tensor(compo_chinese_mapping_matrix())
        # chinese_compo_matrix = tf.convert_to_tensor(chinese_compo_mapping_matrix())
        
        options = {"compo_co_occurrence_prob": compo_co_occurrence_prob,
                   "compo_chinese_matrix": compo_chinese_matrix,
                   # "chinese_compo_matrix": chinese_compo_matrix,
                   "compo_score_thresh": self.compo_score_thresh,
                   "score_adjustment_scale": self.score_adjustment_scale}
        adjusted_scores, compo_hit_indices, compo_hit_scores = \
            tf.map_fn(fn=lambda x: components_to_chinese_char(x, **options),
                      elems=compo_scores,
                      dtype=[tf.float32, tf.int32, tf.float32])
        
        num_compo_hit = tf.reduce_sum(tf.cast(compo_hit_indices != -1, tf.int32), axis=1)
        combined_pred1 = tf.where(num_compo_hit == 1, compo_hit_indices[:, 0], class_indices[:, 0])
        combined_pred2 = tf.where(tf.logical_and(class_scores[:,0] < 0.85, num_compo_hit == 1), compo_hit_indices[:, 0], class_indices[:, 0])
        
        hit_compare = tf.reduce_any(class_indices[:, :, tf.newaxis] == compo_hit_indices[:, tf.newaxis, :], axis=2)
        hit_compare = tf.cast(hit_compare, tf.int8)
        _j = tf.argsort(hit_compare, axis=1, direction='DESCENDING', stable=True)[:, 0]
        _i = tf.range(tf.shape(hit_compare)[0], dtype=_j.dtype)
        combined_hit_indices = tf.stack([_i, _j], axis=1)
        combined_pred3 = tf.gather_nd(class_indices, indices=combined_hit_indices)
        
        return class_indices, class_scores, \
               compo_scores, adjusted_scores, compo_hit_indices, compo_hit_scores, \
               combined_pred1, combined_pred2, combined_pred3


if __name__ == '__main__':
    print("Done !")
