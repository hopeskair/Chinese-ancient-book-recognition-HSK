# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers


def topk_compo_seq(compo_logits, k=10, sorted=True):
    compo_scores = tf.math.softmax(compo_logits, axis=-1)
    compo_scores = tf.math.log(compo_scores)
    
    _shape = tf.shape(compo_scores)
    bsize, seq_len = _shape[0], _shape[1]
    
    topk_seq = tf.zeros((bsize, k, 0), dtype=tf.int32)
    seq_scores = tf.zeros((bsize, k), dtype=tf.float32)
    
    def loop_body(i, _topk_seq, _seq_scores):
        curr_scores, curr_indices = tf.math.top_k(compo_scores[:, i], k=k, sorted=sorted)
        
        _seq_scores = tf.expand_dims(_seq_scores, axis=2)   # (bsize, k, 1)
        curr_scores = tf.expand_dims(curr_scores, axis=1)   # (bsize, 1, k)
        _seq_scores += curr_scores                          # (bsize, k, k)
        _seq_scores = tf.reshape((bsize, k*k))              # (bsize, k*k)
        new_seq_scores, _topk_indices = tf.math.top_k(_seq_scores, k=k, sorted=sorted)
        
        _r, _c = _topk_indices // k, _topk_indices % k
        _b = tf.range(bsize, dtype=tf.int32)[:, tf.newaxis]
        _b = tf.tile(_b, multiples=[1, k])
        _b_r = tf.stack([_b, _r], axis=2)
        _b_c = tf.stack([_b, _c], axis=2)
        
        _pre_seq = tf.gather_nd(_topk_seq, indices=_b_r)
        _curr_seq = tf.gather_nd(curr_indices, indices=_b_c)[..., tf.newaxis]
        new_topk_seq = tf.concat([_pre_seq, _curr_seq], axis=2)
        
        return i+1, new_topk_seq, new_seq_scores

    _, topk_seq, seq_scores = tf.while_loop(cond=lambda i, *unused_args: i<seq_len,
                                            body=loop_body,
                                            loop_vars=[0, topk_seq, seq_scores])
    
    return topk_seq, seq_scores
    

class GeneratePrediction(layers.Layer):
    
    def __init__(self, topk=10, stage="test", **kwargs):
        self.topk = topk
        self.stage = stage
        super(GeneratePrediction, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_char_struc, pred_sc_logits, pred_lr_compo_logits, pred_ul_compo_logits = inputs
        
        # 简单汉字预测
        _, pred_sc_labels = tf.math.top_k(pred_sc_logits, k=self.topk, sorted=True)
        
        # 左右结构汉字部件预测
        pred_lr_compo_seq, _ = topk_compo_seq(pred_lr_compo_logits, k=self.topk, sorted=True)
        
        # 上下结构汉字部件预测
        pred_ul_compo_seq, _ = topk_compo_seq(pred_ul_compo_logits, k=self.topk, sorted=True)
        
        # 输出形式
        if self.stage != "train":
            batch_size, seq_len = tf.shape(pred_char_struc)[0], tf.shape(pred_lr_compo_seq)[1]
            
            pred_sc_batch_indices = tf.where(pred_char_struc == 0)
            pred_lr_batch_indices = tf.where(pred_char_struc == 1)
            pred_ul_batch_indices = tf.where(pred_char_struc == 2)
            
            sc_num = tf.shape(pred_sc_batch_indices)[0]
            lr_num = tf.shape(pred_lr_batch_indices)[0]
            ul_num = tf.shape(pred_ul_batch_indices)[0]

            pred_sc_batch_indices = tf.tile(pred_sc_batch_indices, multiples=[1, self.topk])
            pred_lr_batch_indices = tf.tile(pred_lr_batch_indices, multiples=[1, self.topk])
            pred_ul_batch_indices = tf.tile(pred_ul_batch_indices, multiples=[1, self.topk])
            
            order_indices = tf.range(self.topk, dtype=tf.int32)[tf.newaxis, :]
            
            sc_order_indices = tf.tile(order_indices, multiples=[sc_num, 1])
            lr_order_indices = tf.tile(order_indices, multiples=[lr_num, 1])
            ul_order_indices = tf.tile(order_indices, multiples=[ul_num, 1])
            
            # sc prediction
            _first_pos = tf.zeros_like(pred_sc_batch_indices, dtype=tf.int32)
            sc_indices = tf.stack([pred_sc_batch_indices, sc_order_indices, _first_pos], axis=2)
            pred_results = tf.scatter_nd(indices=sc_indices,
                                         updates=pred_sc_labels,
                                         shape=(batch_size, self.topk, seq_len))  # initially zero
            
            # lr prediction
            lr_indices = tf.stack([pred_lr_batch_indices, sc_order_indices], axis=2)
            pred_results = tf.tensor_scatter_nd_add(tensor=pred_results,
                                                    indices=lr_indices,
                                                    updates=pred_lr_compo_seq)
            
            # ul prediction
            ul_indices = tf.stack([pred_ul_batch_indices, ul_order_indices], axis=2)
            pred_results = tf.tensor_scatter_nd_add(tensor=pred_results,
                                                    indices=ul_indices,
                                                    updates=pred_ul_compo_seq)
        else:
            pred_results = tf.zeros([], dtype=tf.int32)
        
        return pred_sc_labels, pred_lr_compo_seq, pred_ul_compo_seq, pred_results
        
        
if __name__ == '__main__':
    print("Done !")
