# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers


def topk_compo_seq(compo_logits, k=10, sorted=True):
    compo_scores = tf.math.log_softmax(compo_logits, axis=-1)
    compo_scores, compo_indices = tf.math.top_k(compo_scores, k=k, sorted=sorted)
    
    _shape = tf.shape(compo_scores)
    bsize, seq_len = _shape[0], _shape[1]
    
    top_k_sequences = tf.expand_dims(compo_indices[:, 0], axis=2)
    sequence_scores = compo_scores[:, 0]
    
    def loop_body(i, top_k_seq, seq_scores):
        prev_scores = tf.expand_dims(seq_scores, axis=2)            # (bsize, k, 1)
        next_scores = tf.expand_dims(compo_scores[:, i], axis=1)    # (bsize, 1, k)
        _seq_scores = prev_scores + next_scores                     # (bsize, k, k)
        _seq_scores = tf.reshape(_seq_scores, shape=(bsize, k*k))   # (bsize, k*k)
        seq_scores, topk_coord = tf.math.top_k(_seq_scores, k=k, sorted=sorted)
        
        r_coord, c_coord = topk_coord // k, topk_coord % k
        b_coord = tf.range(bsize, dtype=tf.int32)[:, tf.newaxis]
        b_coord = tf.tile(b_coord, multiples=[1, k])
        b_r_coord = tf.stack([b_coord, r_coord], axis=2)
        b_c_coord = tf.stack([b_coord, c_coord], axis=2)
        
        prev_seq = tf.gather_nd(top_k_seq, indices=b_r_coord)
        next_seq = tf.gather_nd(compo_indices[:, i], indices=b_c_coord)[:, :, tf.newaxis]
        top_k_seq = tf.concat([prev_seq, next_seq], axis=2)
        
        return i+1, top_k_seq, seq_scores
    
    _, top_k_sequences, sequence_scores = tf.while_loop(cond=lambda i, *unused_args: i<seq_len,
                                                        body=loop_body,
                                                        loop_vars=[1, top_k_sequences, sequence_scores],    # 从1开始
                                                        shape_invariants=[tf.TensorShape([]),
                                                                          tf.TensorShape([None, k, None]),
                                                                          sequence_scores.get_shape()]
                                                        )
    
    return top_k_sequences, sequence_scores
    

class GeneratePrediction(layers.Layer):
    
    def __init__(self, topk=10, stage="test", **kwargs):
        self.topk = topk
        self.stage = stage
        super(GeneratePrediction, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        pred_char_struc, pred_sc_logits, pred_lr_compo_logits = inputs
        
        # 简单汉字预测
        _, pred_sc_labels = tf.math.top_k(pred_sc_logits, k=self.topk, sorted=True)
        
        # 左右结构汉字部件预测
        pred_lr_compo_seq, _ = topk_compo_seq(pred_lr_compo_logits, k=self.topk, sorted=True)
        
        # 上下结构汉字部件预测
        # pred_ul_compo_seq, _ = topk_compo_seq(pred_ul_compo_logits, k=self.topk, sorted=True)
        
        # 输出形式
        if self.stage != "train":
            batch_size, seq_len = tf.shape(pred_char_struc)[0], tf.shape(pred_lr_compo_seq)[1]
            
            pred_sc_batch_indices = tf.where(pred_char_struc == 0)
            pred_lr_batch_indices = tf.where(pred_char_struc == 1)
            # pred_ul_batch_indices = tf.where(pred_char_struc == 2)
            
            sc_num = tf.shape(pred_sc_batch_indices)[0]
            lr_num = tf.shape(pred_lr_batch_indices)[0]
            # ul_num = tf.shape(pred_ul_batch_indices)[0]

            pred_sc_batch_indices = tf.tile(pred_sc_batch_indices, multiples=[1, self.topk])
            pred_lr_batch_indices = tf.tile(pred_lr_batch_indices, multiples=[1, self.topk])
            # pred_ul_batch_indices = tf.tile(pred_ul_batch_indices, multiples=[1, self.topk])
            
            order_indices = tf.range(self.topk, dtype=tf.int32)[tf.newaxis, :]
            
            sc_order_indices = tf.tile(order_indices, multiples=[sc_num, 1])
            lr_order_indices = tf.tile(order_indices, multiples=[lr_num, 1])
            # ul_order_indices = tf.tile(order_indices, multiples=[ul_num, 1])
            
            # sc prediction
            _first_pos = tf.zeros_like(pred_sc_batch_indices, dtype=tf.int32)
            sc_indices = tf.stack([pred_sc_batch_indices, sc_order_indices, _first_pos], axis=2)
            pred_results = tf.scatter_nd(indices=sc_indices,
                                         updates=pred_sc_labels,
                                         shape=(batch_size, self.topk, seq_len))  # initially zero
            
            # lr prediction
            lr_indices = tf.stack([pred_lr_batch_indices, lr_order_indices], axis=2)
            pred_results = tf.tensor_scatter_nd_add(tensor=pred_results,
                                                    indices=lr_indices,
                                                    updates=pred_lr_compo_seq)
            
            # ul prediction
            # ul_indices = tf.stack([pred_ul_batch_indices, ul_order_indices], axis=2)
            # pred_results = tf.tensor_scatter_nd_add(tensor=pred_results,
            #                                         indices=ul_indices,
            #                                         updates=pred_ul_compo_seq)
        else:
            pred_results = tf.zeros([], dtype=tf.int32)
        
        return pred_sc_labels, pred_lr_compo_seq, pred_results
        
        
if __name__ == '__main__':
    print("Done !")
