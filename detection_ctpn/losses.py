# -*- coding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K


def ctpn_cls_loss(predict_cls_logits, tgt_cls_ids, anchor_indices_sampled):
    """分类损失
    Parameter:
        predict_logits: 预测的anchor类别logits，[batch_num, anchors_num, 2], fg or bg
        tgt_cls_ids: 真实的类别目标，[batch_num, ctpn_train_anchor_num, (class_id, padding_flag)]
        anchor_indices_sampled: 正负样本索引，[batch_num, ctpn_train_anchors, (idx, padding_flag)]
    """
    # 去除padding
    train_indices = tf.where(tf.equal(tgt_cls_ids[:, :, -1], 1))  # no_pad:1, pad:0.
    
    tgt_cls_ids = tf.gather_nd(tgt_cls_ids[..., 0], train_indices)  # fg:1, bg:0
    
    true_anchor_indices = tf.gather_nd(anchor_indices_sampled[..., 0], train_indices)
    
    # batch索引
    batch_indices = train_indices[:, 0]
    # 每个训练anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(true_anchor_indices, dtype=tf.int64)], axis=1)
    # 获取预测的anchor类别
    predict_cls_logits = tf.gather_nd(predict_cls_logits, train_indices_2d)

    # 交叉熵损失函数
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt_cls_ids, logits=predict_cls_logits)
    loss = K.mean(loss)
    
    return loss


def smooth_l1_loss(y_true, y_predict, sigma2=9.0):
    """
    smooth L1损失函数；   0.5 * sigma2 * x^2 if |x| < 1/sigma2 else |x|-0.5/sigma2; x是diff
    """
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1./sigma2), 0.5 * sigma2 * tf.pow(abs_diff, 2), abs_diff - 0.5/sigma2)
    return loss


def ctpn_regress_loss(predict_deltas, tgt_deltas, tgt_cls_ids, anchor_indices_sampled):
    """高度方向的中心点偏移、高度尺寸缩放回归损失
    Parameter:
        predict_deltas: 预测的dy,dh回归目标，[batch_num, anchors_num, 2]
        tgt_deltas: 真实的回归目标，[batch_num, ctpn_train_anchor_num, (dy,dh,dx,padding_flag)]
        tgt_cls_ids: 真实的类别目标，[batch_num, ctpn_train_anchor_num, (class_id, padding_flag)]
        anchor_indices_sampled: 正负样本索引，[batch_num, ctpn_train_anchors, (idx, padding_flag)]
    """
    # 去除padding和负样本
    positive_indices = tf.where(tf.equal(tgt_cls_ids[:,:,0], 1)) # fg:1
    
    deltas_yh = tf.gather_nd(tgt_deltas[..., :2], positive_indices)
    true_anchor_indices = tf.gather_nd(anchor_indices_sampled[..., 0], positive_indices)
    
    # batch索引
    batch_indices = positive_indices[:, 0]
    # 正样本anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(true_anchor_indices, dtype=tf.int64)], axis=1)
    # 正样本anchor对应的预测
    predict_deltas = tf.gather_nd(predict_deltas, train_indices_2d, name='ctpn_regress_loss')
    
    # Smooth-L1 # 非常重要，不然报NAN
    loss = K.switch(tf.size(deltas_yh) > 0, smooth_l1_loss(deltas_yh, predict_deltas), tf.constant(0.0))
    loss = K.mean(loss)
    
    return loss


def side_regress_loss(predict_side_deltas, tgt_deltas, tgt_cls_ids, anchor_indices_sampled):
    """侧边改善回归损失
    Parameter:
        predict_deltas: 预测的dx回归目标，[batch_num, anchors_num, 2]
        tgt_deltas: 真实的回归目标，[batch_num, ctpn_train_anchor_num, (dy,dh,dx,padding_flag)]
        tgt_cls_ids: 真实的类别目标，[batch_num, ctpn_train_anchor_num, (class_id, padding_flag)]
        anchor_indices_sampled: 正负样本索引，[batch_num, ctpn_train_anchors, (idx, padding_flag)]
    """
    # 去除padding和负样本
    positive_indices = tf.where(tf.equal(tgt_cls_ids[:, :, 0], 1))  # fg:1
    
    deltas_x = tf.gather_nd(tgt_deltas[..., 2:3], positive_indices)
    true_anchor_indices = tf.gather_nd(anchor_indices_sampled[..., 0], positive_indices)

    # batch索引
    batch_indices = positive_indices[:, 0]
    # 正样本anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(true_anchor_indices, dtype=tf.int64)], axis=1)
    # 正样本anchor对应的预测
    predict_side_deltas = tf.gather_nd(predict_side_deltas, train_indices_2d, name='ctpn_side_regress_loss')

    # Smooth-L1 # 非常重要，不然报NAN
    loss = K.switch(tf.size(deltas_x) > 0, smooth_l1_loss(deltas_x, predict_side_deltas), tf.constant(0.0))
    loss = K.mean(loss)
    
    return loss
