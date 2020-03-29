# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K


def interval_cls_loss(interval_cls_goals, pred_cls_logit, inside_weights):
    
    loss = K.binary_crossentropy(interval_cls_goals, pred_cls_logit, from_logits=True)
    loss = loss * inside_weights
    
    return tf.reduce_sum(loss) / tf.reduce_sum(inside_weights)


def split_line_regress_loss(split_line_delta, pred_delta, interval_mask):
    
    loss = smooth_l1_loss(split_line_delta, pred_delta)
    loss = tf.reduce_sum(loss, axis=-1) * interval_mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(interval_mask)
    

def smooth_l1_loss(y_true, y_predict, sigma2=9.0):
    """
    smooth L1损失函数；   0.5 * sigma2 * x^2 if |x| < 1/sigma2 else |x|-0.5/sigma2; x是diff
    """
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1./sigma2), 0.5 * sigma2 * tf.pow(abs_diff, 2), abs_diff - 0.5/sigma2)
    return loss


if __name__ == '__main__':
    print("Done !")
