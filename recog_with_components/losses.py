# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import losses


def char_struc_loss(real_char_struc, pred_struc_logits, label_smoothing=0.1):
    # label smoothing
    num_classes = tf.shape(pred_struc_logits)[1]
    y_true = tf.one_hot(indices=real_char_struc, depth=num_classes, dtype=tf.float32)
    y_true = y_true * (1. - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    loss = losses.categorical_crossentropy(y_true, y_pred=pred_struc_logits, from_logits=True)
    
    return tf.reduce_mean(loss)


def sc_char_loss(sc_labels, pred_sc_logits, label_smoothing=0.1):
    # label smoothing
    num_classes = tf.shape(pred_sc_logits)[1]
    y_true = tf.one_hot(indices=sc_labels, depth=num_classes, dtype=tf.float32)
    y_true = y_true * (1. - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    loss = losses.categorical_crossentropy(y_true, y_pred=pred_sc_logits, from_logits=True)
    
    return tf.reduce_mean(loss)


def lr_compo_loss(lr_compo_seq, pred_lr_compo_logits, label_smoothing=0.1):
    # label smoothing
    num_classes = tf.shape(pred_lr_compo_logits)[2]
    y_true = tf.one_hot(indices=lr_compo_seq, depth=num_classes, dtype=tf.float32)
    y_true = y_true * (1. - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    loss = losses.categorical_crossentropy(y_true, y_pred=pred_lr_compo_logits, from_logits=True)
    # loss = tf.reduce_sum(loss, axis=1)
    
    return tf.reduce_mean(loss)


def ul_compo_loss(ul_compo_seq, pred_ul_compo_logits, label_smoothing=0.1):
    # label smoothing
    num_classes = tf.shape(pred_ul_compo_logits)[2]
    y_true = tf.one_hot(indices=ul_compo_seq, depth=num_classes, dtype=tf.float32)
    y_true = y_true * (1. - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    loss = losses.categorical_crossentropy(y_true, y_pred=pred_ul_compo_logits, from_logits=True)
    # loss = tf.reduce_sum(loss, axis=1)
    
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    print("Done !")
