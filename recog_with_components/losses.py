# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend as K


def chinese_class_loss(chinese_char_ids, pred_class_logits, label_smoothing=0.1):
    # label smoothing
    num_classes = tf.shape(pred_class_logits)[1]
    y_true = tf.one_hot(indices=chinese_char_ids, depth=num_classes, dtype=tf.float32)
    y_true = y_true * (1. - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    loss = losses.categorical_crossentropy(y_true, y_pred=pred_class_logits, from_logits=True)
    
    return tf.reduce_mean(loss)


def chinese_compo_loss(compo_embeddings, pred_compo_logits, label_smoothing=0.1):
    # label smoothing
    y_true = tf.where(compo_embeddings == 1, 1. - label_smoothing, label_smoothing)
    
    # weights
    num_compo = tf.cast(tf.shape(compo_embeddings)[1], tf.float32)
    num_pos = tf.reduce_sum(tf.cast(compo_embeddings, tf.float32), axis=1, keepdims=True)
    num_neg = num_compo - num_pos
    weights = tf.where(compo_embeddings == 1, 0.7 / num_pos, 0.3 / num_neg)
    
    # The last dimension has been averaged. output shape [batch_size,]
    # loss = losses.binary_crossentropy(y_true, pred_compo_logits, from_logits=True)
    
    loss = K.binary_crossentropy(y_true, pred_compo_logits, from_logits=True)
    loss = loss * weights
    loss = tf.reduce_sum(loss, axis=1)
    
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    print("Done !")
