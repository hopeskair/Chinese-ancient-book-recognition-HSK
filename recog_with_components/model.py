# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers, regularizers

from networks.resnet import ResNet58V2_for_char_recog as ResNet_for_char_recog
from networks.resnext import ResNeXt58_for_char_recog as ResNeXt_for_char_recog
from networks.densenet import DenseNet60_for_char_recog as DenseNet_for_char_recog

from .data_pipeline import image_preprocess_tf
from .losses import chinese_class_loss, chinese_compo_loss
from .gen_prediction import GeneratePrediction

from config import LABEL_SMOOTHING
from config import COMPO_SCORE_THRESH, COMPO_SCORE_ADJUSTMENT_SCALE
from config import INIT_LEARNING_RATE, CHAE_RECOG_LOSS_WEIGHTS, L2_WEIGHT_DECAY
from config import SGD_LEARNING_MOMENTUM, SGD_GRADIENT_CLIP_NORM


def CNN(inputs, scope="densenet"):
    if scope == "resnet":
        outputs = ResNet_for_char_recog(inputs, scope)      # 1/16 size
    elif scope == "resnext":
        outputs = ResNeXt_for_char_recog(inputs, scope)     # 1/16 size
    elif scope == "densenet":
        outputs = DenseNet_for_char_recog(inputs, scope)    # 1/16 size
    else:
        ValueError("Optional CNN scope: 'resnet', 'resnext', 'densenet'.")
    
    return outputs


def build_model(num_chars, num_compo, stage="test", img_size=64, model_struc="densenet"):
    
    batch_images = layers.Input(shape=[img_size, img_size, 3], name='batch_images')
    chinese_char_ids = layers.Input(shape=[], dtype=tf.int32, name='chinese_char_ids')
    compo_embeddings = layers.Input(shape=[num_compo], dtype=tf.int8, name='compo_embeddings')
    
    # ******************** Build *********************
    # image normalization
    convert_imgs = layers.Lambda(image_preprocess_tf, arguments={"stage": stage}, name="image_preprocess")(batch_images)
    
    features = CNN(convert_imgs, scope=model_struc) # 1/16 size
    feat_size = img_size // 16
    
    x = layers.Conv2D(256, (feat_size, feat_size), use_bias=True, name="global_conv")(features)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="global_conv_bn")(x)
    x = layers.Activation('relu', name="global_conv_relu")(x)
    
    # conv实现fc，汉字预测，汉字部件预测
    pred_class_logits = layers.Conv2D(num_chars, kernel_size=1, name='fc_class_logits')(x)
    pred_compo_logits = layers.Conv2D(num_compo, kernel_size=1, name='fc_compo_logits')(x)
    
    pred_class_logits = tf.squeeze(pred_class_logits, axis=[1, 2])
    pred_compo_logits = tf.squeeze(pred_compo_logits, axis=[1, 2])
    
    recog_model = models.Model(inputs=[batch_images, chinese_char_ids, compo_embeddings],
                               outputs=[pred_class_logits, pred_compo_logits])
    
    return recog_model


def work_net(num_chars, num_compo, stage="test", img_size=64, model_struc="densenet"):
    
    recog_model = build_model(num_chars, num_compo, stage, img_size, model_struc)

    batch_images, chinese_char_ids, compo_embeddings = recog_model.inputs
    pred_class_logits, pred_compo_logits = recog_model.outputs
    
    # ******************** Train model **********************
    # 损失函数
    options = {"label_smoothing": LABEL_SMOOTHING}
    class_loss = layers.Lambda(lambda x: chinese_class_loss(*x, **options),
                               name='chinese_class_loss')([chinese_char_ids, pred_class_logits])
    compo_loss = layers.Lambda(lambda x: chinese_compo_loss(*x, **options),
                               name='chinese_compo_loss')([compo_embeddings, pred_compo_logits])
    
    # ******************** Predict model *********************
    targets = GeneratePrediction(compo_score_thresh=COMPO_SCORE_THRESH,
                                 score_adjustment_scale=COMPO_SCORE_ADJUSTMENT_SCALE,
                                 name="gen_prediction")([pred_class_logits, pred_compo_logits, chinese_char_ids])
    
    # *********************** Summary *************************
    summary_metrics = layers.Lambda(summary_fn,
                                    arguments={"compo_score_thresh": COMPO_SCORE_THRESH},
                                    name="summary_fn")([targets, chinese_char_ids, compo_embeddings])
    
    # ********************* Define model **********************
    if stage == 'train':
        return models.Model(inputs=recog_model.inputs, outputs=[class_loss, compo_loss, summary_metrics])
    else:
        return models.Model(inputs=batch_images, outputs=targets)


def summary_fn(inputs, compo_score_thresh=0.7):
    targets, chinese_char_ids, compo_embeddings = inputs
    class_indices, class_scores, \
    compo_scores, adjusted_scores, compo_hit_indices, compo_hit_scores, \
    combined_pred1, combined_pred2, combined_pred3 = targets
    
    class_pred = class_indices[:, 0]
    class_acc = tf.reduce_mean(tf.cast(class_pred == chinese_char_ids, tf.float32))
    
    top3_cls_acc = tf.reduce_mean(tf.cast(tf.reduce_any(chinese_char_ids[:, tf.newaxis] == class_indices[:, :3], axis=1), tf.float32))
    top5_cls_acc = tf.reduce_mean(tf.cast(tf.reduce_any(chinese_char_ids[:, tf.newaxis] == class_indices[:, :5], axis=1), tf.float32))

    # compo score accuracy
    compo_embeddings = tf.cast(compo_embeddings, tf.float32)
    compo_pred = tf.where(compo_scores > compo_score_thresh, 1., 0.)
    compo_pred_eval = tf.cast(compo_pred == compo_embeddings, tf.float32)
    compo_acc = tf.reduce_mean(compo_pred_eval)
    compo_pred_eval_pos = tf.where(compo_embeddings == 1., compo_pred_eval, 0.)
    compo_pos_acc = tf.reduce_sum(compo_pred_eval_pos) / tf.reduce_sum(compo_embeddings)
    compo_pred_eval_neg = tf.where(compo_embeddings == 0., compo_pred_eval, 0.)
    compo_neg_acc = tf.reduce_sum(compo_pred_eval_neg) / tf.reduce_sum(1. - compo_embeddings)
    
    # adjust compo score accuracy
    compo_pred_adjusted = tf.where(adjusted_scores > compo_score_thresh, 1., 0.)
    compo_pred_eval_adjusted = tf.cast(compo_pred_adjusted == compo_embeddings, tf.float32)
    compo_acc_adjusted = tf.reduce_mean(compo_pred_eval_adjusted)
    compo_pred_eval_pos_adjusted = tf.where(compo_embeddings == 1., compo_pred_eval_adjusted, 0.)
    compo_pos_acc_adjusted = tf.reduce_sum(compo_pred_eval_pos_adjusted) / tf.reduce_sum(compo_embeddings)
    compo_pred_eval_neg_adjusted = tf.where(compo_embeddings == 0., compo_pred_eval_adjusted, 0.)
    compo_neg_acc_adjusted = tf.reduce_sum(compo_pred_eval_neg_adjusted) / tf.reduce_sum(1. - compo_embeddings)

    num_compo_hit = tf.reduce_sum(tf.cast(compo_hit_indices != -1, tf.int32), axis=1)
    compo_hit0_ratio = tf.reduce_mean(tf.cast(num_compo_hit == 0, tf.float32))
    compo_hit1_ratio = tf.reduce_mean(tf.cast(num_compo_hit == 1, tf.float32))
    compo_hitn_ratio = tf.reduce_mean(tf.cast(num_compo_hit > 1, tf.float32))
    
    compo_hit_index = tf.where(num_compo_hit == 1, compo_hit_indices[:, 0], -1)
    compo_hit_acc = tf.reduce_mean(tf.cast(compo_hit_index == chinese_char_ids, tf.float32))

    com_pred1_acc = tf.reduce_mean(tf.cast(combined_pred1 == chinese_char_ids, tf.float32))
    com_pred2_acc = tf.reduce_mean(tf.cast(combined_pred2 == chinese_char_ids, tf.float32))
    com_pred3_acc = tf.reduce_mean(tf.cast(combined_pred3 == chinese_char_ids, tf.float32))
    
    return class_acc, top3_cls_acc, top5_cls_acc, \
           compo_acc, compo_pos_acc, compo_neg_acc, \
           compo_hit0_ratio, compo_hit1_ratio, compo_hitn_ratio, \
           compo_hit_acc, com_pred1_acc, com_pred2_acc, com_pred3_acc
           # compo_acc_adjusted, compo_pos_acc_adjusted, compo_neg_acc_adjusted


def compile(keras_model, loss_names=[]):
    """编译模型，添加损失函数，L2正则化"""
    
    # 优化器
    optimizer = optimizers.SGD(INIT_LEARNING_RATE, momentum=SGD_LEARNING_MOMENTUM, clipnorm=SGD_GRADIENT_CLIP_NORM)
    # optimizer = optimizers.RMSprop(learning_rate=INIT_LEARNING_RATE, rho=0.9)
    # optimizer = optimizers.Adagrad(learning_rate=INIT_LEARNING_RATE)
    # optimizer = optimizers.Adadelta(learning_rate=1., rho=0.95)
    # optimizer = optimizers.Adam(learning_rate=INIT_LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # 添加损失函数，首先清除损失，防止重复计算
    # keras_model._losses = []
    # keras_model._per_input_losses = {}
    
    for loss_name in loss_names:
        loss_layer = keras_model.get_layer(loss_name)
        if loss_layer is None: continue
        loss = loss_layer.output * CHAE_RECOG_LOSS_WEIGHTS.get(loss_name, 1.)
        keras_model.add_loss(loss)
    
    # 添加L2正则化，跳过Batch Normalization的gamma和beta权重
    reg_losses = [regularizers.l2(L2_WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                  for w in keras_model.trainable_weights
                  if "gamma" not in w.name and "beta" not in w.name]
    
    keras_model.add_loss(lambda: tf.reduce_sum(reg_losses))
    
    # 编译, 使用虚拟损失
    keras_model.compile(optimizer=optimizer, loss=[None] * len(keras_model.outputs))
    
    # 为每个损失函数增加度量
    add_metrics(keras_model, metric_name_list=loss_names)


def add_metrics(keras_model, metric_name_list, metric_val_list=None):
    """添加度量
    Parameter:
        metric_name_list: 度量名称列表
    """
    if metric_val_list is None:
        names_and_layers = [(name, keras_model.get_layer(name)) for name in metric_name_list]
        metric_list = [(name, layer.output) for name, layer in names_and_layers if layer is not None]
    else:
        assert len(metric_name_list) == len(metric_val_list)
        metric_list = zip(metric_name_list, metric_val_list)
    
    for metric_name, metric_val in metric_list:
        if metric_name in keras_model.metrics_names: continue
        metric_val = metric_val * CHAE_RECOG_LOSS_WEIGHTS.get(metric_name, 1.)
        keras_model.metrics_names.append(metric_name)
        keras_model.add_metric(metric_val, name=metric_name, aggregation='mean')


if __name__ == '__main__':
    print("Done !")
