# -*- coding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers, regularizers

from ..networks.resnet import ResNet58V2_for_ctpn as ResNet_for_ctpn
from ..networks.resnext import ResNeXt58_for_ctpn as ResNeXt_for_ctpn
from ..networks.densenet import DenseNet60_for_ctpn as DenseNet_for_ctpn

from .anchors import generate_anchors_then_filter
from .gen_target import CtpnTarget
from .text_proposals import TextProposal
from .losses import ctpn_cls_loss, ctpn_regress_loss, side_regress_loss

from ..config import CTPN_ANCHORS_WIDTH, CTPN_ANCHORS_HEIGHTS
from ..config import CTPN_TRAIN_ANCHORS_PER_IMAGE, CTPN_ANCHOR_POSITIVE_RATIO
from ..config import CTPN_USE_SIDE_REFINE, CTPN_PROPOSALS_MAX_NUM
from ..config import CTPN_PROPOSALS_MIN_SCORE, CTPN_PROPOSALS_NMS_THRESH
from ..config import CTPN_INIT_LEARNING_RATE, CTPN_LEARNING_MOMENTUM, CTPN_GRADIENT_CLIP_NORM
from ..config import CTPN_LOSS_WEIGHTS, CTPN_WEIGHT_DECAY


def CNN(inputs, scope="densenet"):
    """cnn for CTPN"""
    
    if "resnet" in scope:
        outputs = ResNet_for_ctpn(inputs, scope)  # 1/16 size
    elif "resnext" in scope:
        outputs = ResNeXt_for_ctpn(inputs, scope)  # 1/16 size
    elif "densenet" in scope:
        outputs = DenseNet_for_ctpn(inputs, scope)  # 1/16 size
    else:
        ValueError("Optional CNN scope: 'resnet*', 'resnext*', 'densenet*'.")
    
    return outputs


def Bidirectional_RNN(inputs, rnn_units=256, scope="gru"):
    """Bidirectional RNN of CTPN."""
    
    if "lstm" in scope:
        rnn_layer = layers.LSTM
    elif "gru" in scope:
        rnn_layer = layers.GRU
    else:
        ValueError("Optional RNN layer: '*lstm', '*gru'.")
    
    with K.name_scope("bidirectional_" + scope):
        # Based on available runtime hardware, this layer will choose different
        # implementations (pure-tf or cudnn-based).
        outputs = layers.TimeDistributed(
            layers.Bidirectional(rnn_layer(units=rnn_units, dropout=0.2, return_sequences=True, kernel_initializer='he_normal'),
                                 merge_mode="concat")
        )(inputs)
    
    return outputs


def ctpn_net(stage="train", model_struc="densenet_gru"):
    num_anchors = len(CTPN_ANCHORS_HEIGHTS)
    
    batch_images = layers.Input(shape=[None, None, 1], name='batch_images')
    batch_boxes = layers.Input(shape=[None, 6], name='batch_boxes') # x1, y1, x2, y2, class_id, padding_flag
    
    features = CNN(batch_images, scope=model_struc.split("_")[0])   # 1/16
    
    predict_class_logits, predict_deltas, predict_side_deltas = \
        CTPN(features, num_anchors, rnn_units=128, fc_units=256, rnn_type=model_struc.split("_")[1])

    valid_anchors, valid_indices = layers.Lambda(generate_anchors_then_filter,
                                                 arguments={"feat_stride":16,
                                                            "anchor_width":CTPN_ANCHORS_WIDTH,
                                                            "anchor_heights":CTPN_ANCHORS_HEIGHTS},
                                                 name="gen_ctpn_anchors"
                                                 )(feat_shape=tf.shape(features))
    
    if stage == 'train':
        targets = CtpnTarget(train_anchors_num=CTPN_TRAIN_ANCHORS_PER_IMAGE,
                             positive_ratios=CTPN_ANCHOR_POSITIVE_RATIO,
                             name='ctpn_target')([batch_boxes, valid_anchors, valid_indices])
        deltas, class_ids, anchor_indices_sampled = targets[:3]
        
        # 损失函数
        cls_loss = layers.Lambda(lambda x: ctpn_cls_loss(*x),
                                 name='ctpn_class_loss')([predict_class_logits, class_ids, anchor_indices_sampled])
        regress_loss = layers.Lambda(lambda x: ctpn_regress_loss(*x),
                                     name='ctpn_regress_loss')([predict_deltas, deltas, class_ids, anchor_indices_sampled])
        # side_loss = layers.Lambda(lambda x: side_regress_loss(*x),
        #                           name='side_regress_loss')([predict_deltas, deltas, class_ids, anchor_indices_sampled])
        
        model = models.Model(inputs=[batch_images, batch_boxes], outputs=[regress_loss, cls_loss])

    else:
        text_boxes, text_scores, text_class_logits = \
            TextProposal(nms_max_outputs=CTPN_PROPOSALS_MAX_NUM,
                         cls_score_thresh=CTPN_PROPOSALS_MIN_SCORE,
                         iou_thresh=CTPN_PROPOSALS_NMS_THRESH,
                         use_side_refine=CTPN_USE_SIDE_REFINE,
                         name="text_proposals"
            )([predict_deltas, predict_side_deltas, predict_class_logits, valid_anchors, valid_indices])
        
        model = models.Model(inputs=batch_images, outputs=[text_boxes, text_scores])
        
    return model


def CTPN(features, num_anchors, rnn_units=128, fc_units=512, rnn_type="gru"):
    
    rnn_outputs = Bidirectional_RNN(features, rnn_units, scope=rnn_type)

    # conv实现fc
    x = layers.Conv2D(fc_units, kernel_size=(1, 1), name='fc_output')(rnn_outputs)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="fc_bn")(x)
    x = layers.Activation('relu', name="fc_relu")(x)
    
    # 分类
    class_logits = layers.Conv2D(2*num_anchors, kernel_size=(1, 1), name='cls')(x)
    class_logits = layers.Reshape(target_shape=(-1, 2), name='cls_reshape')(class_logits)
    # 中心点垂直坐标和高度回归
    predict_deltas = layers.Conv2D(2 * num_anchors, kernel_size=(1, 1), name='deltas')(x)
    predict_deltas = layers.Reshape(target_shape=(-1, 2), name='deltas_reshape')(predict_deltas)
    # 侧边精调(只需要预测x偏移即可)
    predict_side_deltas = layers.Conv2D(num_anchors, kernel_size=(1, 1), name='side_deltas')(x)
    predict_side_deltas = layers.Reshape(target_shape=(-1, 1), name='side_deltas_reshape')(predict_side_deltas)
    
    return class_logits, predict_deltas, predict_side_deltas


def get_layer(keras_model, name):
    for layer in keras_model.layers:
        if layer.name == name:
            return layer
    return None


def compile(keras_model, loss_names=[]):
    """编译模型，添加损失函数，L2正则化"""
    
    # 优化器
    optimizer = optimizers.SGD(lr=CTPN_INIT_LEARNING_RATE,
                               momentum=CTPN_LEARNING_MOMENTUM,
                               clipnorm=CTPN_GRADIENT_CLIP_NORM)
    
    # 添加损失函数，首先清除损失，防止重复计算
    keras_model._losses = []
    keras_model._per_input_losses = {}

    for loss_name in loss_names:
        loss_layer = get_layer(keras_model, loss_name)
        if loss_layer is None or loss_layer.output in keras_model.losses:
            continue
        loss = loss_layer.output * CTPN_LOSS_WEIGHTS.get(loss_name, 1.)
        keras_model.add_loss(loss)

    # 添加L2正则化，跳过Batch Normalization的gamma和beta权重
    reg_losses = [regularizers.l2(CTPN_WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                  for w in keras_model.trainable_weights
                  if "gamma" not in w.name and "beta" not in w.name]
    keras_model.add_loss(tf.add_n(reg_losses))

    # 编译, 使用虚拟损失
    keras_model.compile(optimizer=optimizer, loss=[None] * len(keras_model.outputs))
    
    # 为每个损失函数增加度量
    for loss_name in loss_names:
        if loss_name in keras_model.metrics_names: continue
        
        loss_layer = get_layer(keras_model, loss_name)
        if loss_layer is None: continue
        
        keras_model.metrics_names.append(loss_name)
        loss = loss_layer.output * CTPN_LOSS_WEIGHTS.get(loss_name, 1.)
        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list):
    """添加度量
    Parameter:
        metric_name_list: 度量名称列表
        metric_tensor_list: 度量张量列表
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        keras_model.metrics_tensors.append(tf.reduce_mean(tensor, keepdims=False))
