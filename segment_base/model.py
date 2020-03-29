# -*- coding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers, regularizers

from networks.resnet import ResNet64V2_segment_book_page as ResNet_segment_book_page
from networks.resnet import ResNet40V2_segment_text_line as ResNet_segment_text_line
from networks.resnet import ResNet40V2_segment_mix_line as ResNet_segment_mix_line
from networks.resnet import ResNet40V2_segment_double_line as ResNet_segment_double_line
from networks.resnext import ReNext64_segment_book_page as ReNext_segment_book_page
from networks.resnext import ReNext40_segment_text_line as ReNext_segment_text_line
from networks.resnext import ReNext40_segment_mix_line as ReNext_segment_mix_line
from networks.resnext import ReNext40_segment_double_line as ReNext_segment_double_line
from networks.densenet import DenseNet60_segment_book_page as DenseNet_segment_book_page
from networks.densenet import DenseNet26_segment_text_line as DenseNet_segment_text_line
from networks.densenet import DenseNet36_segment_mix_line as DenseNet_segment_mix_line
from networks.densenet import DenseNet26_segment_double_line as DenseNet_segment_double_line

from .data_pipeline import image_preprocess_tf
from .gen_target import SegmentTarget
from .losses import interval_cls_loss, split_line_regress_loss
from .gen_prediction import ExtractSplitPosition
from .visualize import images_to_summary_tf
from .utils import get_segment_task_params, get_segment_task_thresh

from config import INIT_LEARNING_RATE, LABEL_SMOOTHING
from config import SGD_LEARNING_MOMENTUM, SGD_GRADIENT_CLIP_NORM
from config import SEGMENT_LOSS_WEIGHTS, L2_WEIGHT_DECAY
from config import SEGMENT_LINE_WEIGHTS


def CNN(segment_task, cnn_type="densenet"):
    task_options = ["book_page", "text_line", "mix_line", "double_line"]
    type_options = ["resnet", "resnext", "densenet"]
    networks = [[ResNet_segment_book_page, ReNext_segment_book_page, DenseNet_segment_book_page],
                [ResNet_segment_text_line, ReNext_segment_text_line, DenseNet_segment_text_line],
                [ResNet_segment_mix_line, ReNext_segment_mix_line, DenseNet_segment_mix_line],
                [ResNet_segment_double_line, ReNext_segment_double_line, DenseNet_segment_double_line]]
    
    i = task_options.index(segment_task)
    j = type_options.index(cnn_type)
    return networks[i][j]


def Bidirectional_RNN(inputs, rnn_units=256, rnn_type="gru"):
    type_options = ["lstm", "gru"]
    networks = [layers.LSTM, layers.GRU]
    
    i = type_options.index(rnn_type)
    rnn_layer = networks[i]
    
    with K.name_scope("bidirectional_" + rnn_type):
        # Based on available runtime hardware, this layer will choose different
        # implementations (pure-tf or cudnn-based).
        outputs = layers.Bidirectional(
                rnn_layer(units=rnn_units, dropout=0.2, return_sequences=True, kernel_initializer='he_normal'),
                merge_mode="concat")(inputs)
    
    return outputs
    

def build_crnn(batch_size, fixed_h, feat_stride=16, segment_task="book_page", model_struc="densenet_gru"):
    batch_images = layers.Input(batch_shape=[batch_size, fixed_h, None, 3], name='batch_images')
    real_images_width = layers.Input(batch_shape=[batch_size], name="real_images_width")
    split_lines_pos = layers.Input(batch_shape=[batch_size, None, 2], name='split_lines_pos')
    # 要求划分位置是有序的，从小到大; padding value -1
    
    # ******************** Build *********************
    convert_imgs = layers.Lambda(image_preprocess_tf, name="image_preprocess")(batch_images)   # image normalization
    
    cnn_type, rnn_type = model_struc.split("_")[:2]
    features = CNN(segment_task, cnn_type)(convert_imgs, feat_stride, scope=cnn_type)

    x = layers.Conv2D(256, (fixed_h//feat_stride, 1), use_bias=True, name="conv_into_seq")(features)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="bn_after_conversion")(x)
    x = layers.Activation('relu', name="relu_after_conversion")(x)
    x = layers.Lambda(lambda x: K.squeeze(x, axis=1), name="feature_squeeze")(x)
    
    x = Bidirectional_RNN(x, rnn_units=256, rnn_type=rnn_type)  # output channels 2*256

    # conv实现fc
    x = layers.Conv1D(256, kernel_size=1, name='fc_output')(x)
    x = layers.BatchNormalization(axis=2, epsilon=1.001e-5, name="fc_bn")(x)
    x = layers.Activation('relu', name="fc_relu")(x)
    
    # 分类及分割线坐标回归
    pred_val = layers.Conv1D(3, kernel_size=1, name='pred_val')(x)    # cls, dx1, dx2: 类别, 顶部偏移, 底部偏移
    pred_cls_logit, pred_delta = pred_val[..., 0], pred_val[..., 1:3]
    
    crnn_model = models.Model(inputs=[batch_images, real_images_width, split_lines_pos],
                              outputs=[pred_cls_logit, pred_delta])
    
    return crnn_model


def work_net(stage="train", segment_task="book_page", text_type="horizontal", model_struc="densenet_gru"):
    batch_size, fixed_h, feat_stride = get_segment_task_params(segment_task)
    cls_score_thresh, distance_thresh, nms_max_outputs = get_segment_task_thresh(segment_task)
    if stage != "train": batch_size = 1
    
    crnn_model = build_crnn(batch_size, fixed_h, feat_stride, segment_task, model_struc)

    batch_images, real_images_width, split_lines_pos = crnn_model.inputs
    pred_cls_logit, pred_delta = crnn_model.outputs
    
    # ******************** Train model **********************
    feat_width = layers.Lambda(lambda x: K.shape(x)[2]//feat_stride)(batch_images)
    real_features_width = layers.Lambda(lambda x: x/feat_stride)(real_images_width)
    targets = SegmentTarget(feat_stride=feat_stride,
                            label_smoothing=LABEL_SMOOTHING,
                            pos_weight=SEGMENT_LINE_WEIGHTS["split_line"],
                            neg_weight=SEGMENT_LINE_WEIGHTS["other_space"],
                            pad_weight=SEGMENT_LINE_WEIGHTS["pad_space"],
                            cls_score_thresh=cls_score_thresh,
                            name='segment_target'
                            )([split_lines_pos, feat_width, real_features_width, pred_cls_logit])

    interval_cls_goals, split_line_delta, interval_mask, inside_weights = targets[:4]
    
    # 损失函数
    class_loss = layers.Lambda(lambda x: interval_cls_loss(*x),
                               name='segment_class_loss')([interval_cls_goals, pred_cls_logit, inside_weights])
    regress_loss = layers.Lambda(lambda x: split_line_regress_loss(*x),
                                 name='segment_regress_loss')([split_line_delta, pred_delta, interval_mask])
    
    # ******************** Predict model *********************
    img_width = layers.Lambda(lambda x: K.shape(x)[2])(batch_images)
    split_positions, scores = ExtractSplitPosition(feat_stride=feat_stride,
                                                   cls_score_thresh=cls_score_thresh,
                                                   distance_thresh=distance_thresh,
                                                   nms_max_outputs=nms_max_outputs,
                                                   name="extract_split_positions"
                                                   )([pred_cls_logit, pred_delta, img_width])
    
    # *********************** Summary *************************
    total_acc, pos_acc, neg_acc = layers.Lambda(compute_acc, name="accuracy")([pred_cls_logit, interval_mask])
    summary_img = layers.Lambda(images_to_summary_tf,
                                arguments={"segment_task": segment_task,
                                           "text_type": text_type},
                                name="image_summary")([batch_images, split_positions, scores])
    
    # ********************* Define model **********************
    if stage == 'train':
        train_model = models.Model(inputs=crnn_model.inputs, outputs=[class_loss, regress_loss, total_acc])
        summary_model = models.Model(inputs=crnn_model.inputs, outputs=[summary_img])  # 孪生网络
        return train_model, summary_model
    else:
        return models.Model(inputs=[batch_images, real_images_width], outputs=[split_positions, scores])


def compute_acc(inputs):
    pred_cls_logit, real_cls_ids = inputs
    pred_cls_ids = tf.cast(pred_cls_logit > 0, tf.float32)
    
    total_pred_result = tf.cast(pred_cls_ids == real_cls_ids, tf.float32)
    pos_pred_result = tf.where(real_cls_ids == 1., total_pred_result, 0)
    neg_pred_result = tf.where(real_cls_ids == 0., total_pred_result, 0)
    
    total_accuracy = tf.reduce_mean(total_pred_result)
    pos_accuracy = tf.reduce_sum(pos_pred_result) / tf.reduce_sum(real_cls_ids)
    neg_accuracy = tf.reduce_sum(neg_pred_result) / tf.reduce_sum(1 - real_cls_ids)
    
    return total_accuracy, pos_accuracy, neg_accuracy


def compile(keras_model, loss_names=[]):
    """编译模型，添加损失函数，L2正则化"""
    
    # 优化器
    # optimizer = optimizers.SGD(INIT_LEARNING_RATE, momentum=SGD_LEARNING_MOMENTUM, clipnorm=SGD_GRADIENT_CLIP_NORM)
    # optimizer = optimizers.RMSprop(learning_rate=INIT_LEARNING_RATE, rho=0.9)
    optimizer = optimizers.Adagrad(learning_rate=INIT_LEARNING_RATE)
    # optimizer = optimizers.Adadelta(learning_rate=1., rho=0.95)
    # optimizer = optimizers.Adam(learning_rate=INIT_LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # 添加损失函数，首先清除损失，防止重复计算
    # keras_model._losses = []
    # keras_model._per_input_losses = {}
    
    for loss_name in loss_names:
        loss_layer = keras_model.get_layer(loss_name)
        if loss_layer is None: continue
        loss = loss_layer.output * SEGMENT_LOSS_WEIGHTS.get(loss_name, 1.)
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
        metric_val = metric_val * SEGMENT_LOSS_WEIGHTS.get(metric_name, 1.)
        keras_model.metrics_names.append(metric_name)
        keras_model.add_metric(metric_val, name=metric_name, aggregation='mean')
