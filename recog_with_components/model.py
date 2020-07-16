# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers, regularizers

from networks.resnet import ResNet58V2_for_char_recog as ResNet_for_char_recog
from networks.resnext import ResNeXt58_for_char_recog as ResNeXt_for_char_recog
from networks.densenet import DenseNet35_for_char_recog as DenseNet_for_char_recog

from .data_pipeline import image_preprocess_tf
from .losses import char_struc_loss, sc_char_loss, lr_compo_loss
from .gen_prediction import GeneratePrediction

from util import NUM_CHAR_STRUC, NUM_SIMPLE_CHAR, NUM_LR_COMPO
from config import CHAR_RECOG_FEAT_STRIDE, COMPO_SEQ_LENGTH
from config import LABEL_SMOOTHING
from config import TOP_K_TO_PRED
from config import INIT_LEARNING_RATE, CHAE_RECOG_LOSS_WEIGHTS, L2_WEIGHT_DECAY
from config import SGD_LEARNING_MOMENTUM, SGD_GRADIENT_CLIP_NORM


def CNN(inputs, feat_stride=16, scope="densenet"):
    if scope == "resnet":
        outputs = ResNet_for_char_recog(inputs, feat_stride, scope)     # 1/16 size
    elif scope == "resnext":
        outputs = ResNeXt_for_char_recog(inputs, feat_stride, scope)    # 1/16 size
    elif scope == "densenet":
        outputs = DenseNet_for_char_recog(inputs, feat_stride, scope)   # 1/16 size
    else:
        ValueError("Optional CNN scope: 'resnet', 'resnext', 'densenet'.")
    
    return outputs


def Bidirectional_RNN(inputs, rnn_units=64, rnn_type="gru"):
    type_options = ["lstm", "gru"]
    networks = [layers.LSTM, layers.GRU]
    
    i = type_options.index(rnn_type.split("_")[0])
    rnn_layer = networks[i]
    
    with K.name_scope("bidirectional_" + rnn_type):
        # Based on available runtime hardware, this layer will choose different
        # implementations (pure-tf or cudnn-based).
        outputs = layers.Bidirectional(
            rnn_layer(units=rnn_units, dropout=0.2, return_sequences=True, kernel_initializer='he_normal'),
            merge_mode="concat")(inputs)
    
    return outputs


def data_branch_tf(inputs):
    features, char_struc = inputs
    
    sc_indices = tf.where(char_struc == 0)[:, 0]
    lr_indices = tf.where(char_struc == 1)[:, 0]
    # ul_indices = tf.where(char_struc_used == 2)[:, 0]
    
    sc_features = tf.gather(features, indices=sc_indices)
    lr_features = tf.gather(features, indices=lr_indices)
    # ul_features = tf.gather(features, indices=ul_indices)
    
    return sc_features, lr_features


def label_branch_tf(inputs):
    components_seq, char_struc = inputs
    
    sc_indices = tf.where(char_struc == 0)[:, 0]
    lr_indices = tf.where(char_struc == 1)[:, 0]
    # ul_indices = tf.where(char_struc_used == 2)[:, 0]
    
    sc_labels = tf.gather(components_seq, indices=sc_indices)[:, 0]
    lr_compo_seq = tf.gather(components_seq, indices=lr_indices)
    # ul_compo_seq = tf.gather(components_seq, indices=ul_indices)
    
    return sc_labels, lr_compo_seq


def build_model(stage="predict", img_size=64, model_struc="densenet_gru"):
    
    batch_images = layers.Input(shape=[img_size, img_size, 3], name='batch_images')
    char_struc = layers.Input(shape=[], dtype=tf.int32, name='char_struc')
    components_seq = layers.Input(shape=[COMPO_SEQ_LENGTH], dtype=tf.int32, name='components_seq')
    
    # ******************* Backbone ********************
    # image normalization
    convert_imgs = layers.Lambda(image_preprocess_tf, arguments={"stage": stage}, name="image_preprocess")(batch_images)
    
    cnn_type, rnn_type = model_struc.split("_")[:2]
    features = CNN(convert_imgs, feat_stride=CHAR_RECOG_FEAT_STRIDE, scope=cnn_type)    # 1/16 size
    feat_size = img_size // CHAR_RECOG_FEAT_STRIDE  # 4

    # ***************** 汉字结构预测 ******************
    x_struc = layers.Conv2D(16, 3, padding="same", name="struc_conv")(features)
    x_struc = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="struc_conv_bn")(x_struc)
    x_struc = layers.Activation('relu', name="struc_conv_relu")(x_struc)
    
    pred_struc_logits = layers.Conv2D(NUM_CHAR_STRUC, kernel_size=feat_size, name="pred_struc_logits")(x_struc)
    pred_struc_logits = tf.squeeze(pred_struc_logits, axis=[1, 2], name="pred_struc_squeeze")
    
    # ******************* 模型分支 ********************
    _, pred_char_struc = tf.math.top_k(pred_struc_logits, k=1, name="pred_char_struc")
    pred_char_struc = pred_char_struc[:, 0]
    
    # teacher-forcing
    char_struc_used = char_struc if stage == "train" else pred_char_struc
    sc_features, lr_features = layers.Lambda(data_branch_tf, name="data_branch")([features, char_struc_used])
    
    sc_labels, lr_compo_seq = layers.Lambda(label_branch_tf, name="label_branch")([components_seq, char_struc])
    
    # ***************** 简单汉字预测 ******************
    x_sc = layers.Conv2D(16, 3, padding="same", name="sc_conv")(sc_features)
    x_sc = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="sc_conv_bn")(x_sc)
    x_sc = layers.Activation('relu', name="sc_conv_relu")(x_sc)
    
    pred_sc_logits = layers.Conv2D(NUM_SIMPLE_CHAR, kernel_size=feat_size, name='pred_sc_logits')(x_sc)
    pred_sc_logits = tf.squeeze(pred_sc_logits, axis=[1, 2], name="pred_sc_squeeze")
    
    # ************* 左右结构汉字部件预测 ***************
    x_lr = layers.Conv2D(256, (feat_size, 1), name="lr_conv")(lr_features)
    x_lr = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="lr_conv_bn")(x_lr)
    x_lr = layers.Activation('relu', name="lr_conv_relu")(x_lr)
    x_lr = tf.squeeze(x_lr, axis=1, name="x_lr_squeeze")

    rnn_units = 256
    x_lr = Bidirectional_RNN(x_lr, rnn_units=rnn_units//2, rnn_type=rnn_type+"_lr_compo")
    pred_lr_compo_logits = layers.Dense(NUM_LR_COMPO, name="pred_lr_compo_logits")(x_lr)
    
    # ******************** Build *********************
    recog_model = models.Model(inputs=[batch_images, char_struc, components_seq],
                               outputs=[pred_struc_logits, pred_char_struc, sc_labels, lr_compo_seq,
                                        pred_sc_logits, pred_lr_compo_logits])
    
    return recog_model


def work_net(stage="test", img_size=64, model_struc="densenet_gru"):
    
    recog_model = build_model(stage, img_size, model_struc)
    
    batch_images, char_struc, components_seq = recog_model.inputs
    pred_struc_logits, pred_char_struc, sc_labels, lr_compo_seq, \
    pred_sc_logits, pred_lr_compo_logits = recog_model.outputs
    
    # ******************** Train model **********************
    # 损失函数
    options = {"label_smoothing": LABEL_SMOOTHING}
    struc_loss = layers.Lambda(lambda x: char_struc_loss(*x, **options),
                               name='char_struc_loss')([char_struc, pred_struc_logits])
    sc_loss = layers.Lambda(lambda x: sc_char_loss(*x, **options),
                            name='sc_char_loss')([sc_labels, pred_sc_logits])
    lr_loss = layers.Lambda(lambda x: lr_compo_loss(*x, **options),
                            name='lr_compo_loss')([lr_compo_seq, pred_lr_compo_logits])
    
    # ******************** Predict model *********************
    targets = GeneratePrediction(topk=TOP_K_TO_PRED, stage=stage, name="gen_prediction")(
        [pred_char_struc, pred_sc_logits, pred_lr_compo_logits])
    
    pred_sc_labels, pred_lr_compo_seq, pred_results = targets
    
    # *********************** Summary *************************
    metrics_summary = layers.Lambda(summary_fn, name="summary_fn")(
        [char_struc, sc_labels, lr_compo_seq, pred_char_struc, pred_sc_labels, pred_lr_compo_seq])
    
    # ********************* Define model **********************
    if stage == 'train':
        return models.Model(inputs=recog_model.inputs, outputs=[struc_loss, sc_loss, lr_loss, metrics_summary])
    else:
        return models.Model(inputs=batch_images, outputs=[pred_char_struc, pred_results])


def summary_fn(inputs, **kwargs):
    real_char_struc, sc_labels, lr_compo_seq, pred_char_struc, pred_sc_labels, pred_lr_compo_seq = inputs
    
    # char structure prediction
    char_struc_acc = tf.reduce_mean(tf.cast(pred_char_struc == real_char_struc, tf.float32))
    
    # sc prediction
    sc_acc = tf.reduce_mean(tf.cast(pred_sc_labels[:, 0] == sc_labels, tf.float32))
    sc_top3 = tf.reduce_mean(tf.cast(tf.reduce_any(pred_sc_labels[:, :3] == sc_labels[:, tf.newaxis], axis=1), tf.float32))
    sc_top5 = tf.reduce_mean(tf.cast(tf.reduce_any(pred_sc_labels[:, :5] == sc_labels[:, tf.newaxis], axis=1), tf.float32))
    
    # lr prediction
    lr_acc = tf.reduce_mean(tf.cast(tf.reduce_all(pred_lr_compo_seq[:, 0] == lr_compo_seq, axis=1), tf.float32))
    exp_lr_compo_seq = lr_compo_seq[:, tf.newaxis, :]
    lr_top3 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.reduce_all(pred_lr_compo_seq[:, :3] == exp_lr_compo_seq, axis=2), axis=1), tf.float32))
    lr_top5 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.reduce_all(pred_lr_compo_seq[:, :5] == exp_lr_compo_seq, axis=2), axis=1), tf.float32))

    # ul prediction
    
    # The sequence ends early when 0(EOC) appears
    # Correct pred_lr_compo_seq
    # tf.math.top_k: If two elements are equal, the lower-index element appears first.
    _max_neg_values, pos_indices = tf.math.top_k(-pred_lr_compo_seq, k=1)       # find the first 0(EOC) position
    min_values = -_max_neg_values
    seq_len = tf.shape(pred_lr_compo_seq)[2]
    zero_pos = tf.where(min_values == 0, pos_indices, seq_len)                  # (bsize, topk, 1)
    seq_pos = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]      # (1, 1, seq_len)
    correct_lr_compo_seq = tf.where(seq_pos >= zero_pos, 0, pred_lr_compo_seq)  # (bsize, topk, seq_len)
    
    # Correct pred_ul_compo_seq
    
    # lr prediction corrected
    correct_lr_acc = tf.reduce_mean(tf.cast(tf.reduce_all(correct_lr_compo_seq[:, 0] == lr_compo_seq, axis=1), tf.float32))
    correct_lr_top3 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.reduce_all(correct_lr_compo_seq[:, :3] == exp_lr_compo_seq, axis=2), axis=1), tf.float32))
    correct_lr_top5 = tf.reduce_mean(tf.cast(tf.reduce_any(tf.reduce_all(correct_lr_compo_seq[:, :5] == exp_lr_compo_seq, axis=2), axis=1), tf.float32))
    
    # ul prediction corrected
    
    # summary prediction
    sc_num = tf.cast(tf.shape(sc_labels)[0], tf.float32)
    lr_num = tf.cast(tf.shape(lr_compo_seq)[0], tf.float32)
    total_acc = (sc_num * sc_acc + lr_num * lr_acc) / (sc_num + lr_num)
    total_top3 = (sc_num * sc_top3 + lr_num * lr_top3) / (sc_num + lr_num)
    total_top5 = (sc_num * sc_top5 + lr_num * lr_top5) / (sc_num + lr_num)
    
    return char_struc_acc, sc_acc, sc_top3, sc_top5, \
           lr_acc, lr_top3, lr_top5, \
           correct_lr_acc, correct_lr_top3, correct_lr_top5, \
           total_acc, total_top3, total_top5


def compile(keras_model, loss_names=[]):
    """编译模型，添加损失函数，L2正则化"""
    
    # 优化器
    # optimizer = optimizers.SGD(INIT_LEARNING_RATE, momentum=SGD_LEARNING_MOMENTUM, clipnorm=SGD_GRADIENT_CLIP_NORM)
    # optimizer = optimizers.RMSprop(learning_rate=INIT_LEARNING_RATE, rho=0.9)
    # optimizer = optimizers.Adagrad(learning_rate=INIT_LEARNING_RATE)
    # optimizer = optimizers.Adadelta(learning_rate=1., rho=0.95)
    optimizer = optimizers.Adam(learning_rate=INIT_LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
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
