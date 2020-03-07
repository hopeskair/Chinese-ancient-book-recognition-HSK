# -*- encoding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import backend, layers, models

from networks.resnet import ResNet58V2_for_crnn as ResNet_for_crnn      # size to 1/8
from networks.resnext import ResNeXt58_for_crnn as ResNeXt_for_crnn     # size to 1/8
from networks.densenet import DenseNet53_for_crnn as DenseNet_for_crnn  # size to 1/8

from utils import NUM_CHARS
from config import TEXT_LINE_SIZE


class CRNN(object):
    def __init__(self, model_type="horizontal", model_struc="resnet_lstm"):
        self.classes = NUM_CHARS         # blank label is 0.
        self.model_type = model_type     # h, horizontal, v, vertical
        self.model_struc = model_struc   # resnet/resnext/densenet, lstm/gru
    
    def crnn(self):
        
        inputs = layers.Input(shape=[None, None, 1], dtype="float32")
        
        features = CNN(inputs, scope=self.model_struc.split("_")[0])  # 1/8 size
        
        # Model for horizontal text or vertical text
        if self.model_type in ("v", "vertical"):
            features = backend.permute_dimensions(features, (0, 2, 1, 3))
        else:
            assert self.model_type in ("h", "horizontal")
        
        x = layers.Conv2D(filters=1024, kernel_size=(TEXT_LINE_SIZE//8, 1), name="features_conv")(features)
        x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="features_bn")(x)
        x = layers.Activation('relu', name="features_relu")(x)
        
        x = backend.squeeze(x, axis=1)
        x = Bidirectional_RNN(x, scope=self.model_struc.split("_")[1])
        
        logits = layers.Dense(self.classes, name="logits")(x)
        
        return models.Model(inputs=[inputs], outputs=[logits], name="crnn_body")
        
    def ctc_loss(self, args_list):
        logits, labels, logit_len, label_len = args_list
        
        # Permute dimensions to compute ctc loss
        logits = backend.permute_dimensions(logits, (1, 0, 2))

        # Loss and cost calculation
        # Default blank label is 0 rather num_classes-1
        loss = tf.nn.ctc_loss(labels=labels,
                              logits=logits,
                              label_length=label_len,
                              logit_length=logit_len)
        
        loss = backend.mean(loss, axis=0, keepdims=True)
        
        return loss
        
    def ctc_decode(self, args_list, greedy = False, beam_size = 5):
        logits, logit_len = args_list
        
        # Permute dimensions for ctc decoding
        logits = backend.permute_dimensions(logits, (1, 0, 2))
    
        if greedy:
            decoded, _neg_sum_logits  = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=logit_len)
        else:
            decoded, _log_probability = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                      sequence_length=logit_len,
                                                                      beam_width=beam_size,
                                                                      top_paths=1)
        
        decoded = decoded[0]
        return decoded.indices, decoded.values
    
    def model_for_training(self):
        
        crnn_body = self.crnn()

        batch_imgs = crnn_body.inputs[0]
        logits = crnn_body.outputs[0]
        
        batch_labels = layers.Input(shape=[None], dtype="int32")
        label_len = layers.Input(shape=[], dtype="int32")
        img_len = layers.Input(shape=[], dtype="int32")
        
        logit_len = backend.cast(backend.round(img_len/8), dtype="int32")
        
        out_loss = layers.Lambda(self.ctc_loss,
                                 output_shape=[],
                                 name='ctc_loss')([logits, batch_labels, logit_len, label_len])
        crnn_model = models.Model(inputs=[batch_imgs, img_len, batch_labels, label_len],
                                  outputs=[out_loss], name="crnn_model")
        
        return crnn_model

    def model_for_predicting(self):
        
        crnn_body = self.crnn()

        batch_imgs = crnn_body.inputs[0]
        logits = crnn_body.outputs[0]

        img_len = layers.Input(shape=[], dtype="int32")

        logit_len = backend.cast(backend.round(img_len / 8), dtype="int32")

        sp_indices, sp_values = layers.Lambda(self.ctc_decode, name="ctc_decoding")([logits, logit_len])
        
        
        crnn_model = models.Model(inputs=[batch_imgs, img_len],
                                  outputs=[sp_indices, sp_values], name="crnn_model")
        
        return crnn_model
        

def CNN(inputs, scope="densenet"):
    """cnn of crnn."""
    
    if "resnet" in scope:
        outputs = ResNet_for_crnn(inputs, scope)  # 1/8 size
    elif "resnext" in scope:
        outputs = ResNeXt_for_crnn(inputs, scope)   # 1/8 size
    elif "densenet" in scope:
        outputs = DenseNet_for_crnn(inputs, scope)  # 1/8 size
    else:
        ValueError("Optional CNN scope: 'resnet*', 'resnext*', 'densenet*'.")
    
    return outputs


def Bidirectional_RNN(inputs, scope="gru"):
    """Bidirectional RNN of crnn."""
    
    if "lstm" in scope:
        rnn_layer = layers.LSTM
    elif "gru" in scope:
        rnn_layer = layers.GRU
    else:
        ValueError("Optional RNN layer: '*lstm', '*gru'.")
    
    with backend.name_scope(scope):
        # Based on available runtime hardware, this layer will choose different
        # implementations (pure-tf or cudnn-based).
        outputs = layers.Bidirectional(rnn_layer(units=1024, dropout=0.2, return_sequences=True),
                                       merge_mode="concat",
                                       )(inputs)
    
    return outputs