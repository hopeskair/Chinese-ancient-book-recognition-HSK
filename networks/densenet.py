"""Keras model: DenseNet

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""

from tensorflow.keras import layers, backend


def conv_block(x, growth, name):
    """A building block for a dense block."""
    
    bn_axis = 3  # image data format: channels_last
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def dense_block(x, blocks, name):
    """A dense block."""
    
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction_rate, name):
    """A transition block."""
    
    bn_axis = 3  # image data format: channels_last
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction_rate), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def DenseNet(x,
             blocks,
             include_top=False,
             classes=None,
             pooling=None):
    """Instantiates the DenseNet architecture."""

    bn_axis = 3  # image data format: channels_last
    
    x = layers.ZeroPadding2D(padding=((2, 2), (2, 2)))(x)
    x = layers.Conv2D(64, 5, strides=2, use_bias=False, name='conv1/conv')(x)
    # x = layers.BatchNormalization(
    #     axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    # x = layers.Activation('relu', name='conv1/relu')(x)
    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    # x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)
    
    for i in range(len(blocks)-1):
        x = dense_block(x, blocks[i], name='conv%d'%(i+2))
        x = transition_block(x, 0.5, name='pool%d'%(i+2))
    x = dense_block(x, blocks[-1], name='conv%d'%(len(blocks)-1+2))

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    
    return x


def DenseNet53_for_crnn(inputs, scope="densenet"):
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=[3, 12, 10])  # 1/8 size
    return outputs


def DenseNet60_for_ctpn(inputs, scope="densenet"):
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=[3, 9, 8, 8])  # 1/16 size
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs


def DenseNet73_for_yolo(inputs, scope="densenet"):
    blocks = [3, 9, 8, 8, 6]
    
    with backend.name_scope(scope):
        x = layers.ZeroPadding2D(padding=((2, 2), (2, 2)))(inputs)
        x = layers.Conv2D(64, 5, strides=2, use_bias=False, name='conv1/conv')(x)
        
        x_list = []
        for i in range(len(blocks) - 1):
            x = dense_block(x, blocks[i], name='conv%d' % (i + 2))
            x_list.append(x)
            x = transition_block(x, 0.5, name='pool%d' % (i + 2))
        x = dense_block(x, blocks[-1], name='conv%d' % (len(blocks) - 1 + 2))
        x_list.append(x)
        
        bn_axis = 3  # image data format: channels_last
        features_list = []
        for i, x in enumerate(x_list[-3:]):
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name='post_bn_%d'%i)(x)
            feat = layers.Activation('relu', name='post_relu_%d'%i)(x)
            features_list.append(feat)
        
    return features_list


def DenseNet60_segment_book_page(inputs, feat_stride=16, scope="densenet"): # 60 or 65
    blocks = [3, 9, 8, 8] if feat_stride == 16 else [3, 9, 8, 8, 2]
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=blocks)  # 1/16 or 1/32
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs


def DenseNet26_segment_text_line(inputs, feat_stride=16, scope="densenet"): # 36 or 39
    blocks = [2, 3, 3, 3] if feat_stride == 16 else [2, 3, 3, 3, 2]
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=blocks)  # 1/16 or 1/32
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs


def DenseNet36_segment_mix_line(inputs, feat_stride=16, scope="densenet"): # 36 or 39
    blocks = [3, 5, 4, 4] if feat_stride == 16 else [3, 4, 4, 4, 2]
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=blocks)  # 1/16 or 1/32
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs


def DenseNet26_segment_double_line(inputs, feat_stride=16, scope="densenet"): # 36 or 39
    blocks = [2, 3, 3, 3] if feat_stride == 16 else [2, 3, 3, 3, 2]
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=blocks)  # 1/16 or 1/32
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs


def DenseNet60_for_char_recog(inputs, scope="densenet"):
    with backend.name_scope(scope):
        outputs = DenseNet(inputs, blocks=[3, 9, 8, 8])  # 1/16 size
        # outputs = DenseNet(inputs, blocks=[1, 1, 1, 1])  # for test
    return outputs
