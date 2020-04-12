"""Keras models: ResNet, ResNetV2

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
"""

from tensorflow.keras import layers, backend


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block."""
    
    bn_axis = 3  # image data format: channels_last

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks."""
    
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.(version 2)"""
    
    bn_axis = 3  # image data format: channels_last

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.(version 2)"""
    
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def ResNet(x,
           stack_fn,
           use_bias,
           block_preact,
           include_top=False,
           classes=None,
           pooling=None):
    """Instantiates the ResNet, ResNetV2 and ResNeXt architecture."""

    bn_axis = 3  # image data format: channels_last

    x = layers.ZeroPadding2D(padding=((2, 2), (2, 2)), name='conv1_pad')(x)
    x = layers.Conv2D(64, 5, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if block_preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    # x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    x = stack_fn(x)

    if block_preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    return x


def ResNet58V2_for_crnn(inputs, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')
        x = stack2(x, 256, 8, stride1=1, name='conv4')
        # x = stack2(x, 256, 28, name='conv4')
        # x = stack2(x, 512, 6, stride1=1, name='conv5')
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)  # 1/8 size
    
    return outputs


def ResNet58V2_for_ctpn(inputs, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')  # 1/8 size
        x = stack2(x, 256, 8, name='conv4')  # 1/16 size
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)  # 1/16 size
    
    return outputs


def ResNet76V2_for_yolo(inputs, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x1 = stack2(x, 128, 8, name='conv3')   # 1/8 size
        x2 = stack2(x1, 256, 8, name='conv4')  # 1/16 size
        x3 = stack2(x2, 512, 6, name='conv5')  # 1/32 size
        return [x1, x2, x3]
    
    with backend.name_scope(scope):
        x_list = ResNet(inputs, stack_fn, use_bias=True, block_preact=None)

        bn_axis = 3  # image data format: channels_last
        features_list = []
        for i, x in enumerate(x_list):
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name='post_bn_%d'%i)(x)
            feat = layers.Activation('relu', name='post_relu_%d'%i)(x)
            features_list.append(feat)
    
    return features_list


def ResNet64V2_segment_book_page(inputs, feat_stride=16, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')  # 1/8 size
        x = stack2(x, 256, 8, name='conv4')  # 1/16 size
        x = stack2(x, 512, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)
    
    return outputs


def ResNet40V2_segment_text_line(inputs, feat_stride=16, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')  # 1/8 size
        x = stack2(x, 256, 4, name='conv4')  # 1/16 size
        x = stack2(x, 512, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)
    
    return outputs


def ResNet40V2_segment_mix_line(inputs, feat_stride=16, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')  # 1/8 size
        x = stack2(x, 256, 4, name='conv4')  # 1/16 size
        x = stack2(x, 512, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)
    
    return outputs


def ResNet40V2_segment_double_line(inputs, feat_stride=16, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')  # 1/8 size
        x = stack2(x, 256, 4, name='conv4')  # 1/16 size
        x = stack2(x, 512, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)
    
    return outputs


def ResNet58V2_for_char_recog(inputs, scope="resnet"):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')  # 1/8 size
        x = stack2(x, 256, 8, name='conv4')  # 1/16 size
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=True)  # 1/16 size
    
    return outputs
