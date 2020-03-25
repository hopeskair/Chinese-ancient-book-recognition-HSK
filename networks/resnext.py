"""Keras model: ResNeXt

# Reference papers

- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
"""

from tensorflow.keras import layers, backend
from networks.resnet import ResNet


def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block of ResNeXt."""
    
    bn_axis = 3  # image data format: channels_last
    
    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
    
    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    
    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    x_shape = backend.int_shape(x)[1:-1]
    x = layers.Reshape(x_shape + [groups, c, c])(x)
    x = layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]),
                      name=name + '_2_reduce')(x)  # shape [b, h, w, groups, c]
    x = layers.Reshape(x_shape + [filters, ])(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    
    x = layers.Conv2D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)
    
    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks for ResNeXt."""
    
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x


def ResNeXt58_for_crnn(inputs, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 8, name='conv3')
        x = stack3(x, 512, 8, name='conv4')
        # x = stack3(x, 1024, 6, name='conv5')
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=False, block_preact=False)  # 1/8 size
    
    return outputs


def ResNeXt58_for_ctpn(inputs, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x = stack3(x, 256, 8, name='conv3')  # 1/8 size
        x = stack3(x, 512, 8, name='conv4')  # 1/16 size
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=False, block_preact=False)  # 1/16 size
    
    return outputs


def ResNeXt76_for_yolo(inputs, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x1 = stack3(x, 256, 8, name='conv3')    # 1/8 size
        x2 = stack3(x1, 512, 8, name='conv4')   # 1/16 size
        x3 = stack3(x2, 1024, 6, name='conv5')  # 1/32 size
        return [x1, x2, x3]
    
    with backend.name_scope(scope):
        features_list = ResNet(inputs, stack_fn, use_bias=False, block_preact=False)
    
    return features_list


def ReNext64_segment_book_page(inputs, feat_stride=16, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x = stack3(x, 256, 8, name='conv3')  # 1/8 size
        x = stack3(x, 512, 8, name='conv4')  # 1/16 size
        x = stack3(x, 1024, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=False)
    
    return outputs


def ReNext40_segment_text_line(inputs, feat_stride=16, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x = stack3(x, 256, 4, name='conv3')  # 1/8 size
        x = stack3(x, 512, 4, name='conv4')  # 1/16 size
        x = stack3(x, 1024, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=False)
    
    return outputs


def ReNext40_segment_mix_line(inputs, feat_stride=16, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x = stack3(x, 256, 4, name='conv3')  # 1/8 size
        x = stack3(x, 512, 4, name='conv4')  # 1/16 size
        x = stack3(x, 1024, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=False)
    
    return outputs


def ReNext40_segment_double_line(inputs, feat_stride=16, scope="resnext"):
    def stack_fn(x):
        x = stack3(x, 128, 3, name='conv2')
        x = stack3(x, 256, 4, name='conv3')  # 1/8 size
        x = stack3(x, 512, 4, name='conv4')  # 1/16 size
        x = stack3(x, 1024, 2, stride1=feat_stride//16, name='conv5')  # 1/16 or 1/32
        return x
    
    with backend.name_scope(scope):
        outputs = ResNet(inputs, stack_fn, use_bias=True, block_preact=False)
    
    return outputs