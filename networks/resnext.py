"""Keras model: ResNeXt

# Reference papers

- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
"""

from tensorflow.keras import layers, models, backend
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


def ResNeXt151_for_crnn(inputs):
    def stack_fn(x):
        x = stack3(x, 64, 4, stride1=1, name='conv2')
        x = stack3(x, 128, 12, name='conv3')
        x = stack3(x, 256, 28, name='conv4')
        x = stack3(x, 512, 6, name='conv5')
        return x
    
    return ResNet(inputs=inputs,
                   stack_fn=stack_fn,
                   use_bias=False,
                   block_preact=False,
                   model_name='resnext')  # size 1/8
