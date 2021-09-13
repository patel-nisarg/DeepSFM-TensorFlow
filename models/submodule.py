from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, ReLU, ZeroPadding3D, ZeroPadding2D, AveragePooling2D


# def pad_tensor(tensor, padding, mode='2d'):
#     if mode == '2d':
#         if padding > 0:
#             paddings = tf.constant([[padding] * 2] * 2)
#             return tf.pad(tensor, paddings, "CONSTANT")
#         else:
#             raise ValueError("Padding must be >= 0!")

#     elif mode == '3d':
#         if padding > 0:
#             paddings = tf.constant([[padding] * 2] * 3)
#             return tf.pad(tensor, paddings, "CONSTANT")
#         else:
#             raise ValueError("Padding must be >= 0!")

#     else:
#         raise TypeError("Padding mode can only be '2d' or '3d'!")


def convbn(filters, kernel_size, stride, pad, dilation=1):
    # 2D convolution layer followed by batch normalization
    return Sequential([ZeroPadding2D(dilation if dilation > 1 else pad),
                       Conv2D(
                        filters, 
                        kernel_size, 
                        stride, 
                        'valid',
                        dilation_rate=dilation),
                       BatchNormalization()])


def convbn_3d_o(filters, kernel_size, stride, pad):
    # 3D convolution layer followed by batch normalization
    return Sequential([
        ZeroPadding3D(pad),
        Conv3D(filters, kernel_size, stride, padding='valid'),
        BatchNormalization()
    ])


def convbn_3d(filters, kernel_size, stride, pad):
    # 3D conv layer followed by group normalization layer
    # group normalization paper: https://arxiv.org/pdf/1803.08494.pdf
    # example implementation for tfa: https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization

    return Sequential([
        ZeroPadding3D(pad),
        Conv3D(filters, kernel_size, stride, padding='valid'),
        tfa.layers.GroupNormalization(groups=8)
    ])


def conv_3d(filters, kernel_size, stride, pad):
    # 3D conv layer
    return Sequential([ZeroPadding3D(pad),
                       Conv3D(filters, kernel_size, stride, padding='valid')])


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    # residual block
    def __init__(self, filters, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = Sequential([convbn(filters, 3, stride, pad, dilation),
                                ReLU()])

        self.conv2 = convbn(filters, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x1 = x.shape
        if self.downsample is not None:
            x = self.downsample(x)

        # identity addition of res block
        out += x

        return out


class disparityregression(tf.keras.Model):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = tf.Variable(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))

    def __call__(self, x):
        disp = tf.tile(self.disp, [x.shape[0], 1, x.shape[2], x.shape[3]])
        out = tf.reduce_sum(x * disp, axis=1)
        return out


class feature_extraction(tf.keras.Model):
    def __init__(self, pool=False):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = Sequential([
            convbn(32, (3, 3), 2, 1, 1),
            ReLU(),
            convbn(32, (3, 3), 1, 1, 1),
            ReLU(),
            convbn(32, (3, 3), 1, 1, 1),
            ReLU()
        ])

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = Sequential([
                                AveragePooling2D(pool_size=(32, 32), strides=32),
                                convbn(32, 1, 1, 0, 1),
                                ReLU()
        ])

        self.branch2 = Sequential([
                                AveragePooling2D(pool_size=(16, 16), strides=16),
                                convbn(32, 1, 1, 0, 1),
                                ReLU()
        ])

        self.branch3 = Sequential([
                                AveragePooling2D(pool_size=(8, 8), strides=8),
                                convbn(32, 1, 1, 0, 1),
                                ReLU()
        ])

        self.branch4 = Sequential([
                                AveragePooling2D(pool_size=(4, 4), strides=4),
                                convbn(32, 1, 1, 0, 1),
                                ReLU()
        ])

        if pool:
            self.lastconv = Sequential([
                                        convbn(128, 3, 1, 1, 1),
                                        ReLU(),
                                        Conv2D(32, 1, padding='valid'),
                                        AveragePooling2D((2, 2), strides=(2, 2))
            ])
        else:
            self.lastconv = Sequential([
                                        convbn(128, 3, 1, 1, 1),
                                        ReLU(),
                                        Conv2D(32, 1, padding='valid')
            ])


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(planes * block.expansion, kernel_size=(1, 1), strides=stride, use_bias=False),
                BatchNormalization()
            ])

        layers = []
        layers.append(block(planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, 1, None, pad, dilation))
        return Sequential(layers)


    def __call__(self, x):
        # x.shape is [B, 240, 320, 3]
        output = self.firstconv(x)
        output = self.layer1(output) # (B, 120, 160, 32)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        
        output_branch1 = self.branch1(output_skip)
        output_branch1 = tf.image.resize(output_branch1, (output_skip.shape[1], output_skip.shape[2])) # default interpolation method is bilinear

        output_branch2 = self.branch2(output_skip)
        output_branch2 = tf.image.resize(output_branch2, (output_skip.shape[1], output_skip.shape[2]))

        output_branch3 = self.branch3(output_skip)
        output_branch3 = tf.image.resize(output_branch3, (output_skip.shape[1], output_skip.shape[2]))

        output_branch4 = self.branch4(output_skip)
        output_branch4 = tf.image.resize(output_branch4, (output_skip.shape[1], output_skip.shape[2]))
        # output shapes of below concat list: (2, 60, 80, 64) (2, 60, 80, 128) (2, 60, 80, 32) (2, 60, 80, 32) (2, 60, 80, 32) (2, 60, 80, 32)        
        output_feature = tf.concat([output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1], 3)
        output_feature = self.lastconv(output_feature)

        return output_feature
        