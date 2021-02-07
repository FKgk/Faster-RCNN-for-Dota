# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed, \
                                        UpSampling2D, Layer, Flatten, Dense, Input, Conv2D, \
                                        MaxPooling2D, BatchNormalization, Concatenate

from .roi_pooling_conv import RoiPoolingConv
from .attention_module import attach_attention_module
# from tensorflow.keras.utils import get_source_inputs
# from tensorflow.keras.utils import layer_utils
# from tensorflow.keras.utils.data_utils import get_file

out_rpn_strides = 8

def get_weight_path():
    return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return int(input_length // out_rpn_strides)

    return get_output_length(width), get_output_length(height)

class FusionConcatenate(Layer):
    def call(self, xlist):
        x1, x2, x3 = xlist
        _b, h1, w1, _c = x1.shape
        _b, h2, w2, _c = x2.shape
        _b, h3, w3, _c = x3.shape

        x3 = tf.pad(x3, ((0, abs(h3 - h2)), (0, abs(w3 - w2)), (0, 0)), "CONSTANT", constant_values=0)

        out = Concatenate()([x1,x2,x3])

        return out


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x1 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x) # 5 X 5 
    x2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) # 2 X 2
    # 2 ** 4

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x3 = UpSampling2D((2,2), interpolation='bilinear')(x) # 4 X 4
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)
    x3 = BatchNormalization()(x3)

    x = Concatenate()([x1,x2,x3])
    x = attach_attention_module(x)

    return x


def rpn(base_layers, num_anchors):
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x) # num_anchors * 2, softmax
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
