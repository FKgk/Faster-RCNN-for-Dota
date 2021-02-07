import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-8

def concatenate(y, p):
    return K.concatenate((K.expand_dims(y, axis=-1), K.expand_dims(p, axis=-1)), axis=-1)

def cal_S(Y, P):
    return K.abs((Y[0] - Y[1]) * (Y[2] - Y[3])), \
            K.abs((P[0] - P[1]) * (P[2] - P[3]))

def cal_S_I(Y, P): # Concat
    x1 = K.max(concatenate(Y[0], P[0]), axis=-1)
    x2 = K.min(concatenate(Y[1], P[1]), axis=-1)
    y1 = K.max(concatenate(Y[2], P[2]), axis=-1)
    y2 = K.min(concatenate(Y[3], P[3]), axis=-1)

    return K.abs((x2-x1) * (y2-y1))

def cal_S_C(Y, P):
    x1 = K.min(concatenate(Y[0], P[0]), axis=-1)
    x2 = K.max(concatenate(Y[1], P[1]), axis=-1)
    y1 = K.min(concatenate(Y[2], P[2]), axis=-1)
    y2 = K.max(concatenate(Y[3], P[3]), axis=-1)

    return K.abs((x2-x1) * (y2-y1))

def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        batch_size, height, width, c = y_true.shape
        
        y_rpn_overlap_repeat = K.reshape(y_true[:, :, :, :4 * num_anchors], (batch_size * height * width * 9, 4))
        y_rpn_regr = K.reshape(y_true[:, :, :, 4 * num_anchors:], (batch_size * height * width * 9, 4))
        pred_rpn_regr = K.reshape(y_pred, (batch_size * height * width * 9, 4))

        y_rpn_overlap = K.transpose(y_rpn_overlap_repeat)[0]
        y_rpn_regr = K.transpose(y_rpn_regr)
        pred_rpn_regr = K.transpose(pred_rpn_regr)
        
#         print("shape")
#         print(y_rpn_overlap.shape, y_rpn_overlap.dtype)
#         print(y_rpn_regr.shape, y_rpn_regr[0].shape, y_rpn_regr.dtype)
#         print(pred_rpn_regr.shape, pred_rpn_regr[0].shape, pred_rpn_regr.dtype)
        
        S_asterisk, S = cal_S(y_rpn_regr, pred_rpn_regr)
        S_I = cal_S_I(y_rpn_regr, pred_rpn_regr)
        S_C = cal_S_C(y_rpn_regr, pred_rpn_regr)
        
#         print("S")
#         print(S_asterisk.shape, S.shape)
#         print(S_I.shape, S_C.shape)
        
        IOU = S_I / (S + S_asterisk - S_I)
        IIOU = IOU - (S_C - (S + S_asterisk - S_I)) / S_C
        L_IIOU = 1 - IIOU
        
#         print("IOU")
#         print(IOU.shape, IIOU.shape, L_IIOU.shape)
#         print(tf.math.reduce_sum(y_rpn_overlap * L_IIOU))
#         print("END+++++++++++++++++++++++")
        
        return lambda_rpn_regr * K.sum(y_rpn_overlap * L_IIOU) / (epsilon + K.sum(y_rpn_overlap))

#         print('loss_regr', y_true.shape)
#         x = y_true[:, :, :, 4 * num_anchors:] - y_pred
#         x_abs = K.abs(x)
#         x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

#         return lambda_rpn_regr * K.sum(
#             y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / (epsilon + K.sum(y_true[:, :, :, :4 * num_anchors])) # smooth l1

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
#         print('loss_cls', y_true.shape)
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * \
            K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
