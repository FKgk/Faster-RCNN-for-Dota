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
        print('rpn_loss_regr', y_true.shape, y_pred.shape)
        
        batch_size, height, width, c = y_true.shape
        
        y_rpn_overlap_repeat = K.reshape(y_true[:, :, :, :4 * num_anchors], (batch_size * height * width * 9, 4))
        y_rpn_regr = K.reshape(y_true[:, :, :, 4 * num_anchors:], (batch_size * height * width * 9, 4))
        pred_rpn_regr = K.reshape(y_pred, (batch_size * height * width * 9, 4))

        y_rpn_overlap = K.transpose(y_rpn_overlap_repeat)[0]
        y_rpn_regr = K.transpose(y_rpn_regr)
        pred_rpn_regr = K.transpose(pred_rpn_regr)
        
        S_asterisk, S = cal_S(y_rpn_regr, pred_rpn_regr)
        S_I = cal_S_I(y_rpn_regr, pred_rpn_regr)
        S_C = cal_S_C(y_rpn_regr, pred_rpn_regr)
        
        IOU = S_I / (S + S_asterisk - S_I + K.epsilon())
        IIOU = IOU - (S_C - (S + S_asterisk - S_I)) / (S_C + K.epsilon())
        L_IIOU = 1 - IIOU
        
        return lambda_rpn_regr * K.sum(y_rpn_overlap * L_IIOU) / (K.sum(y_rpn_overlap) + K.epsilon())
    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        print('rpn_loss_cls', y_true.shape)
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * \
            K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num

def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        print(f"class_loss_regr {y_true.shape} , {y_pred.shape}")
        print(f"class_loss_regr {y_true.dtype} , {y_pred.dtype}")

        batch_size, object_num, class_num = y_true.shape
        
        print(y_true[:, :, :4 * num_classes].shape)
        print(y_true[:, :, 4*num_classes:].shape)
        
        y_rpn_overlap_repeat = K.reshape(y_true[:, :, :4 * num_classes], (batch_size * object_num * num_classes, 4))
        y_rpn_regr = K.reshape(y_true[:, :, 4*num_classes:], (batch_size * object_num * num_classes, 4))
        pred_rpn_regr = K.reshape(y_pred, (batch_size * object_num * num_classes, 4))      
        
        y_rpn_overlap = K.transpose(y_rpn_overlap_repeat)[0]
        y_rpn_regr = K.transpose(y_rpn_regr)
        pred_rpn_regr = K.transpose(pred_rpn_regr)
        
        S_asterisk, S = cal_S(y_rpn_regr, pred_rpn_regr)
        S_I = cal_S_I(y_rpn_regr, pred_rpn_regr)
        S_C = cal_S_C(y_rpn_regr, pred_rpn_regr)
        
        IOU = S_I / (S + S_asterisk - S_I + K.epsilon())
        IIOU = IOU - (S_C - (S + S_asterisk - S_I)) / (S_C + K.epsilon())
        L_IIOU = 1 - IIOU
        
        return lambda_cls_regr * K.sum(y_rpn_overlap * L_IIOU) / (K.sum(y_rpn_overlap) + K.epsilon())
    return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    print(f"class_loss_cls {y_true.shape} , {y_pred.shape}")
    print(f"class_loss_cls {y_true.dtype} , {y_pred.dtype}")
    
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
