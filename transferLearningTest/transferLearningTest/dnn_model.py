import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

def get_model_type(model_type):
    return {
        'VGG':tf.contrib.slim.nets.vgg, 
        'AlexNet':tf.contrib.slim.nets.alexnet,
        'ResNet':tf.contrib.slim.nets.resnet_v2, 
        'Inception':tf.contrib.slim.nets.inception
        }[model_type]


def predict_VGG(model_type, images, num_classes, weight_decay, drop_out, is_training):
    dnn_model = get_model_type(model_type)
    with slim.arg_scope(dnn_model.vgg_arg_scope(weight_decay=weight_decay)):
        logits, _ = dnn_model.vgg_16(images, num_classes=num_classes, is_training=is_training, 
                                     dropout_keep_prob=drop_out)

    return logits

def predict_AlexNet(model_type, images, num_classes, weight_decay, drop_out, is_training):
    dnn_model = get_model_type(model_type)
    with slim.arg_scope(dnn_model.alexnet_v2_arg_scope(weight_decay=weight_decay)):
        logits, _ = dnn_model.alexnet_v2(images, num_classes=num_classes, is_training=is_training, 
                                         dropout_keep_prob=drop_out)

    return logits

def predict_ResNet(model_type, images, num_classes, weight_decay, drop_out, is_training):
    dnn_model = get_model_type(model_type)
    with slim.arg_scope(dnn_model.resnet_arg_scope()):
        logits, _ = dnn_model.resnet_v2_152(images, num_classes=num_classes, is_training=is_training)

    return logits

def predict_Inception(model_type, images, num_classes, weight_decay, drop_out, is_training):
    dnn_model = get_model_type(model_type)
    with slim.arg_scope(dnn_model.inception_v3_arg_scope(weight_decay=weight_decay)):
        logits, _ = dnn_model.inception_v3(images, num_classes=num_classes, is_training=is_training, 
                                           dropout_keep_prob=drop_out)

    return logits

# class interface
def init_variables(model_path, excluded_variables):
    var_restore_init = None
    var_reinitialize_init = None

    excluded_variables_list = []
    excluded_variables_list.append(excluded_variables)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=excluded_variables_list)
    var_restore_init = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

    # Initialization operation from scratch for the new "fc8" layers
    # `get_variables` will only return the variables whose name starts with the given pattern
    variables_to_reinitialize = tf.contrib.framework.get_variables(excluded_variables)
    var_reinitialize_init = tf.variables_initializer(variables_to_reinitialize)

    return var_restore_init, var_reinitialize_init


def predict(model_type, images, num_classes, weight_decay, drop_out, is_training):
    # switch-case paradigma :)
    map_predictor = {'VGG':predict_VGG, 'AlexNet':predict_AlexNet,
                    'ResNet':predict_ResNet, 'InceptNet':predict_Inception}

    prediction_result = map_predictor[model_type](model_type, images, num_classes, weight_decay, drop_out, is_training)
    prediction_result = tf.squeeze(prediction_result)

    return prediction_result
