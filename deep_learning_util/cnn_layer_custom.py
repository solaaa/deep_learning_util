from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import random

def conv_layer(x, kernel_size, stride, padding='SAME', mode='standard', name='c'):
    '''
    to customize conv. layer
    params:
        x: input tensor
        kernel_size: [length, width, input_channel, output_channel], same as 'tf.nn.conv2d'
        stride: same as 'tf.nn.conv2d'
        mode: 'standard', 'keras' and 'other'
            standard: use 'he_normal' to initialize the w and b
            keras: use keras_api to initialize conv. layer directly, also can choose init. method by keras.layers.Conv2D()
            other: use 'selu' method to initialize w and b, to satisfy selu function needs.
    '''
    if mode == 'standard':
        #he_normal
        w = tf.Variable(
             tf.random.truncated_normal(
              kernel_size,
              stddev=np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
             name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001),
             name=name+'_b')
        c_out = tf.nn.conv2d(x, w, stride, padding) + b
    elif mode == 'keras':
        c_out = tf.keras.layers.Conv2D(kernel_size[3],
                                       [kernel_size[0], kernel_size[1]],
                                       [stride[1], stride[2]],
                                       padding=padding)(x)
    elif mode == 'other' :
        w = tf.Variable(
             tf.random.truncated_normal(
              kernel_size,
              stddev = np.sqrt( 1/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )),
             name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001),
                 name=name+'_b')
        c_out = tf.nn.conv2d(x, w, stride, padding) + b
    return c_out

def fc_layer(x, last_layer_element_count, unit_num, mode = 'standard', name='fc'):
    '''
    to customize fully connected layer
    params:
        x: input tensor
        last_layer_element_count: the number of elements of x
        unit_num: the number of hidden units
        mode: same as conv_layer
    '''
    if mode =='standard':
        w = tf.Variable(
                tf.random.truncated_normal(
                [last_layer_element_count, unit_num], stddev=0.01),
                name = name+'_w')
        b = tf.Variable(
                tf.random.truncated_normal([unit_num], stddev=0.01),
                name = name+'_b')

        fc_out = tf.matmul(x, w) + b
    elif mode == 'keras':
        fc_out = tf.keras.layers.Dense(unit_num)(x)
    elif mode == 'selu' or mode == 'chip_acti_fun' :
        w = tf.Variable(
             tf.random.truncated_normal(
                [last_layer_element_count, unit_num], 
                stddev=np.sqrt(1/last_layer_element_count)),
                name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [unit_num], 
                stddev=0),
                name=name+'_b')
        fc_out = tf.matmul(x, w) + b
    return fc_out

def identity_block_v1(x, kernel_size, stride, padding='SAME', mode='standard', name='b'):
    '''
    refer to 'Deep Residual Learning for Image Recognition.2015' (https://arxiv.org/pdf/1512.03385.pdf)
    for my own needs, batch normalization is not adopted
    meanwhile, activation function is set before tf.add()
    '''
    inp = x # channel=32
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[2]], 
                   stride, padding='SAME', mode=mode, name=name+'_c1')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x, mode)

    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x, mode)
    out = tf.add(inp, x)
    return out

def identity_block_v2(x, kernel_size, stride, padding='SAME', mode='selu', name='b'):
    '''
    refer to 'Deep Residual Learning for Image Recognition.2015' (https://arxiv.org/pdf/1512.03385.pdf)
    the bottleneck version of identity_block
    '''
    expend_channel = 128
    inp = x # channel=32
    x = conv_layer(x, [1,1,kernel_size[2],expend_channel], 
                   stride, padding='SAME', mode=mode, name=name+'_c1')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x, mode)

    x = conv_layer(x, [kernel_size[0],kernel_size[1],expend_channel,expend_channel], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x, mode)

    x = conv_layer(x, [1,1,expend_channel,kernel_size[3]], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = activation(x, mode)
    
    out = tf.add(inp, x)
    return out

def group_conv_block(x, group, kernel_size, stride, padding='SAME', mode='selu', name='b'):
    '''
    refer to ResNeXt: https://arxiv.org/pdf/1611.05431.pdf
    '''
    inp = x
    channel_per_group = int(kernel_size[2]/group)
    x_group = tf.split(x, group, -1)
    for i in range(group):
        x_group[i]=conv_layer(x_group[i], 
                                    [kernel_size[0],kernel_size[1],channel_per_group,channel_per_group], 
                                    stride, padding='SAME', mode=mode, name=name+'_g%d_c1'%(i))
        x_group[i] = activation(x_group[i], mode)

    x = tf.concat([i for i in x_group], axis=-1)
    x=conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[2]], 
                    stride, padding='SAME', mode=mode, name=name+'_c2')
    x = activation(x, mode)
    out = tf.add(inp, x)
    return out

def conv_block(x, kernel_size, stride, padding='SAME', mode='selu'):
    '''
    refer to 'Deep Residual Learning for Image Recognition.2015' (https://arxiv.org/pdf/1512.03385.pdf)
    use to change the dimension
    '''
    inp = x 
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    x = activation(x, mode)

    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[3],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    x = activation(x, mode)

    # direct shortcut
    d = conv_layer(inp, [1,1,kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    d = activation(d, mode)
    out = tf.add(d, x)
    return out

def depthwise_seperable_conv(x, kernel_size, stride):

    # init the w, b
    w_depthwise = tf.Variable(
             tf.random.truncated_normal(
              [kernel_size[0], kernel_size[1], kernel_size[2], 1],
              stddev = np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )))
    b_depthwise = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[2]], 
                stddev=0.001))
    w_pointwise = tf.Variable(
             tf.random.truncated_normal(
              [1,1,kernel_size[2],kernel_size[3]],
              stddev = np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )))
    b_pointwise = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001))

    # depthwise
    out_depthwise = tf.nn.depthwise_conv2d(x, w_depthwise, stride, padding='SAME') + b_depthwise
    out_depthwise = tf.nn.relu(out_depthwise)

    # pointwise, linear (refer from mobile_net)
    out_pointwise = tf.nn.conv2d(out_depthwise, w_pointwise, [1,1,1,1], padding='SAME') + b_pointwise
    return out_pointwise

def ds_identity_block(x, kernel_size, stride):
    inp = x
    x = depthwise_seperable_conv(x, kernel_size, stride)
    x = depthwise_seperable_conv(x, kernel_size, stride)
    out = tf.add(inp, x)
    return out

def activation(x, mode):
    if mode=='standard':
        out = tf.nn.relu(x)
    elif mode=='selu':
        out = tf.nn.selu(x)
    elif mode=='other':
        out = chip_acti_fun(x)
    return out
def chip_acti_fun(x):
    x = tf.nn.relu(x)
    x = 1-tf.exp(-2*x)
    return x

def dropout_with_mode(x,dropout_prob, mode):
    if mode=='standard':
        y=tf.nn.dropout(x, rate=dropout_prob)
    elif mode=='selu':
        y=dropout_selu(x, dropout_prob, training=True)
    elif mode=='other':
        y=tf.nn.dropout(x, rate=dropout_prob)
    return y
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling.
        1. Self-Normalizing Neural Networks
        https://arxiv.org/pdf/1706.02515.pdf
        https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    """

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        #keep_prob = rate # original code
        keep_prob = 1.0 - rate # rate means dropout_prob, keep_prob should be 1-rate.
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))


