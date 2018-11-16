import math

import tensorflow as tf
from keras.layers import Activation, advanced_activations, BatchNormalization, MaxPooling2D, Conv2D

from toolkit.dirty import summary_filters


def sigmoid_math(x):
    return 1 / (1 + math.exp(-x))


def Sigmoid(x):
    return Activation('sigmoid')(x)


def ReLU(x):
    return Activation('relu')(x)


def LeakyReLU(x):
    return advanced_activations.LeakyReLU()(x)


def Softmax(x):
    return Activation('softmax')(x)


def BatchNorm(x, use=True):
    if use:
        return BatchNormalization()(x)
    else:
        return x


def MaxPool(x, pool_size=[2, 2], strides=None, padding='same'):
    if strides is None:
        strides_param = pool_size
    else:
        strides_param = strides
    resolved = MaxPooling2D(pool_size=pool_size, strides=strides_param, padding=padding)(x)
    print("MaxPool:", resolved)
    return resolved


def MaxPoolTF(x, pool_size=[2, 2], strides=None, padding='SAME', name=None):
    if strides is None:
        strides_param = pool_size
    else:
        strides_param = strides

    _ksize = [1] + pool_size + [1]
    _strides = [1] + strides_param + [1]

    resolved = tf.nn.max_pool(value=x, ksize=_ksize, strides=_strides, padding=padding, name=name)
    print("MaxPool:", resolved)
    return resolved


def ResolvedSyn(x=None, type='sigmoid'):

    if type == 'sigmoid':
        resolved = Sigmoid(x)
    elif type == 'tanh':
        resolved = Activation('tanh')(x)
    elif type == 'relu':
        resolved = ReLU(x)
    elif type == 'leakyrelu':
        resolved = LeakyReLU(x)
    elif type == 'elu':
        resolved = advanced_activations.ELU()(x)
    else:
        raise TypeError
    print("activated:", resolved)
    return resolved


def ResolvedSynTF(x=None, type='sigmoid', name=None):

    print('preact:', x)
    if type == 'sigmoid':
        resolved = tf.nn.sigmoid(x=x, name=name)
    elif type == 'tanh':
        resolved = tf.nn.tanh(x=x, name=name)
    elif type == 'relu':
        resolved = tf.nn.relu(features=x, name=name)
    elif type == 'elu':
        resolved = tf.nn.elu(features=x, name=name)
    else:
        raise TypeError
    print("activated:", resolved)
    return resolved


def InceptionUnit(x=None, scale=1, batch_norm=False, namespace=None, activation='relu'):

    print('-----')
    # 1x1 filter with s/2x2
    x_a = ResolvedSyn(x=Conv2D(filters=scale*5, kernel_size=[1, 1], strides=[2, 2], padding='same')(x),
                      type=activation)
    print('x_a:', x_a)

    # 1x1 -> 3x3 filter with s/2x2
    x_b1 = ResolvedSyn(x=Conv2D(filters=scale*3, kernel_size=[1, 1], padding='same')(x),
                      type=activation)
    print('x_b1:', x_b1)
    x_b2 = ResolvedSyn(x=Conv2D(filters=scale*4, kernel_size=[3, 3], strides=[2, 2], padding='same')(x_b1),
                      type=activation)
    print('x_b2:', x_b2)

    # 1x1 -> 5x5 filter with s/2x2
    x_c1 = ResolvedSyn(x=Conv2D(filters=scale*2, kernel_size=[1, 1], padding='same')(x),
                      type=activation)
    print('x_c1:', x_c1)
    x_c2 = ResolvedSyn(x=Conv2D(filters=scale*3, kernel_size=[5, 5], strides=[2, 2], padding='same')(x_c1),
                      type=activation)
    print('x_c2:', x_c2)

    x_d1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='same')(x)
    print('x_d1:', x_d1)
    x_d2 = ResolvedSyn(x=Conv2D(filters=scale*4, kernel_size=[1, 1], padding='same')(x_d1),
                      type=activation)
    print('x_d2:', x_d2)

    concat = tf.concat(values=[x_a, x_b2, x_c2, x_d2], axis=-1)
    if batch_norm:
        concat = BatchNorm(concat)
    else:
        pass

    if namespace is not None:
        print("%s: %s" % (namespace, str(concat)))
        summary_filters(namespace=namespace, tensor=concat, num_of_filters=16)
    else:
        print("concat:", concat)
    print('-----')
    return concat


def inception_unit_resolve_act(x=None, scale=1, batch_norm=False, activation=None, namespace=None):

    print('-----')
    # 1x1 filter with s/2x2
    x_a = conv_layer_resolve_act(input=x, filters=scale * 5, kernel_size=[1, 1], strides=[2, 2], padding='SAME',
                                 activation=activation, name='%s/incept_x_a' % namespace)
    print('x_a:', x_a)

    # 1x1 -> 3x3 filter with s/2x2
    x_b1 = conv_layer_resolve_act(input=x, filters=scale * 3, kernel_size=[1, 1], padding='SAME',
                                  activation=activation, name='%s/incept_x_b1' % namespace)
    print('x_b1:', x_b1)

    x_b2 = conv_layer_resolve_act(input=x_b1, filters=scale * 4, kernel_size=[3, 3], strides=[2, 2], padding='SAME',
                                  activation=activation, name='%s/incept_x_b2' % namespace)
    print('x_b2:', x_b2)

    # 1x1 -> 5x5 filter with s/2x2
    x_c1 = conv_layer_resolve_act(input=x, filters=scale * 2, kernel_size=[1, 1], padding='SAME',
                                  activation=activation, name='%s/incept_x_c1' % namespace)
    print('x_c1:', x_c1)
    x_c2 = conv_layer_resolve_act(input=x_c1, filters=scale * 3, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
                                  activation=activation, name='%s/incept_x_c2' % namespace)
    print('x_c2:', x_c2)

    x_d1 = MaxPoolTF(x=x, pool_size=[3, 3], strides=[2, 2], padding='SAME',
                     name='%s/incept_x_d1' % namespace)
    print('x_d1:', x_d1)

    x_d2 = conv_layer_resolve_act(input=x_d1, filters=scale * 4, kernel_size=[1, 1], padding='SAME',
                                  activation=activation, name='%s/incept_x_d2' % namespace)
    print('x_d2:', x_d2)

    concat = tf.concat(values=[x_a, x_b2, x_c2, x_d2], axis=-1)
    # if batch_norm:
    #     concat = BatchNorm(concat)
    # else:
    #     pass

    if namespace is not None:
        print("%s: %s" % (namespace, str(concat)))
        with tf.device('/cpu:0'):
            summary_filters(namespace=namespace, tensor=concat, num_of_filters=16)
    else:
        print("concat:", concat)
    print('-----')
    return concat


def conv_layer_resolve_act(input=None, filters=None, kernel_size=None, strides=None,
                   padding='VALID', activation=None, name=None):

    _strides = strides if strides is not None else [1, 1]

    net = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size,
                           strides=_strides, padding=padding, name=name)
    net = ResolvedSynTF(x=net, type=activation, name='%s_act' % name)

    return net


def conv_maxpool_layer_resolve_act(input=None, filters=None, kernel_size=None, strides=None,
                       padding='VALID', activation=None, name=None, summary_num_of_filters=1):

    _strides = strides if strides is not None else [1, 1]

    print('_strides:', _strides)

    net = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size,
                           strides=_strides, padding=padding, name=name)
    net = ResolvedSynTF(x=net, type=activation, name='%s_act' % name)
    net = MaxPoolTF(x=net, padding=padding, name='%s_maxpool' % name)
    # omit batch normalization

    with tf.device('/cpu:0'):
        summary_filters(namespace=name, tensor=net, num_of_filters=summary_num_of_filters)

    return net


def dense_layer_resolve_act(input=None, units=None, activation=None, name=None):

    net = tf.layers.dense(inputs=input, units=units, name=name)
    net = ResolvedSynTF(x=net, type=activation, name='%s_act' % name)

    return net
