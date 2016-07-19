import tensorflow as tf
import numpy as np

def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def bias_variable(shape):
    initial = tf.random_normal(stddev=0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def connected_layers(input, out_dims, nonlinearities, prefix=''):
    next = input
    vars = dict()
    for layer, (out_dim, nonlinearity) in enumerate(zip(out_dims, nonlinearities)):
        is_final = (layer == (len(nonlinearities)-1))
        next, layer_vars = connected_layer(next, out_dim, nonlinearity, is_final, prefix=prefix+'_layer%s_' % layer)
        vars.update(layer_vars)
    return next, vars

def connected_layer(input, out_dim, nonlinearity, final_layer, prefix=''):
    in_dim = input.get_shape()[1].value
    vars = dict()
    if out_dim == 3:
        w = vars[prefix+'w'] = tf.Variable(tf.constant(0.001, tf.float32, shape=[in_dim, out_dim]))
        b = vars[prefix+'b'] = tf.Variable(initial_value=np.array([1, 0, 0], dtype=np.float32))
    else:
        w = vars[prefix+'w'] = weight_variable([in_dim, out_dim])
        b = vars[prefix+'b'] = bias_variable([out_dim])
    nonlinearity_mapping = {
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'none': lambda x: x
    }
    if nonlinearity not in nonlinearity_mapping:
        raise Exception('unsupported nonlinearity')
    res = nonlinearity_mapping[nonlinearity](tf.matmul(input, w) + b)
    if final_layer:
        return res, vars
    else:
        res = tf.nn.batch_normalization(res, tf.zeros_like(res), tf.ones_like(res), 0, 1, 1)
        return res, vars

