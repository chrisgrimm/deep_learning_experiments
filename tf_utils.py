import tensorflow as tf

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
        next, layer_vars = connected_layer(next, out_dim, nonlinearity, prefix=prefix+'_layer%s_' % layer)
        vars.update(layer_vars)
    return next, vars

def connected_layer(input, out_dim, nonlinearity, prefix=''):
    in_dim = input.get_shape()[1].value
    vars = dict()
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
    return nonlinearity_mapping[nonlinearity](tf.matmul(input, w) + b), vars

