import tensorflow as tf
import numpy as np
from tf_utils import *


def hookLocalizer(input, weights, p):
    hidden_layer = tf.nn.tanh(tf.matmul(input, weights[p + "W1"]) + weights[p + "b1"])
    output = tf.nn.tanh(tf.matmul(hidden_layer, weights[p + "W2"]) + weights[p + "b2"])
    return output, hidden_layer

def createLocalizerWeights(input_dims, hidden_dims, output_dims, p):
    weights = {p + "W1" : tf.Variable(tf.random_normal(shape=(input_dims, hidden_dims))),
               p + "b1" : tf.Variable(tf.random_normal(shape=(1, hidden_dims))),
               p + "W2" : tf.Variable(tf.random_normal(shape=(hidden_dims, output_dims))),
               p + "b2" : tf.Variable(tf.random_normal(shape=(1, output_dims)))}
    return weights


def fully_connected_localizer(input, hidden_dims=20):
    batch_size = tf.shape(input)[0]
    height, width = input.get_shape()[1:]
    flat = tf.reshape(input, [-1, width * height])
    batch_size = tf.shape(input)[0]
    loc_w1 = weight_variable([height * width, hidden_dims])
    loc_b1 = bias_variable([hidden_dims])
    loc_h1 = tf.nn.tanh(tf.matmul(flat, loc_w1) +  loc_b1)
    loc_w2 = weight_variable([hidden_dims, 6])
    loc_b2 = bias_variable([6])
    theta = tf.nn.tanh(tf.matul(loc_h1, loc_w2) + loc_b2)
    return tf.reshape(theta, [-1, 2, 3])

class SpatialTransformerNetwork(object):

    def __init__(self, input_layer, (output_height, output_width), kernel='nearest', custom_kernel=None):
        self.input_layer = input_layer
        self.batch_size = tf.shape(self.input_layer)[0]
        self.height, self.width = self.input_layer.get_shape()[1:]
        self.height = self.height.value
        self.width = self.width.value
        self.kernel = kernel
        supported_kernels = ['nearest', 'bilinear', 'custom']
        if self.kernel not in supported_kernels:
            raise Exception('Kernel type %s not supported. Supported types are '+' '.join(supported_kernels)+'.')
        if self.kernel == 'custom' and custom_kernel is None:
            raise Exception('custom_kernel keyword argument must be specified if kernel=\'custom\'')
        self.output_width = output_width
        self.output_height = output_height
        self.flattened = tf.reshape(self.input_layer, shape=[-1, self.height * self.width])


    def _2dMeshGrid(self, height, width):
        # produces two index vectors
        # for 4 (height) x 3 (width) vector
        # (1 1 1) (2 2 2) (3 3 3) (4 4 4) <- height (numbers in height_grid go up to
        # (1 2 3) (1 2 3) (1 2 3) (1 2 3) <- width
        width_grid = tf.tile(tf.reshape(tf.range(0, width), [1, -1]), [1, height])
        height_grid = tf.reshape(tf.tile(tf.reshape(tf.range(0, height), [-1, 1]), [1, width]), [1, -1])
        return tf.to_float(width_grid), tf.to_float(height_grid)

    def transform(self, localizers):
        # build meshgrids
        w_grid, h_grid = self._2dMeshGrid(self.output_height, self.output_width)
        # normalize meshgrids,
        # add two extra dimensions 1 is batch_size, 2 is for later batch_matmul
        # tile along new dim 1
        x_target_norm_grid = tf.tile(tf.reshape((w_grid - self.output_width/2.0) / (self.output_width/2.0), [1, 1, -1]), [self.batch_size, 1, 1])
        y_target_norm_grid = tf.tile(tf.reshape((h_grid - self.output_height/2.0) / (self.output_height/2.0), [1, 1, -1]), [self.batch_size, 1, 1])
        w_int_grid, h_int_grid = self._2dMeshGrid(self.height, self.width)
        # add one extra dimension to x_source_int_grid for batch size.
        x_source_int_grid = tf.tile(tf.reshape(w_int_grid, [1, -1]), [self.batch_size, 1])
        y_source_int_grid = tf.tile(tf.reshape(h_int_grid, [1, -1]), [self.batch_size, 1])
        # use dimension 1 added to target grids to stack x y and 1 terms
        # this is used for batch applying the localizer transform
        combined = tf.concat(1, [x_target_norm_grid, y_target_norm_grid, tf.ones_like(x_target_norm_grid)])
        # multiply the corresponding localizer matrix by each [x; y; 1] matrix column on the 2nd dim of combined.
        source = tf.batch_matmul(localizers, combined)
        # after multiplying source is [batch_size, 2, out_width * out_height], next slice the x and y out of dim 1.
        source_x = tf.slice(source, [0, 0, 0], [-1, 1, -1])
        source_y = tf.slice(source, [0, 1, 0], [-1, 1, -1])
        # convert slices into 0 -> end of image coordinates.
        converted_x = (source_x + 1) * self.width/2
        converted_y = (source_y + 1) * self.height/2
        tiled_x = tf.tile(tf.reshape(converted_x, [self.batch_size, 1, -1]), [1, self.width * self.height, 1])
        tiled_y = tf.tile(tf.reshape(converted_y, [self.batch_size, 1, -1]), [1, self.width * self.height, 1])
        tiled_x_source_int_grid = tf.tile(tf.reshape(x_source_int_grid, [self.batch_size, -1, 1]), [1, 1, self.output_width * self.output_height])
        tiled_y_source_int_grid = tf.tile(tf.reshape(y_source_int_grid, [self.batch_size, -1, 1]), [1, 1, self.output_width * self.output_height])
        tiled_flattened = tf.tile(tf.reshape(self.flattened, [self.batch_size, -1, 1]), [1, 1, self.output_width*self.output_height])
        # integer sampling
        if self.kernel == 'nearest':
            kern_x = tf.to_float(tf.equal(tf.floor(tiled_x + 0.5), tiled_x_source_int_grid))
            kern_y = tf.to_float(tf.equal(tf.floor(tiled_y + 0.5), tiled_y_source_int_grid))
        elif self.kernel == 'bilinear':
            kern_x = tf.maximum(tf.zeros_like(tiled_x), 1 - tf.abs(tiled_x - tiled_x_source_int_grid))
            kern_y = tf.maximum(tf.zeros_like(tiled_y), 1 - tf.abs(tiled_y - tiled_y_source_int_grid))
        elif self.kernel == 'custom':
            # TODO add support for custom kernels
            raise NotImplemented
        V = tf.reduce_sum(kern_x * kern_y * tiled_flattened, reduction_indices=[1])
        return tf.reshape(V, shape=[-1, self.output_height, self.output_width])