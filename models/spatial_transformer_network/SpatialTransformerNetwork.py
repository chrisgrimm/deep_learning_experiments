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

class SpatialTransformerNetwork(object):

    def __init__(self, input_layer, (input_height, input_width), (output_height, output_width), kernel='nearest', custom_kernel=None):
        self.input_layer = input_layer
        self.batch_size = tf.shape(self.input_layer)[0]
        self.height, self.width = input_height, input_width
        self.kernel = kernel
        supported_kernels = ['nearest', 'bilinear', 'custom']
        if self.kernel not in supported_kernels:
            raise Exception('Kernel type %s not supported. Supported types are '+' '.join(supported_kernels)+'.')
        if self.kernel == 'custom' and custom_kernel is None:
            raise Exception('custom_kernel keyword argument must be specified if kernel=\'custom\'')
        self.output_width = output_width
        self.output_height = output_height
        # create the network
        self._localizationNetwork()
        self._gridGeneration()

    def _localizationNetwork(self):
        self.flattened = tf.reshape(self.input_layer, shape=[-1, self.height * self.width])
        # hidden 1
        self.loc_w1 = weight_variable([self.height * self.width, 20])
        self.loc_b1 = bias_variable([20])
        self.loc_h1 = tf.nn.tanh(tf.add(tf.matmul(self.flattened, self.loc_w1), self.loc_b1))
        self.loc_w2 = weight_variable([20, 6])
        self.loc_b2 = tf.Variable(initial_value=np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0]], dtype=np.float32).flatten())
        self.theta = tf.nn.tanh(tf.add(tf.matmul(self.loc_h1, self.loc_w2), self.loc_b2))
        self.localizers = tf.reshape(self.theta, shape=[-1, 2, 3])


    def _2dMeshGrid(self, height, width):
        # produces two index vectors
        # for 4 (height) x 3 (width) vector
        # (1 1 1) (2 2 2) (3 3 3) (4 4 4) <- height (numbers in height_grid go up to
        # (1 2 3) (1 2 3) (1 2 3) (1 2 3) <- width
        width_grid = tf.tile(tf.reshape(tf.range(0, width), [1, -1]), [1, height])
        height_grid = tf.reshape(tf.tile(tf.reshape(tf.range(0, height), [-1, 1]), [1, width]), [1, -1])
        return tf.to_float(width_grid), tf.to_float(height_grid)

    def _gridGeneration(self):
        # build meshgrids
        w_grid, h_grid = self._2dMeshGrid(self.output_height, self.output_width)
        # normalize meshgrids,
        # add two extra dimensions 1 is batch_size, 2 is for later batch_matmul
        # tile along new dim 1
        self.x_target_norm_grid = tf.tile(tf.reshape((w_grid - self.output_width/2.0) / (self.output_width/2.0), [1, 1, -1]), [self.batch_size, 1, 1])
        self.y_target_norm_grid = tf.tile(tf.reshape((h_grid - self.output_height/2.0) / (self.output_height/2.0), [1, 1, -1]), [self.batch_size, 1, 1])
        w_int_grid, h_int_grid = self._2dMeshGrid(self.height, self.width)
        # add one extra dimension to x_source_int_grid for batch size.
        self.x_source_int_grid = tf.tile(tf.reshape(w_int_grid, [1, -1]), [self.batch_size, 1])
        self.y_source_int_grid = tf.tile(tf.reshape(h_int_grid, [1, -1]), [self.batch_size, 1])
        # use dimension 1 added to target grids to stack x y and 1 terms
        # this is used for batch applying the localizer transform
        self.combined = tf.concat(1, [self.x_target_norm_grid, self.y_target_norm_grid, tf.ones_like(self.x_target_norm_grid)])
        # multiply the corresponding localizer matrix by each [x; y; 1] matrix column on the 2nd dim of combined.
        self.source = tf.batch_matmul(self.localizers, self.combined)
        # after multiplying source is [batch_size, 2, out_width * out_height], next slice the x and y out of dim 1.
        self.source_x = tf.slice(self.source, [0, 0, 0], [-1, 1, -1])
        self.source_y = tf.slice(self.source, [0, 1, 0], [-1, 1, -1])
        # convert slices into 0 -> end of image coordinates.
        self.converted_x = (self.source_x + 1) * self.width/2
        self.converted_y = (self.source_y + 1) * self.height/2
        self.tiled_x = tf.tile(tf.reshape(self.converted_x, [self.batch_size, 1, -1]), [1, self.width * self.height, 1])
        self.tiled_y = tf.tile(tf.reshape(self.converted_y, [self.batch_size, 1, -1]), [1, self.width * self.height, 1])
        self.tiled_x_source_int_grid = tf.tile(tf.reshape(self.x_source_int_grid, [self.batch_size, -1, 1]), [1, 1, self.output_width * self.output_height])
        self.tiled_y_source_int_grid = tf.tile(tf.reshape(self.y_source_int_grid, [self.batch_size, -1, 1]), [1, 1, self.output_width * self.output_height])
        self.tiled_flattened = tf.tile(tf.reshape(self.flattened, [self.batch_size, -1, 1]), [1, 1, self.output_width*self.output_height])
        # integer sampling
        if self.kernel == 'nearest':
            self.kern_x = tf.to_float(tf.equal(tf.floor(self.tiled_x + 0.5), self.tiled_x_source_int_grid))
            self.kern_y = tf.to_float(tf.equal(tf.floor(self.tiled_y + 0.5), self.tiled_y_source_int_grid))
        elif self.kernel == 'bilinear':
            self.kern_x = tf.maximum(tf.zeros_like(self.tiled_x), 1 - tf.abs(self.tiled_x - self.tiled_x_source_int_grid))
            self.kern_y = tf.maximum(tf.zeros_like(self.tiled_y), 1 - tf.abs(self.tiled_y - self.tiled_y_source_int_grid))
        elif self.kernel == 'custom':
            # TODO add support for custom kernels
            raise NotImplemented
        self.V = tf.reduce_sum(self.kern_x * self.kern_y * self.tiled_flattened, reduction_indices=[1])
        self.output = tf.reshape(self.V, shape=[-1, self.output_height, self.output_width])