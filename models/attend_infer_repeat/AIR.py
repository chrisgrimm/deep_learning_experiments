import tensorflow as tf
import numpy as np
from models.spatial_transformer_network.SpatialTransformerNetwork import SpatialTransformerNetwork as STN
from models.variational_autoencoder.VariationalAutoencoder import VAE, VAE_realize, log_bernoulli_pmf, log_normal_pdf
from tf_utils import *

class AIR(object):

    def __init__(self, sess, input, height, width, lstm_units, max_objects, batch_size):
        self.ih, self.iw = height, width
        self.batch_size = batch_size
        self.sess = sess
        self.input = input
        self.lstm_u = lstm_units
        self.N = max_objects
        self.z_what_random = []
        self.z_where_random = []
        self.z_pres_random = []
        self.inv_where_random = []
        self.vars = dict()
        self.setup_network()

    def setup_network(self):
        x = self.input
        ih, iw = self.ih, self.iw
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        state = np.zeros([self.batch_size, self.lstm_u])
        # the 'initial' pres value cant be trainable.
        prev_pres = tf.ones([self.batch_size, 1])
        cum_q_z_pres = cum_q_z_where = cum_q_z_what = cum_p_x_given_z = cum_p_z = tf.zeros([self.batch_size, 1])
        self.output = tf.zeros_like(x)
        # step through the lstm.
        for iter in range(self.N):
            output, state = lstm(x, state)
            # ENCODER
            # set up z_where params
            z_where_params, vars = VAE(output, 100, 3, 'gaussian', prefix='z_where_%s_' % iter)
            z_where_random_step = tf.placeholder(tf.float32, [self.batch_size, 3])
            self.z_where_random.append(z_where_random_step)
            self.vars.update(vars)
            z_where = VAE_realize(z_where_params, z_where_random_step, 'gaussian')
            # set up z_pres params multiply by old previous value... (if prev pres is zero for an image, it kills the bernoulli distr)
            z_pres_params, vars = VAE(output, 100, 1, 'bernoulli', prefix='z_pres_%s_' % iter)
            z_pres_params = z_pres_params[0] * prev_pres
            z_pres_random_step = tf.placeholder(tf.float32, [self.batch_size, 1])
            self.z_pres_random.append(z_pres_random_step)
            self.vars.update(vars)
            z_pres = prev_pres = VAE_realize([z_pres_params], z_pres_random_step, 'bernoulli')
            # set up z_what params
            where_flat, vars = connected_layers(z_where, [100, 3], ['tanh', 'tanh'], prefix='localizer_%s_' % iter)
            self.vars.update(vars)
            where = tf.gather(where_flat, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            # use spatially transformed where param to make z_what
            stn = STN(x, (28, 28))
            x_att = stn.transform(where)
            z_what_params, vars = VAE(x_att, 500, 20, 'gaussian', prefix='z_what_%s_' % iter)
            self.vars.update(vars)
            z_what_random_step = tf.placeholder(tf.float32, [self.batch_size, 20])
            self.z_what_random.append(z_what_random_step)
            z_what = VAE_realize(z_what_params, z_what_random_step, 'gaussian')
            # update the cumulative q(z|x) and p(z) nodes.
            cum_q_z_pres += log_bernoulli_pmf(x, z_pres_params)
            cum_q_z_where += log_normal_pdf(x, z_where_params[0], z_where_params[1])
            cum_q_z_what += log_normal_pdf(x, z_what_params[0], z_what_params[1])
            cum_p_z += log_normal_pdf(z_where, tf.zeros_like(z_where_params[0]), tf.ones_like(z_where_params[1])) + \
                       log_bernoulli_pmf(z_pres, tf.ones_like(z_pres_params)/2) + \
                       log_normal_pdf(z_what, tf.zeros_like(z_what_params[0]), tf.ones_like(z_what_params[1]))
            # DECODER
            # decode z_what into image
            y_att_flat_params, vars = VAE(z_what, 500, 28*28, 'gaussian', prefix='y_att_%s_' % iter)
            y_att = tf.reshape(y_att_flat_params[0], [-1, 28, 28])
            z_inv_where_params, vars = VAE(y_att, 3, 3, 'gaussian', prefix='z_inv_where_%s_' % iter)
            self.vars.update(vars)
            z_inv_where_random_step = tf.placeholder(tf.float32, [self.batch_size, 3])
            self.inv_where_random.append(z_inv_where_random_step)
            z_inv_where = VAE_realize(z_inv_where_params, z_inv_where_random_step, 'gaussian')
            inv_where, vars = connected_layers(z_inv_where, [100, 3], ['tanh', 'tanh'], prefix='inv_localizer_%s_' % iter)
            self.vars.update(vars)
            inv_where_loc = tf.gather(inv_where, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            inv_stn = STN(y_att, (ih, iw))
            y_i = inv_stn.transform(inv_where_loc)
            pres_mask = tf.tile(z_pres, [1, ih, iw])
            y_i_masked = y_i * pres_mask
            self.output += y_i_masked
            # update the p_x_given_z node
            # CHECK THIS LINE: UNCLEAR WHAT TO DO TO P(X|Z) WHEN SHIFT FROM Z_INV_WHERE TO INV_WHERE
            cum_p_x_given_z += log_normal_pdf(tf.reshape(x_att, [-1, 28, 28]), y_att_flat_params[0], y_att_flat_params[1])
        p = cum_p_z + cum_p_x_given_z
        q = cum_q_z_what + cum_q_z_where + cum_q_z_pres
        self.loss = tf.reduce_mean(-(p - q), reduction_indices=0)
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def add_randomness(self, feed_dict, batch_size):
        for i in range(self.N):
            feed_dict[self.z_what_random[i]] = np.random.random((batch_size, 20))
            feed_dict[self.z_where_random[i]] = np.random.random((batch_size, 3))
            feed_dict[self.z_pres_random[i]]  = np.random.random((batch_size, 1))
            feed_dict[self.inv_where_random[i]] = np.random.random((batch_size, 3))

    def train_batch(self, batch, batch_size):
        feed_dict = {}
        self.add_randomness(feed_dict, batch_size)
        feed_dict[self.input] = batch
        [_, loss] = self.sess([self.train, self.loss], feed_dict)
        return loss

    def reconstruct(self, image, batch_size):
        feed_dict = {}
        self.add_randomness(feed_dict, batch_size)
        feed_dict[self.input] = image
        return self.sess(self.output, feed_dict)



