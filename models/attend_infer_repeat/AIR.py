import tensorflow as tf
import numpy as np
from models.spatial_transformer_network.SpatialTransformerNetwork import SpatialTransformerNetwork as STN
from models.variational_autoencoder.VariationalAutoencoder import VAE, VAE_realize, log_bernoulli_pmf, log_normal_pdf
from tf_utils import *

class AIR(object):

    def __init__(self, x, input_height, input_width, lstm_units, max_objects):
        self.ih, self.iw = input_height, input_width
        self.lstm_u = lstm_units
        self.N = max_objects

    def setup_network(self, x):
        x = tf.placeholder(tf.float32, [None, self.ih, self.iw])
        batch_size = tf.shape(x)[0]
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        self.init_state = np.zeros([batch_size, self.lstm_u])
        state = self.init_state
        # the 'initial' pres value cant be trainable.
        prev_pres = tf.Variable(tf.ones([batch_size, 1]), trainable=False)
        cum_q_z_pres = tf.Variable(tf.zeros([batch_size, 1]), trainable=False)
        cum_q_z_where = tf.Variable(tf.zeros([batch_size, 1]), trainable=False)
        cum_q_z_what = tf.Variable(tf.zeros([batch_size, 1]), trainable=False)
        cum_p_x_given_z = tf.Variable(tf.zeros([batch_size, 1]), trainable=False)
        cum_p_z = tf.Variable(tf.zeros([batch_size, 1]), trainable=False)
        y = tf.zeros_like(x)
        # step through the lstm.
        for iter in range(self.N):
            output, state = self.lstm(x, state)
            # set up z_where params
            z_where_params = VAE(output, 100, 3, 'gaussian')
            z_where_random_step = tf.placeholder(tf.float32, [batch_size, 3])
            z_where = VAE_realize(z_where_params, z_where_random_step, 'gaussian')
            cum_p_z += log_normal_pdf(z_where, tf.ones_like(z_where_params[0]), tf.ones_like(z_where_params[1]))
            cum_q_z_where += log_normal_pdf(x, z_where_params[0], z_where_params[1])
            # set up z_pres params
            # multiply by old previous value... (if prev pres is zero for an image, it kills the bernoulli distr)
            z_pres_params = VAE(output, 100, 1, 'bernoulli')[0] * prev_pres
            cum_q_z_pres += log_bernoulli_pmf(x, z_pres_params)
            z_pres_random_step = tf.placeholder(tf.float32, [batch_size, 1])
            # move the prev_pres variable forward
            z_pres = prev_pres = VAE_realize([z_pres_params], z_pres_random_step, 'bernoulli')
            cum_p_z += log_bernoulli_pmf(z_pres, tf.ones_like(z_pres_params))
            # set up z_what params
            where_flat = connected_layers(z_where, [100, 3], ['tanh', 'tanh'])
            where = tf.gather(where_flat, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            stn = STN(x, (28, 28))
            x_att = stn.transform(where)
            z_what_params = VAE(x_att, 500, 20, 'gaussian')
            z_what_random_step = tf.placeholder(tf.float32, [batch_size, 20])
            z_what = VAE_realize(z_what_params, z_what_random_step, 'gaussian')
            cum_p_z += log_normal_pdf(z_what, tf.ones_like(z_what_params[0]), tf.ones_like(z_what_params[1]))
            cum_q_z_what += log_normal_pdf(x, z_what_params[0], z_what_params[1])
            y_att_flat_params = VAE(z_what, 500, 28*28, 'gaussian')
            cum_p_x_given_z += log_normal_pdf(x, y_att_flat_params[0], y_att_flat_params[1])
            y_att = tf.reshape(y_att_flat_params[0], [-1, 28, 28])
            inv_where_flat = connected_layers(z_where, [100, 3], ['tanh', 'tanh'])
            inv_where = tf.gather(inv_where_flat, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            inv_stn = STN(y_att, (40, 40))
            y_i = inv_stn.transform(inv_where)
            pres_mask = tf.tile(z_pres, [1, 40, 40])
            y_i_masked = y_i * pres_mask
            y += y_i_masked





