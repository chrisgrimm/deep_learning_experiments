import tensorflow as tf
import numpy as np
from models.spatial_transformer_network.SpatialTransformerNetwork import SpatialTransformerNetwork as STN, fully_connected_localizer
from models.variational_autoencoder.VariationalAutoencoder import VAE, VAE_realize
from tf_utils import *

class AIR(object):


    def __init__(self, input_height, input_width, lstm_units, max_objects):
        self.ih, self.iw = input_height, input_width
        self.lstm_u = lstm_units
        self.N = max_objects

    def setup_network(self):
        self.input = tf.placeholder(tf.float32, [None, self.ih, self.iw])
        batch_size = tf.shape(self.input)[0]
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        self.init_state = np.zeros([batch_size, self.lstm_u])
        state = self.init_state
        # step through the lstm.
        for iter in range(self.N):
            output, state = self.lstm(self.input, state)
            # set up z_where params
            z_where_params = VAE(output, 100, 3, 'gaussian')
            z_where_random_step = tf.placeholder(tf.float32, [batch_size, 3])
            z_where = VAE_realize(z_where_params, z_where_random_step, 'gaussian')
            # set up z_pres params
            z_pres_params = VAE(100, 1, 'bernoulli')
            z_pres_random_step = tf.placeholder(tf.float32, [batch_size, 1])
            z_pres = VAE_realize(z_pres_params, z_pres_random_step, 'bernoulli')
            pres = connected_layers(output, [100, 1], ['relu', 'sigmoid'])
            # set up z_what params
            where_flat = connected_layers(z_where, [100, 3], ['tanh', 'tanh'])
            where = tf.gather(where_flat, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            stn = STN(self.input, (28, 28))
            x_att = stn.transform(where)
            what_params = VAE(x_att, 500, 20, 'gaussian')
            what_random_step = tf.placeholder(tf.float32, [batch_size, 20])
            z_what = VAE_realize(what_params, what_random_step, 'gaussian')
            y_att_flat_params = VAE(z_what, 500, 28*28, 'gaussian')
            y_att = tf.reshape(y_att_flat_params[0], [-1, 28, 28])
            inv_where_flat = connected_layers(z_where, [100, 3], ['tanh', 'tanh'])
            inv_where = tf.gather(inv_where_flat, np.mat('0 0 1; 0 0 2')) * np.mat('1 0 1; 0 1 1')
            inv_stn = STN(y_att, (40, 40))
            y_i = inv_stn.transform(inv_where)
            pres_mask = tf.tile(z_pres, [1, 40, 40])
            y_i_masked = y_i * pres_mask








