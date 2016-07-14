import tensorflow as tf
import numpy as np
from models.attend_infer_repeat.SpatialTransformerNetwork import SpatialTransformerNetwork as STN, hook_net, create_localizer_weights, \
    create_weights
from models.attend_infer_repeat.VariationalAutoencoder import log_std_normal_pdf, log_normal_pdf, create_vae_weights, hook_vae_and_sample, hook_vae, log_bernoulli_pmf
#from tf_utils import *
from models.attend_infer_repeat.transformer import transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

class AIR(object):

    def __init__(self, height, width, lstm_units, max_objects, batch_size, name, input):
        self.ih, self.iw = height, width
        self.lstm_u = lstm_units
        self.N = max_objects
        self.vars = {}
        self.batch_size = batch_size
        self.input = input
        self.name = name
        self.setup_network()
        del self.vars['name']


    def extract_where(self, flat_where):
        sub_where = tf.transpose(tf.gather(tf.transpose(flat_where), np.mat('0 0 1; 0 0 2')), perm=[2, 0, 1]) * np.mat('1 0 1; 0 1 1')
        return sub_where

    def copyTo(self, other):
        # copy lstm
        assert other.N == self.N
        for i in range(self.N):
            with tf.variable_scope(other.name + str(i)):
                otherMat = tf.get_variable('Matrix')
                otherBias = tf.get_variable('Bias')
            with tf.variable_scope(self.name + str(i)):
                mat = tf.get_variable('Matrix')
                bias = tf.get_variable('Bias')
            tf.assign(otherMat, mat)
            tf.assign(otherBias, bias)
        # copy vars...
        for name in self.vars.keys():
            tf.assign(other.vars[name], self.vars[name])



    def setup_network(self):
        print 'setting up network!'
        # setup lstm
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)

        state = tf.zeros(shape=[self.batch_size, lstm.state_size])
        # set up zwhat and localizer weights
        self.localizer_weights = create_localizer_weights(self.lstm_u, 20, 3, "localizer_weights")
        self.vars.update(self.localizer_weights)
        self.where_registers = []
        self.what_registers = []
        # step through the lstm.
        for iter in range(self.N):
            # pull something off the lstm
            print self.input.get_shape()
            output, state = lstm(input, state, scope=self.name+str(iter))
            # construct where from lstm output
            where_flat, _ = hook_net(output, self.localizer_weights, [tf.nn.softplus, tf.nn.tanh])
            where = tf.reshape(self.extract_where(where_flat), [-1, 6])
            # use the image itself as the encoding, use small number of pixels because this should be enough
            # to identify the object's type and further forces localization
            capture_size = 5
            x_att = tf.reshape(
                    transformer(tf.reshape(self.input, [-1, self.ih, self.iw, 1]), where, (capture_size, capture_size)),
                               [-1, capture_size, capture_size])
            x_att_flat = tf.reshape(x_att, [-1, capture_size*capture_size])
            # where registers contain the raw transforms as (s, x, y).
            self.where_registers.append(where_flat)
            # what registers contain the "natural encodings of the images"
            self.what_registers.append(x_att_flat)

        # construct dense output
        self.output = tf.concat(1, [self.what_registers[0], self.where_registers[0]])
        for what, where in zip(self.what_registers, self.where_registers)[1:]:
            self.output = tf.concat(1, tf.concat(1, [what, where]))


