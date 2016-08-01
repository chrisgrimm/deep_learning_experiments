import tensorflow as tf
import numpy as np

class LocExtractor(object):

    def __init__(self, batch_size, num_objects, lstm_size):
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        self.first_lstm_hook = True
        self.num_objects = num_objects
        self.batch_size = batch_size


    def hook(self, inp):
        coords = []
        state = tf.zeros([self.batch_size, self.lstm.state_size])
        for i in range(self.num_objects):
            with tf.variable_scope('lstm') as scope:
                if not (i == 0 and self.first_lstm_hook):
                    scope.reuse_variables()
                output, state = self.lstm(inp, state, scope=scope)
                coord = tf.reshape(self.getCoords(output), [-1, 1, 2])
                coords.append(coord)
        self.first_lstm_hook = False
        output = tf.concat(1, coords)
        flat_output = tf.reshape(output, [-1, self.num_objects*2])
        return flat_output


    def getCoords(self, output):
        w1 = tf.Variable(tf.random_normal([output.get_shape()[1].value, 20]))
        b1 = tf.Variable(tf.constant(0.1, shape=[20]))
        h1 = tf.nn.relu(tf.matmul(output, w1) + b1)
        w2 = tf.Variable(tf.random_normal([20, 2]))
        b2 = tf.Variable(tf.constant(0.0, shape=[2]))
        return tf.nn.tanh(tf.matmul(h1, w2) + b2)


