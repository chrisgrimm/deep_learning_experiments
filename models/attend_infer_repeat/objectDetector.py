import tensorflow as tf
import numpy as np
from SpatialTransformerNetwork import SpatialTransformerNetwork as STN, hook_net, create_localizer_weights, \
    create_weights
from VariationalAutoencoder import log_std_normal_pdf, log_normal_pdf, create_vae_weights, hook_vae_and_sample, hook_vae, log_bernoulli_pmf
#from tf_utils import *
from transformer import transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random



class AIR(object):

    def __init__(self, sess, height, width, lstm_units, max_objects, batch_size):
        self.ih, self.iw = height, width
        self.batch_size = batch_size
        self.sess = sess
        self.lstm_u = lstm_units
        self.N = max_objects
        self.z_what_random = []
        self.z_where_random = []
        self.z_pres_random = []
        self.inv_where_random = []
        self.slot_images = []
        self.vars = {}
        self.where_size = 6
        self.z_what_size = 40
        self.learning_rate = 0.001
        self.setup_network()
        del self.vars["name"]
        self.saver = tf.train.Saver(self.vars)

    def save(self):
        print "saving"
        self.saver.save(self.sess, "AIRweights.data")
        print "saved"

    def restore(self):
        self.saver.restore(self.sess, "AIRweights.data")
        print "Restored"

    def extract_where(self, flat_where):
        sub_where = tf.transpose(tf.gather(tf.transpose(flat_where), np.mat('0 0 1; 0 0 2')), perm=[2, 0, 1]) * np.mat('1 0 1; 0 1 1')
        return sub_where# - sub_where * np.mat('1 0 0; 0 1 0') + np.mat('1 0 0; 0 1 0')

    def hook_convolution_layers(self, x):
        # conv1
        size = 6
        channels = 1
        filters = 16
        stride = 1
        self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]))
        self.c1 = tf.nn.conv2d(tf.reshape(x, (-1, self.ih, self.iw, 1)), self.w1,
                               strides=[1, stride, stride, 1], padding='SAME')
        self.o1 = tf.nn.relu(tf.add(self.c1, self.b1))

        # conv2
        size = 3
        channels = 16
        filters = 32
        stride = 1
        self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01))
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]))
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME')
        self.o2 = tf.nn.relu(tf.add(self.c2, self.b2))
        # flat
        self.o2_flat = tf.reshape(self.o2, [self.batch_size, filters * self.ih * self.iw])
        return self.o2_flat

    def setup_network(self):
        self.input = tf.placeholder(tf.float32, [None, self.ih * self.iw], name='inp')

        ih, iw = self.ih, self.iw
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        state = tf.zeros(shape=[self.batch_size, lstm.state_size])
        self.x_mean = tf.zeros_like(self.input)
        self.x_std = tf.zeros_like(self.input)
        # self.z_where_weights = create_vae_weights(self.lstm_u, 100, 3, "z_where_weights")
        # self.vars.update(z_where_weights)
        self.z_what_encoder_weights = create_vae_weights(28 ** 2, 500, self.z_what_size, "z_what_encoder_weights")
        self.vars.update(self.z_what_encoder_weights)
        self.z_what_decoder_weights = create_vae_weights(self.z_what_size, 500, 28 ** 2, "z_what_decoder_weights")
        self.vars.update(self.z_what_decoder_weights)
        self.z_pres_weights = create_weights(self.lstm_u, 20, 1, "z_pres_weights")
        self.vars.update(self.z_pres_weights)
        self.localizer_weights = create_localizer_weights(self.lstm_u, 20, 3, "localizer_weights")
        self.vars.update(self.localizer_weights)
        self.background_extraction_weights = create_weights(self.ih * selfiw, 500, 1, "background_extraction_weights")
        self.vars.update(self.background_extraction_weights)
        background = hook_net(self.inpt)
        q_z_given_x = 0
        p_z = 0
        # step through the lstm.
        self.slot_images = []
        convolved = self.hook_convolution_layers(self.input)
        for iter in range(self.N):
            output, state = lstm(self.input, state, scope=str(iter))
            z_pres, _ = hook_net(output, self.z_pres_weights, [tf.nn.softplus, tf.nn.sigmoid])
            where_flat, _ = hook_net(output, self.localizer_weights, [tf.nn.softplus, tf.nn.tanh])

            where = tf.reshape(self.extract_where(where_flat), [-1, 6])
            x_att = tf.reshape(transformer(tf.reshape(self.input, [-1, self.ih, self.iw, 1]), where, (28, 28)),
                               [-1, 28, 28])
            x_att_flat = tf.reshape(x_att, [-1, 28 * 28])

            z_what, log_z_given_x, log_z = hook_vae_and_sample(x_att_flat, self.z_what_encoder_weights,
                                                               'gaussian')
            p_z += log_z
            q_z_given_x += log_z_given_x

            y_att_mean = hook_vae(z_what, self.z_what_decoder_weights, "bernoulli")
            y_att_std = y_att_mean
            bias = 1
            y_att_mean = tf.reshape(y_att_mean, [-1, 28, 28])
            y_att_std = tf.reshape(y_att_std, [-1, 28, 28])
            boundary = np.ones((28, 28))
            boundary[-bias:, :] = 0
            boundary[:bias, :] = 0
            boundary[:, :bias] = 0
            boundary[:, -bias:] = 0
            y_att_mean *= boundary
            y_att_std *= boundary
            inv_where = tf.concat(1, [1.0 / (tf.reshape(where_flat[:, 0], [-1, 1]) + 10**-5),
                                      -tf.reshape(where_flat[:, 1] * 1.0 / (where_flat[:, 0] + 10**-5), [-1,1]),
                                      -tf.reshape(where_flat[:, 2] * 1.0 / (where_flat[:, 0] + 10**-5), [-1,1])])
            inv_where_loc = tf.reshape(self.extract_where(inv_where), [-1, 6])

            y_att_mean = tf.reshape(transformer(tf.reshape(y_att_mean, [-1, 28, 28, 1]), inv_where_loc, (ih, iw)),
                                    [-1, self.ih, self.iw])

            y_att_std = tf.reshape(transformer(tf.reshape(y_att_std, [-1, 28, 28, 1]), inv_where_loc, (ih, iw)),
                                    [-1, self.ih, self.iw])
            new_image = tf.reshape(y_att_mean, [-1, ih * iw]) * z_pres
            self.slot_images.append(new_image)
            self.x_mean = tf.maximum(self.x_mean, new_image)
            self.x_std += tf.reshape(y_att_std, [-1, ih * iw]) * z_pres

        lmbd = tf.abs(tf.Variable(tf.constant(10.0)))
        self.x_std = (1.0 + self.x_std) + 10**-5
        p_x_given_z = log_bernoulli_pmf(self.input, self.x_mean)

        self.loss = - tf.reduce_mean(p_z + p_x_given_z - q_z_given_x)
        self.msq = tf.reduce_mean(tf.pow(self.x_mean - self.input, 2))
        self.lr = tf.placeholder(tf.float32)
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train_batch(self, batch, iter_no):
        #print [x[0].shape for x in batch]
        print "-"*100
        if iter_no % 500 == 0:
            self.learning_rate /= 1.1
        feed_dict = {self.input: batch,
                     self.lr: self.learning_rate}
        values = self.sess.run([self.train, self.msq, self.loss], feed_dict)
        print values
        return values[-1]

    def visualize_result(self, batch, name):
        feed_dict = {self.input: batch,
                     self.lr: self.learning_rate}
        [_, loss, output, s1, s2] = self.sess.run([self.train, self.loss, self.x_mean, self.slot_images[0],
                                                   self.slot_images[1]], feed_dict)
        f, [[ax1, ax2, _], [slot1, slot2, slot3]] = plt.subplots(2, 3)
        index = random.randint(0, len(batch) - 1)
        ax1.imshow(np.reshape(batch[index], (self.ih, self.iw)))
        ax2.imshow(np.reshape(output[index], (self.ih, self.iw)))
        slot1.imshow(np.reshape(s1[index], (self.ih, self.iw)))
        slot2.imshow(np.reshape(s2[index], (self.ih, self.iw)))
        f.savefig("./fig" + "latest" + ".png")
        return loss

    def reconstruct(self, image):
        feed_dict = {self.input: image}
        return self.sess.run(self.x_mean, feed_dict)



