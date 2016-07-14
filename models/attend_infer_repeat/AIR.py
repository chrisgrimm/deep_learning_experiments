import tensorflow as tf
import numpy as np
from SpatialTransformerNetwork import SpatialTransformerNetwork as STN, hook_localizer, create_localizer_weights
from VariationalAutoencoder import log_normal_pdf, create_vae_weights, hook_vae_and_sample, hook_vae, log_bernoulli_pmf
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

    def setup_network(self):
        self.input = tf.placeholder(tf.float32, [None, 40 * 40])
        ih, iw = self.ih, self.iw
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        state = tf.zeros(shape=[self.batch_size, lstm.state_size])
        self.x_mean = tf.zeros_like(self.input)
        self.x_std = tf.zeros_like(self.input)
        where_dims = 3
        z_where_weights = create_vae_weights(self.lstm_u, 100, where_dims, "z_where_weights")
        self.vars.update(z_where_weights)
        z_what_encoder_weights = create_vae_weights(28 ** 2, 500, 20, "z_what_encoder_weights")
        self.vars.update(z_what_encoder_weights)
        z_what_decoder_weights = create_vae_weights(20, 500, 28 ** 2, "z_what_decoder_weights")
        self.vars.update(z_what_decoder_weights)
        z_pres_weights = create_vae_weights(self.lstm_u, 20, 1, "z_pres_weights")
        self.vars.update(z_pres_weights)
        localizer_weights = create_localizer_weights(where_dims, 20, 3, "localizer_weights")
        self.vars.update(localizer_weights)
        cum_z_pres = 1
        q_z_given_x = 0
        p_z_what = 0
        p_z_where = 0
        p_z_pres = 0
        # step through the lstm.
        overlap = 0
        self.slot_images = []
        for iter in range(self.N):
            output, state = lstm(self.input, state, scope=str(iter))

            z_pres, log_z_given_x, log_z = hook_vae_and_sample(output, z_pres_weights, "bernoulli")
            z_pres = 1
            q_z_given_x += log_z_given_x * cum_z_pres
            p_z_pres += log_z * cum_z_pres
            cum_z_pres *= z_pres

            z_where, log_z_given_x, log_z = hook_vae_and_sample(output, z_where_weights, "gaussian")
            q_z_given_x += log_z_given_x * cum_z_pres
            p_z_where += log_z * cum_z_pres
            where_flat, hidden_layer = hook_localizer(z_where, localizer_weights)

            where = tf.reshape(self.extract_where(where_flat), [-1, 6])

            x_att = tf.reshape(transformer(tf.reshape(self.input, [-1, ih, iw, 1]), where, (28, 28)), [-1, 28, 28])
            x_att_flat = tf.reshape(x_att, [-1, 28*28])

            z_what, log_z_given_x, log_z = hook_vae_and_sample(x_att_flat, z_what_encoder_weights, 'gaussian')
            q_z_given_x += log_z_given_x * cum_z_pres
            p_z_what += log_z * cum_z_pres

            y_att_mean = hook_vae(z_what, z_what_decoder_weights, "bernoulli")
            self.slot_images.append(y_att_mean)
            inv_where = tf.concat(1, [1.0/(tf.reshape(where_flat[:, 0], [-1, 1]) + 10**-5),
                                      -tf.reshape(where_flat[:, 1] * 1.0/(where_flat[:, 0] + 10**-5), [-1,1]),
                                      -tf.reshape(where_flat[:, 2] * 1.0/(where_flat[:, 0] + 10**-5), [-1,1])])
            inv_where_loc = tf.reshape(self.extract_where(inv_where), [-1, 6])

            y_att_mean = tf.reshape(transformer(tf.reshape(y_att_mean, [-1, 28, 28, 1]), inv_where_loc, (ih, iw)),
                                    [-1, 40, 40])
            new_image = tf.reshape(y_att_mean, [-1, ih * iw]) * cum_z_pres
            #overlap += tf.reduce_sum(new_image * self.x_mean)
            self.x_mean  += new_image

            # y_att_std = tf.reshape(transformer(tf.reshape(y_att_std, [-1, 28, 28, 1]), inv_where_loc, (ih, iw)),
            #                        [-1, 40, 40])
            # self.x_std += tf.reshape(y_att_std, [-1, ih * iw]) * cum_z_pres

        p_z_pres += 0.5 * cum_z_pres
        p_z = p_z_pres + p_z_what + p_z_where
        p_x_given_z = log_bernoulli_pmf(self.input, self.x_mean)
        self.q_z_given_x = q_z_given_x
        self.p_z = p_z
        self.p_x_given_z = p_x_given_z
        self.global_step = tf.Variable(initial_value = 1.0)
        #self.loss = - tf.reduce_mean(p_z + p_x_given_z - q_z_given_x)
        self.loss = tf.reduce_mean(tf.reduce_mean(tf.pow(self.input - self.x_mean, 2), reduction_indices=1),
                                   reduction_indices=0)  #/ self.global_step
        #self.loss =  - tf.reduce_mean(p_x_given_z)
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def add_randomness(self, feed_dict, batch_size):
        for i in range(self.N):
            feed_dict[self.z_what_random[i]] = np.random.normal(size=(batch_size, 20))
            feed_dict[self.z_where_random[i]] = np.random.normal(size=(batch_size, 3))
            feed_dict[self.z_pres_random[i]] = np.random.normal(size=(batch_size, 1))
            feed_dict[self.inv_where_random[i]] = np.random.normal(size=(batch_size, 3))

    def train_batch(self, batch):
        feed_dict = {self.input: batch}
        _, loss = self.sess.run([self.train, self.loss], feed_dict)
        return loss

    def visualize_result(self, batch, name):
        feed_dict = {self.input: batch}
        [_, loss, output, s1, s2] = self.sess.run([self.train, self.loss, self.x_mean, self.slot_images[0],
                                                   self.slot_images[1]], feed_dict)
        f, [[ax1, ax2, _], [slot1, slot2, slot3]] = plt.subplots(2, 3)
        index = random.randint(0, len(batch) - 1)
        ax1.imshow(np.reshape(batch[index], (40, 40)))
        ax2.imshow(np.reshape(output[index], (40, 40)))
        slot1.imshow(np.reshape(s1[index], (28, 28)))
        slot2.imshow(np.reshape(s2[index], (28, 28)))
        f.savefig("./fig" + name + ".png")
        return loss

    def reconstruct(self, image):
        print image
        feed_dict = {self.input: image}
        return self.sess.run(self.x_mean, feed_dict)



