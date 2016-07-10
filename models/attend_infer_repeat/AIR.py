import tensorflow as tf
import numpy as np

from models.spatial_transformer_network.SpatialTransformerNetwork import SpatialTransformerNetwork as STN
from models.spatial_transformer_network.SpatialTransformerNetwork import hookLocalizer, createLocalizerWeights
from models.variational_autoencoder.VariationalAutoencoder import log_bernoulli_pmf, log_normal_pdf, createVAEweights, hookVAE
from tf_utils import *
from transformer import transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self.slot_images = []
        self.vars = dict()
        self.setup_network()
        self.saver = tf.train.Saver(self.vars)

    def save(self):
        self.saver.save(self.sess, "AIRweights.data")

    def restore(self):
        self.saver.restore(self.sess, "AIRweights.data")

    def extract_where(self, flat_where):
        return tf.transpose(tf.gather(tf.transpose(flat_where), np.mat('0 0 1; 0 0 2')), perm=[2, 0, 1]) * np.mat('1 0 1; 0 1 1')

    def setup_network(self):
        x = self.input
        ih, iw = self.ih, self.iw
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_u)
        state = tf.zeros([self.batch_size, lstm.state_size])
        # the 'initial' pres value cant be trainable.
        prev_pres = tf.ones([self.batch_size, 1])
        cum_q_z_pres = cum_q_z_where = cum_q_z_what = cum_p_x_given_z = cum_p_z = tf.zeros([self.batch_size])
        self.output = tf.zeros_like(x)
        where_dims = 3
        z_where_weights = createVAEweights(self.lstm_u, 100, where_dims, 'z_where')
        self.vars.update(z_where_weights)
        z_what_decoder_weights = createVAEweights(28 ** 2, 500, 20, 'z_what_decoder')
        self.vars.update(z_what_decoder_weights)
        z_what_encoder_weights = createVAEweights(20, 500, 28 ** 2, 'z_what_encoder')
        self.vars.update(z_what_encoder_weights)
        z_pres_weights = createVAEweights(self.lstm_u, 100, 1, "z_pres")
        self.vars.update(z_pres_weights)
        localizer_weights = createLocalizerWeights(where_dims, 20, 3, "localizer")
        self.vars.update(localizer_weights)
        # step through the lstm.
        for iter in range(self.N):
            output, state = lstm(x, state, scope=str(iter))
            z_pres_params, hidden_layer = hookVAE(output, z_where_weights, "z_where")
            #z_pres = VAE_realize(z_pres_params, 'gaussian')
            #with tf.variable_scope(str(iter)) as vs:
            #    lstm_variables = dict([(v.name, v) for v in tf.all_variables() if v.name.startswith(vs.name)])
            #    self.vars.update(vars)
            # ENCODER
            # set up z_where params
            z_where_params, hidden_layer = hookVAE(output, z_where_weights, "z_where")
            z_where = VAE_realize(z_where_params, 'gaussian')
            where_flat, hidden_layer = hookLocalizer(z_where, localizer_weights, "localizer")
            where = tf.reshape(self.extract_where(where_flat), [-1, 6])

            # set up z_pres params multiply by old previous value... (if prev pres is zero for an image, it kills the bernoulli distr)
            #z_pres_params, vars = VAE(output, 100, 1, 'bernoulli', prefix='z_pres_%s_' % iter)
            #z_pres_params = z_pres_params[0] * tf.to_float(prev_pres)

            #self.vars.update(vars)
            #z_pres = prev_pres = VAE_realize([z_pres_params], z_pres_random_step, 'bernoulli')
            # set up z_what params

            # use spatially transformed where param to make z_what
            #stn = STN(tf.reshape(x, [-1, ih, iw]), (28, 28))
            #x_att = stn.transform(where)
            x_att = tf.reshape(transformer(tf.reshape(x, [-1, ih, iw, 1]), where, (28, 28)), [-1, 28, 28])
            x_att_flat = tf.reshape(x_att, [-1, 28*28])

            z_what_decoder_params, hidden_layer = hookVAE(x_att_flat, z_what_decoder_weights, "z_what_decoder")
            z_what = VAE_realize(z_what_decoder_params, 'gaussian')
            # update the cumulative q(z|x) and p(z) nodes.
            #cum_q_z_pres += log_bernoulli_pmf(z_pres, z_pres_params)
            # cum_q_z_where += log_normal_pdf(z_where, z_where_params[0], z_where_params[1])
            # cum_q_z_what += log_normal_pdf(z_what, z_what_params[0], z_what_params[1])
            #cum_p_z += log_normal_pdf(z_where, tf.zeros_like(z_where_params[0]), tf.ones_like(z_where_params[1])) + \
            #           log_bernoulli_pmf(z_pres, tf.ones_like(z_pres_params)/2) + \
            #           log_normal_pdf(z_what, tf.zeros_like(z_what_params[0]), tf.ones_like(z_what_params[1]))
            # DECODER
            # decode z_what into image
            print 'moop'
            (y_att_mean, y_att_std), hidden_layer = hookVAE(z_what, z_what_encoder_weights, "z_what_encoder")
            #pres_mask = tf.to_float(tf.tile(tf.reshape(z_pres, [-1, 1, 1]), [1, 28, 28]))
            self.slot_images.append(y_att_mean)# * pres_mask)
            #z_inv_where_params, vars = VAE(z_where, 100, 3, 'gaussian', prefix='z_inv_where_%s_' % iter)
            #self.vars.update(vars)
            # z_inv_where_random_step = tf.placeholder(tf.float32, [self.batch_size, 3])
            # self.inv_where_random.append(z_inv_where_random_step)
            #z_inv_where = VAE_realize(z_inv_where_params, z_inv_where_random_step, 'gaussian')
            #inv_where, vars = connected_layers(z_inv_where, [100, 3], ['tanh', 'tanh'], prefix='inv_localizer_%s_' % iter)
            #self.vars.update(vars)
            inv_where = tf.concat(1, [1.0/(tf.reshape(where_flat[:, 0], [-1, 1]) + 10**-5),
                                      -tf.reshape(where_flat[:, 1] * 1.0/(where_flat[:, 0] + 10**-5), [-1,1]),
                                      -tf.reshape(where_flat[:, 2] * 1.0/(where_flat[:, 0] + 10**-5), [-1,1])])
            inv_where_loc = tf.reshape(self.extract_where(inv_where), [-1, 6])
            #inv_stn = STN(y_att, (ih, iw))
            #y_i = inv_stn.transform(inv_where_loc)
            y_i = tf.reshape(transformer(tf.reshape(y_att_mean, [-1, 28, 28, 1]), inv_where_loc, (ih, iw)), [-1, 40, 40])
            #pres_mask = tf.to_float(tf.tile(tf.reshape(z_pres, [-1, 1, 1]), [1, ih, iw]))
            y_i_masked = y_i# * pres_mask
            self.output += tf.reshape(y_i_masked, [-1, ih * iw])
            # update the p_x_given_z node
            # CHECK THIS LINE: UNCLEAR WHAT TO DO TO P(X|Z) WHEN SHIFT FROM Z_INV_WHERE TO INV_WHERE
            #cum_p_x_given_z += log_normal_pdf(x_att_flat, y_att_flat_params[0], y_att_flat_params[1])
        self.p = p = cum_p_z + cum_p_x_given_z
        self.q = q = cum_q_z_what + cum_q_z_where + cum_q_z_pres
        #self.cum_q_z_what = cum_q_z_what
        #self.cum_q_z_where = cum_q_z_where
        self.cum_q_z_pres = cum_q_z_pres
        self.loss = tf.reduce_mean(tf.reduce_mean(tf.pow(self.output - self.input, 2), reduction_indices=1), reduction_indices=0)
        #self.loss = tf.reduce_mean(-(p - q), reduction_indices=0)
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def add_randomness(self, feed_dict, batch_size):
        for i in range(self.N):
            feed_dict[self.z_what_random[i]] = np.random.normal(size=(batch_size, 20))
            feed_dict[self.z_where_random[i]] = np.random.normal(size=(batch_size, 3))
            feed_dict[self.z_pres_random[i]] = np.random.normal(size=(batch_size, 1))
            feed_dict[self.inv_where_random[i]] = np.random.normal(size=(batch_size, 3))

    def train_batch(self, batch, batch_size, i):
        feed_dict = {}
        #self.add_randomness(feed_dict, batch_size)
        feed_dict[self.input] = batch
        [_, loss, output, s1, s2] = self.sess.run([self.train, self.loss, self.output, self.slot_images[0], self.slot_images[1]], feed_dict)
        if i % 100 == 0:
            f, [[ax1, ax2, _], [slot1, slot2, slot3]] = plt.subplots(2, 3)
            ax1.imshow(np.reshape(batch[0], (40, 40)))
            ax2.imshow(np.reshape(output[0], (40, 40)))
            slot1.imshow(np.reshape(s1[0], (28, 28)))
            slot2.imshow(np.reshape(s2[0], (28, 28)))
            f.savefig('./fig.png')
        return loss

    def reconstruct(self, image, batch_size):
        feed_dict = {}
        #self.add_randomness(feed_dict, batch_size)
        feed_dict[self.input] = image
        return self.sess.run(self.output, feed_dict)



