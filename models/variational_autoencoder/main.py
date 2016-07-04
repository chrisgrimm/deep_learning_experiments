import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from models.variational_autoencoder.VariationalAutoencoder import VAE_realize, VAE, log_normal_pdf
sess = tf.Session()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# set up autoencoder graph
x = tf.placeholder(tf.float32, [None, 28*28])
x_ = x + 10**-2
encoding_params, enc_vars = VAE(x_, 500, 20, 'gaussian', prefix='enc_')
z_random = tf.placeholder(tf.float32, [None, 20])
encoding = VAE_realize(encoding_params, z_random, 'gaussian')
recon_params, dec_vars = VAE(encoding, 500, 28*28, 'gaussian', prefix='dec_')
#recon_random = tf.placeholder(tf.float32, [None, 28*28])
#recon = VAE_realize(recon_params, recon_random, 'gaussian')
# set up loss function and training
#q = log_normal_pdf(encoding, encoding_params[0], encoding_params[1])
#p = log_normal_pdf(x, recon_params[0], recon_params[1]) + \
#    log_normal_pdf(encoding, tf.zeros_like(encoding_params[0]), tf.ones_like(encoding_params[1]))
#loss = tf.reduce_mean(-(p - q), reduction_indices=0)
images = x_
[p_mu, p_sigma] = recon_params
[q_mu, q_sigma] = encoding_params
exp_part = -0.5 * tf.reduce_sum(tf.mul(tf.mul((images - p_mu), (1.0/(tf.pow(p_sigma,2)))), (images - p_mu)), reduction_indices=1)
normal_p_x_given_z = (-28**2 / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(p_sigma + 10**-10), reduction_indices=1) + exp_part
KL_term = 0.5 * tf.reduce_sum(1.0 + 2*tf.log(q_sigma + 10**-10) - tf.pow(q_mu, 2) - tf.pow(q_sigma, 2), reduction_indices=1)
loss = -1*tf.reduce_mean(KL_term + normal_p_x_given_z, reduction_indices=0)
#loss = tf.reduce_mean(tf.reduce_mean(tf.pow(x - recon_params[0], 2), reduction_indices=1), reduction_indices=0)
train = tf.train.AdamOptimizer(0.001).minimize(loss)

params = {}
params.update(enc_vars)
params.update(dec_vars)
saver = tf.train.Saver(params)
for name, item in params.items():
    print name, item
sess.run(tf.initialize_all_variables())
should_train = False
if should_train:
    batch_size = 100
    for i in range(20):
        avg_loss = []
        for y in range(len(mnist.train.images)/batch_size):
            data = mnist.train.next_batch(batch_size)
            feed_dict = {x: data[0], z_random: np.random.normal(size=(batch_size, 20)),
                         }#recon_random: np.random.normal(size=(batch_size, 28*28))}
            _, ll = sess.run([train, loss], feed_dict=feed_dict)
            avg_loss.append(ll)
        print i
        print 'average loss:', np.mean(avg_loss)

    saver.save(sess, './vars')
else:
    saver.restore(sess, './vars')
    batch_size = 1
    data = mnist.train.next_batch(batch_size)
    feed_dict = {x: data[0],
             z_random: np.random.normal(size=(batch_size, 20)),
                 }#recon_random: np.random.normal(size=(batch_size, 28*28))}
    res_mu = sess.run(recon_params[0], feed_dict)
    f, [ax2, ax3] = plt.subplots(1, 2)
    #ax1.imshow(np.reshape(res, [28, 28]))
    ax2.imshow(np.reshape(res_mu, [28, 28]))
    ax3.imshow(np.reshape(data[0], [28, 28]))

    f.show()
    raw_input()