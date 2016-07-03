import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from models.variational_autoencoder.VariationalAutoencoder import VAE_realize, VAE, log_normal_pdf
sess = tf.Session()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# set up autoencoder graph
x = tf.placeholder(tf.float32, [None, 28*28])
hidden_params, enc_vars = VAE(x, 500, 20, 'gaussian', prefix='enc')
z_random = tf.placeholder(tf.float32, [None, 20])
hidden = VAE_realize(hidden_params, z_random, 'gaussian')
recon_params, dec_vars = VAE(hidden, 500, 28*28, 'gaussian', prefix='dec')
recon_random = tf.placeholder(tf.float32, [None, 28*28])
recon = VAE_realize(recon_params, recon_random, 'gaussian')
# set up loss function and training
q = log_normal_pdf(hidden, hidden_params[0], hidden_params[1])
p = log_normal_pdf(recon, recon_params[0], recon_params[1]) + \
    log_normal_pdf(hidden, tf.zeros_like(hidden_params[0]), tf.ones_like(hidden_params[1]))
loss = -tf.reduce_mean((p - q), reduction_indices=0)
global_step =
train = tf.train.AdamOptimizer().minimize(loss)

params = {}
params.update(enc_vars)
params.update(dec_vars)
saver = tf.train.Saver(params)

sess.run(tf.initialize_all_variables())
# train for 75 batches of 50
batch_size = 50
for i in range(75):
    avg_loss = []
    for y in range(len(mnist.train.images)/batch_size):
        data = mnist.train.next_batch(batch_size)
        feed_dict = {x: data[0], z_random: np.random.random((batch_size, 20)), recon_random: np.random.random((batch_size, 28*28))}
        _, ll = sess.run([train, loss], feed_dict=feed_dict)
        avg_loss.append(ll)
    print i
    print 'average loss:', np.mean(avg_loss)

saver.save(sess, './vars')