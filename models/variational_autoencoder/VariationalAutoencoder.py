import tensorflow as tf
import numpy as np


def log_std_normal_pdf(x):
    return (-x.get_shape()[1] / 2) * tf.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.pow(x, 2), reduction_indices=1)


def log_normal_pdf(x, mu, diag_sigmas):
    D = mu.get_shape()[1].value
    exp_part = -0.5 * tf.reduce_sum((x - mu) * (1.0/(tf.pow(diag_sigmas,2))) * (x - mu), reduction_indices=1)
    return (-D / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(diag_sigmas), reduction_indices=1) + exp_part


def log_bernoulli_pmf(x, p1):
    return tf.reduce_sum(tf.log(p1 * x + (1 - p1) * (1 - x) + 10**-5), reduction_indices=1)


def create_vae_weights(input_dim, hidden_size, code_size, p=""):
    weights = {p + "W1": tf.Variable(tf.random_normal(shape=[input_dim, hidden_size], stddev=0.01)),
               p + 'b1': tf.Variable(tf.constant(0.1, shape=[hidden_size])),
               p + 'W2_mu': tf.Variable(tf.random_normal(shape=[hidden_size, code_size], stddev=0.01)),
               p + 'b2_mu': tf.Variable(tf.constant(0.1, shape=[code_size])),
               p + 'W2_sigma': tf.Variable(tf.random_normal(shape=[hidden_size, code_size], stddev=0.01)),
               p + 'b2_sigma': tf.Variable(tf.constant(0.1, shape=[code_size]))}
    return weights


def hook_vae(x, weights, p=""):
    # also sigmoid seems to work better than relu. gradient was exploding before.
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[p + "W1"]), weights[p + "b1"]))
    q_mu = tf.add(tf.matmul(h1, weights[p + "W2_mu"]), weights[p + "b2_mu"])
    # no idea why this value works as well as it does. should probably change it later. (0.25)
    q_sigma = tf.abs(tf.add(tf.matmul(h1, weights[p + "W2_sigma"]), weights[p + "b2_sigma"])) + 0.1
    return [q_mu, q_sigma], h1


def vae_realize(params, rv_type):
    random_placeholder = tf.random_normal(tf.shape(params[0]))

    if rv_type == 'gaussian':
        outcome = params[0] + tf.mul(params[1], random_placeholder)
        log_z_given_x = log_normal_pdf(outcome, params[0], params[1])
        log_z = log_std_normal_pdf(outcome)
        return outcome, log_z_given_x, log_z
    elif rv_type == 'bernoulli':
        outcome = tf.to_int32(random_placeholder < params[0])
        log_z_given_x = log_bernoulli_pmf(outcome, params[0])
        log_z = 0.5
        return outcome, log_z_given_x, log_z
    else:
        raise Exception('Unrecognized rv_type')



    #exp_part = -0.5 * tf.reduce_sum(tf.mul(tf.mul((input - p_mu), (1.0/(tf.pow(p_sigma,2)))), (input - p_mu)), reduction_indices=1)
    #normal_p_x_given_z = (-self.id / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(p_sigma), reduction_indices=1) + exp_part
    #KL_term = 0.5 * tf.reduce_sum(1.0 + tf.log(tf.pow(self.q_sigma, 2) + 10**-10) - tf.pow(self.q_mu, 2) - tf.pow(self.q_sigma, 2), reduction_indices=1)
    #self.loss = -1*tf.reduce_mean(self.KL_term + self.normal_p_x_given_z, reduction_indices=0)
    # adam optimizer over loss function
    #self.global_step = tf.Variable(tf.constant(0), trainable=False)
    #self.run_batch = tf.train.AdamOptimizer(0.001).minimize(self.loss, global_step=self.global_step)
