import tensorflow as tf
import numpy as np
import random


def log_std_normal_pdf(x):
    dim = x.get_shape()[1].value
    return (-dim / 2.0) * tf.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.pow(x, 2), reduction_indices=1)


def log_normal_pdf(x, mu, diag_sigmas):
    D = mu.get_shape()[1].value
    variance = tf.pow(diag_sigmas, 2) + 10**-6
    exp_part = -0.5 * tf.reduce_sum((x - mu) * (1.0/variance) * (x - mu), reduction_indices=1)
    return (-D / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(tf.log(variance), reduction_indices=1) + exp_part


def log_bernoulli_pmf(x, p1):
    return tf.reduce_sum(tf.log(p1 * x + (1 - p1) * (1 - x) + 10**-5), reduction_indices=1)


def create_vae_weights(input_dim, hidden_size, code_size, p):
    weights = {p + "W1": tf.Variable(tf.random_normal(shape=[input_dim, hidden_size], stddev=0.01)),
               p + 'b1': tf.Variable(tf.constant(0.1, shape=[hidden_size])),
               p + 'W2_mu': tf.Variable(tf.random_normal(shape=[hidden_size, code_size], stddev=0.01)),
               p + 'b2_mu': tf.Variable(tf.constant(-1.1, shape=[code_size])),
               p + 'W2_sigma': tf.Variable(tf.random_normal(shape=[hidden_size, code_size], stddev=0.01)),
               p + 'b2_sigma': tf.Variable(tf.constant(1.0, shape=[code_size])),
               "name": p}
    return weights

def hook_vae_and_sample(x, weights, rv_type):
    if rv_type == "gaussian":
        q_mu, q_sigma = hook_vae(x, weights, rv_type)
        outcome = q_mu + q_sigma * tf.random_normal(tf.shape(q_mu))
        log_z_given_x = log_normal_pdf(outcome, q_mu, q_sigma)
        log_z = log_std_normal_pdf(outcome)
        return outcome, log_z_given_x, log_z
    elif rv_type == "final_gaussian":
        q_mu, q_sigma = hook_vae(x, weights, rv_type)
        outcome = q_mu + q_sigma * tf.random_normal(tf.shape(q_mu))
        log_z_given_x = log_normal_pdf(outcome, q_mu, q_sigma)
        log_z = log_std_normal_pdf(outcome)
        return outcome, log_z_given_x, log_z
    elif rv_type == "bernoulli":
        q_mu = hook_vae(x, weights, rv_type)
        outcome = tf.to_float(tf.random_uniform(tf.shape(q_mu)) < q_mu)
        log_z_given_x = log_bernoulli_pmf(outcome, q_mu)
        log_z = 0.5
        return outcome, log_z_given_x, log_z
    else:
        raise Exception('Unrecognized rv_type')





def hook_vae(x, weights, rv_type):
    p = weights["name"]
    # also sigmoid seems to work better than relu. gradient was exploding before.
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[p + "W1"]), weights[p + "b1"]))
    # no idea why this value works as well as it does. should probably change it later. (0.25)
    if rv_type == 'gaussian':
        q_mu = tf.add(tf.matmul(h1, weights[p + "W2_mu"]), weights[p + "b2_mu"])
        q_sigma = tf.nn.softplus(tf.add(tf.matmul(h1, weights[p + "W2_sigma"]), weights[p + "b2_sigma"]))
        return q_mu, q_sigma
    elif rv_type == "final_gaussian":
        q_mu = tf.nn.sigmoid(tf.add(tf.matmul(h1, weights[p + "W2_mu"]), weights[p + "b2_mu"]))
        q_sigma = tf.nn.tanh(tf.add(tf.matmul(h1, weights[p + "W2_sigma"]), weights[p + "b2_sigma"]))
        return q_mu, q_sigma
    elif rv_type == 'bernoulli':
        q_mu = tf.nn.sigmoid(tf.add(tf.matmul(h1, weights[p + "W2_mu"]), weights[p + "b2_mu"]))
        return q_mu
    else:
        raise Exception('Unrecognized rv_type')



    #exp_part = -0.5 * tf.reduce_sum(tf.mul(tf.mul((input - p_mu), (1.0/(tf.pow(p_sigma,2)))), (input - p_mu)), reduction_indices=1)
    #normal_p_x_given_z = (-self.id / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(p_sigma), reduction_indices=1) + exp_part
    #KL_term = 0.5 * tf.reduce_sum(1.0 + tf.log(tf.pow(self.q_sigma, 2) + 10**-10) - tf.pow(self.q_mu, 2) - tf.pow(self.q_sigma, 2), reduction_indices=1)
    #self.loss = -1*tf.reduce_mean(self.KL_term + self.normal_p_x_given_z, reduction_indices=0)
    # adam optimizer over loss function
    #self.global_step = tf.Variable(tf.constant(0), trainable=False)
    #self.run_batch = tf.train.AdamOptimizer(0.001).minimize(self.loss, global_step=self.global_step)
