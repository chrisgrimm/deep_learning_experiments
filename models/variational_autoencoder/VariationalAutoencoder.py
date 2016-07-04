import tensorflow as tf
import numpy as np

def log_normal_pdf(x, mu, diag_sigmas):
    D = mu.get_shape()[1].value
    exp_part = -0.5 * tf.reduce_sum((x - mu) * (1.0/(tf.pow(diag_sigmas,2))) * (x - mu), reduction_indices=1)
    return (-D / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(diag_sigmas), reduction_indices=1) + exp_part

def log_bernoulli_pmf(x, p1):
    return tf.log(p1 * tf.float32(x == 1) + (1 - p1) * tf.float32(x == 0))

def VAE(input, hiddenSize, codeSize, rvType, prefix=''):
    p = prefix
    vars = dict()
    input_dim = input.get_shape()[1].value
    # encoding layers
    W1 = vars[p+'W1'] = tf.Variable(tf.random_normal(shape=[input_dim, hiddenSize], stddev=0.01))
    b1 = vars[p+'b1'] = tf.Variable(tf.constant(0.1, shape=[hiddenSize]))
    h1 = tf.nn.relu(tf.add(tf.matmul(input, W1), b1))
    W2_mu = vars[p+'W2_mu'] = tf.Variable(tf.random_normal(shape=[hiddenSize, codeSize], stddev=0.01))
    b2_mu = vars[p+'b2_mu'] = tf.Variable(tf.constant(0.1, shape=[codeSize]))
    W2_sigma = vars[p+'W2_sigma'] = tf.Variable(tf.random_normal(shape=[hiddenSize, codeSize], stddev=0.01))
    b2_sigma = vars[p+'b2_sigma'] = tf.Variable(tf.constant(0.1, shape=[codeSize]))
    if rvType == 'gaussian':
        q_mu = tf.add(tf.matmul(h1, W2_mu), b2_mu)
        q_sigma = tf.abs(tf.add(tf.matmul(h1, W2_sigma), b2_sigma)) + 10**-10
        return [q_mu, q_sigma], vars
    elif rvType == 'bernoulli':
        q_p1 = tf.sigmoid(tf.add(tf.matmul(h1, W2_mu), b2_mu))
        return [q_p1], vars
    else:
        raise Exception('Unrecognized rvType.')

def VAE_realize(params, random_placeholder, rvType):
    if rvType == 'gaussian':
        return params[0] + tf.mul(params[1], random_placeholder)
    elif rvType == 'bernoulli':
        return tf.int32(random_placeholder < params[0])
    else:
        raise Exception('Unrecognized rvType')




    #exp_part = -0.5 * tf.reduce_sum(tf.mul(tf.mul((input - p_mu), (1.0/(tf.pow(p_sigma,2)))), (input - p_mu)), reduction_indices=1)
    #normal_p_x_given_z = (-self.id / 2)*tf.log(2*np.pi) - 0.5*tf.reduce_sum(2*tf.log(p_sigma), reduction_indices=1) + exp_part
    #KL_term = 0.5 * tf.reduce_sum(1.0 + tf.log(tf.pow(self.q_sigma, 2) + 10**-10) - tf.pow(self.q_mu, 2) - tf.pow(self.q_sigma, 2), reduction_indices=1)
    #self.loss = -1*tf.reduce_mean(self.KL_term + self.normal_p_x_given_z, reduction_indices=0)
    # adam optimizer over loss function
    #self.global_step = tf.Variable(tf.constant(0), trainable=False)
    #self.run_batch = tf.train.AdamOptimizer(0.001).minimize(self.loss, global_step=self.global_step)
