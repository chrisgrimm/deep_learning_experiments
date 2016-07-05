import tensorflow as tf
import numpy as np
from models.attend_infer_repeat.AIR import AIR
from data_generation import get_batch

sess = tf.Session()

batch_size = 100

x = tf.placeholder(tf.float32, [None, 40, 40])
x_flat = tf.reshape(x, [-1, 40*40])
air = AIR(sess, x_flat, 40, 40, 3, 2, batch_size)
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(air.vars)

num_batches = 10000000
for i in range(0, num_batches):
    print i
    batch = get_batch(batch_size)[0]
    loss = air.train_batch(batch, batch_size, i)
    print loss
