import tensorflow as tf
import numpy as np
from models.attend_infer_repeat.AIR import AIR
from data_generation import get_batch

sess = tf.Session()

batch_size = 100

x = tf.placeholder(tf.float32, [None, 40, 40])
x_flat = tf.reshape(x, [-1, 40*40])
air = AIR(sess, x_flat, 40, 40, 50, 3, batch_size)
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(sess, air.vars)

num_batches = 1000
for i in range(num_batches):
    batch = get_batch(batch_size)
    loss = air.train_batch(batch, batch_size)
    print loss
    if i % 100 == 0:
        saver.save('./vars')