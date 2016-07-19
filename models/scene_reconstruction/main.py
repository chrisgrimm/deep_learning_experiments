import tensorflow as tf
from emulator import w, h, getFrames
from RollingBuffer import RollingBuffer
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')
batch_size = 50
frame_buffer_size = 1000

# put initial frames in the buffer
frame_buffer = RollingBuffer(frame_buffer_size)
initial_frames = getFrames(frame_buffer_size)
for frame in initial_frames:
    frame_buffer.add(frame)



# set up input for random frames.
inp = tf.placeholder(tf.float32, shape=[None, 10, h, w])
desired_outs = tf.placeholder(tf.float32, shape=[None, h, w])

lstm = tf.nn.rnn_cell.BasicLSTMCell(h*w)
state = tf.zeros(shape=[1, lstm.state_size])
outs_list = []
for i in range(10):
    with tf.variable_scope('lstm_scope') as scope:
        if i > 0:
            scope.reuse_variables()
        output, state = lstm(tf.reshape(inp[:, i, :, :], [-1, h*w]), state, scope=scope)
output = tf.reshape(output, [-1, h, w])

loss = tf.reduce_mean(tf.pow(output - desired_outs, 2))
train_batch = tf.train.AdamOptimizer().minimize(loss)
print 'here!'
sess = tf.Session()
sess.run(tf.initialize_all_variables())
i = 0
print 'here!'


def make_batch(frame_buffer, batch_size):
    batch = np.zeros((batch_size, 10, h, w))
    for i in range(10):
        batch[:, i, :, :] = frame_buffer.sample(batch_size)
    return batch


while True:
    print i
    feed_dict = {inp: make_batch(frame_buffer, batch_size), desired_outs: frame_buffer.sample(batch_size)}
    [_, bgs] = sess.run([train_batch, output], feed_dict)
    if i % 100 == 0:
        plt.imshow(bgs[0], cmap='Greys_r')
        plt.savefig('./images/background_%s.png' % i)
    i += 1


