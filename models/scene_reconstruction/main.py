import tensorflow as tf
from emulator import getFrames
from RollingBuffer import RollingBuffer
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 50
frame_buffer_size = 1000
w, h = 84, 84
# put initial frames in the buffer
frame_buffer = RollingBuffer(frame_buffer_size)
initial_frames = getFrames(frame_buffer_size)
for frame in initial_frames:
    frame_buffer.add(frame)



# set up input for random frames.
inp = tf.placeholder(tf.float32, shape=[None, 10, h, w])
desired_outs = tf.placeholder(tf.float32, shape=[None, h, w])
lstm = tf.nn.rnn_cell.BasicLSTMCell(256)
state = tf.zeros(shape=[batch_size, lstm.state_size])
outs_list = []
for i in range(10):
    with tf.variable_scope('lstm_scope') as scope:
        if i > 0:
            scope.reuse_variables()
        tmp = tf.reshape(inp[:, i, :, :], [-1, h*w])
        print tmp.get_shape()
        output, state = lstm(tmp, state, scope=scope)
lstm_output = output
W1 = tf.Variable(tf.random_normal([256, 256]))
b1 = tf.Variable(tf.constant(0.5, shape=[256]))
W2 = tf.Variable(tf.random_normal([256, w*h]))
b2 = tf.Variable(tf.constant(0.1, shape=[w*h]))
hidden = tf.nn.relu(tf.matmul(lstm_output, W1) + b1)
output = 255.0 * tf.nn.sigmoid(tf.matmul(hidden, W2) + b2)
output = tf.reshape(output, [-1, h, w])


loss = tf.reduce_mean(tf.pow(output - 255.0*desired_outs, 2))
train_batch = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
print 'here!'
print w, h

#sess = tf.Session()

#sess.run(tf.initialize_all_variables())
i = 0
print 'here!'


def make_batch(frame_buffer, batch_size):
    batch = np.zeros((batch_size, 10, h, w))
    for i in range(10):
        batch[:, i, :, :] = frame_buffer.sample(batch_size)
    return batch


#while True:
    #inp_data = make_batch(frame_buffer, batch_size)
    #out_data = frame_buffer.sample(batch_size)
    #feed_dict = {inp: inp_data, desired_outs: out_data}
    #[_, bgs, _loss] = sess.run([train_batch, output, loss], feed_dict)
    #print i, _loss
    #if i % 100 == 0:
    #    print i, loss
    #    plt.imshow(bgs[0], cmap='Greys_r')
    #    plt.savefig('./images/background_%s.png' % i)
    #    plt.savefig('./images/background.png')
    #    plt.close()
    #    plt.imshow(out_data[0])
    #    plt.savefig('./images/actual.png')
    #i += 1
    #getFrames(1)


res = np.array(frame_buffer.buffer)
res = np.median(res, 0)
plt.imshow(res)
plt.show()
raw_input()

