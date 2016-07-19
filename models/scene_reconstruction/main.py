import tensorflow as tf
from emulator import w, h, getFrames
from RollingBuffer import RollingBuffer
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
batch_size = 50
frame_buffer_size = 1000

# put initial frames in the buffer
frame_buffer = RollingBuffer(frame_buffer_size)
initial_frames = getFrames(frame_buffer_size)
for frame in initial_frames:
    frame_buffer.add(frame)



# set up input for random frames.
inp = tf.placeholder(tf.float32, shape=[None, h, w])
desired_outs = tf.placeholder(tf.float32, shape=[None, h, w])

lstm = tf.nn.rnn_cell.BasicLSTMCell(h*w)
state = tf.zeros(shape=[1, lstm.state_size])
outs_list = []
for i in range(batch_size):
    with tf.variable_scope('lstm_scope') as scope:
        if i > 0:
            scope.reuse_variables()
        output, state = lstm(tf.reshape(inp[i, :, :], [1, h*w]), state, scope=scope)
    output = tf.reshape(output, [1, h, w])
    outs_list.append(output)
outs = tf.concat(0, outs_list)

loss = tf.reduce_mean(tf.pow(outs - desired_outs, 2))
train_batch = tf.train.AdamOptimizer().minimize(loss)
print 'here!'
sess = tf.Session()
sess.run(tf.initialize_all_variables())
i = 0
print 'here!'
while True:
    print i
    feed_dict = {inp: frame_buffer.sample(50), desired_outs: frame_buffer.sample(50)}
    [_, bgs] = sess.run([train_batch, outs], feed_dict)
    if i % 100 == 0:
        plt.imshow(bgs[0], cmap='Greys_r')
        plt.savefig('./images/background_%s.png' % i)
    i += 1


