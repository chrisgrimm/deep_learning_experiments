import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer import transformer
from emulator import getFrames

# make objects
# set up lstm
batch_size = 50
height, width = 84, 84
inp = tf.placeholder(tf.float32, [batch_size, height, width])
inp_fed = inp
def extract_where(flat_where):
    sub_where = tf.transpose(tf.gather(tf.transpose(flat_where), np.mat('0 0 1; 0 0 2')),
                             perm=[2, 0, 1]) * np.mat('1 0 1; 0 1 1')
    return sub_where


def makeLocalizer(lstm_output):
    #w1 = tf.Variable(tf.random_normal([5, 5, 1, 5]))
    #b1 = tf.Variable(tf.constant(0.5, shape=[5]))
    #c1 = tf.nn.relu(tf.nn.conv2d(lstm_output, w1, [1, 1, 1, 1], 'SAME') + b1)
    #p1 = tf.nn.max_pool(c1, [1, 4, 4, 1], [1, 4, 4, 1])
    # 21 x 21 x 5
    #w2 = tf.Variable(tf.random_normal([5, 5, 5, 10]))
    #b2 = tf.Variable(tf.constant(0.5, shape=[10]))
    #c2 = tf.nn.relu(tf.nn.conv2d(p1, w2, [1, 1, 1, 1], 'SAME') + b2)
    #p2 = tf.nn.max_pool(c2, [1, 4, 4, 1], [1, 4, 4, 1])
    #height, width = p2.get_shape()[1].value, p2.get_shape()[2].value
    # ~ 4 x 4 x 10
    ##flat = tf.reshape(-1, height * width * 10)
    #w3 = tf.Variable(tf.random_normal([height * width * 10, 100]))
    #b3 = tf.Variable(tf.constant(0.5, shape=[100]))
    #h3 = tf.nn.relu(tf.matmul(flat, w3) + b3)
    #w4 = tf.Variable(tf.random_normal([100, 3]))
    #b4 = tf.Variable(tf.constant(0.5, shape=[3]))
    w1 = tf.Variable(tf.random_normal([256, 100]))
    b1 = tf.Variable(tf.constant(0.1, shape=[100]))
    h1 = tf.nn.relu(tf.matmul(lstm_output, w1) + b1)
    w2 = tf.Variable(tf.random_normal([100, 3]))
    b2 = tf.Variable(initial_value=[1., 0., 0.])
    return tf.nn.tanh(tf.matmul(h1, w2) + b2)

    #return tf.nn.tanh(tf.matmul(h3, w4) + b4)

def makeMask(object, height, width):
    flat_object = tf.reshape(object, [-1, height*width])
    w1 = tf.Variable(tf.random_normal([height * width, 100]))
    b1 = tf.Variable(tf.constant(0.1, shape=[100]))
    h1 = tf.nn.relu(tf.matmul(flat_object, w1) + b1)
    w2 = tf.Variable(tf.random_normal([100, 100]))
    # bias here is zero to encourage 50/50 split originally.
    b2 = tf.Variable(tf.constant(0.1, shape=[100]))
    return tf.reshape(tf.nn.sigmoid(tf.matmul(h1, w2) + b2), [-1, height, width])


background = np.median(getFrames(1000), axis=0)
plt.figure()
plt.imshow(background)
plt.savefig('background.png')
output_image = tf.zeros([batch_size, 84, 84])
lstm = tf.nn.rnn_cell.BasicLSTMCell(256)
state = tf.zeros(shape=[batch_size, lstm.state_size])
for i in range(3):
    with tf.variable_scope('lstm') as scope:
        if i > 0:
            scope.reuse_variables()
        flat_inp = tf.reshape(inp, [-1, height * width])
        output, state = lstm(flat_inp, state, scope=scope)
    where_flat = makeLocalizer(output)
    where = extract_where(where_flat)
    object = transformer(
                tf.reshape(inp, [-1, height, width, 1]),
                tf.reshape(where, [-1, 6]),
                [10, 10])
    obj_background = transformer(
        tf.tile(tf.reshape(background, [1, height, width, 1]), [batch_size, 1, 1, 1]),
        tf.reshape(where, [-1, 6]),
        [10, 10]
    )
    object = tf.reshape(object, [-1, 10, 10])
    obj_background = tf.reshape(obj_background, [-1, 10, 10])
    mask = makeMask(object, 10, 10)
    inv_where_flat = tf.concat(1, [1.0 / (tf.reshape(where_flat[:, 0], [-1, 1]) + 10**-5),
                                     -tf.reshape(where_flat[:, 1] * 1.0 / (where_flat[:, 0] + 10**-5), [-1,1]),
                                     -tf.reshape(where_flat[:, 2] * 1.0 / (where_flat[:, 0] + 10**-5), [-1,1])])
    inv_where = extract_where(inv_where_flat)
    restored_object = transformer(
        tf.reshape(object, [-1, 10, 10, 1]),
        tf.reshape(inv_where, [-1, 6]),
        [height, width]
    )
    restored_object = tf.reshape(restored_object, [-1, height, width])
    restored_obj_background = transformer(
        tf.reshape((1-mask)*obj_background, [-1, 10, 10, 1]),
        tf.reshape(inv_where, [-1, 6]),
        [height, width]
    )
    restored_obj_background = tf.reshape(restored_obj_background, [-1, height, width])
    output_image += restored_object
output_image += (1 - output_image) * background

loss = tf.reduce_mean(tf.pow(output_image - inp, 2))
train_batch = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
i = 0
while True:
    inp_ = getFrames(batch_size)
    feed_dict = {
        inp: inp_
    }
    [_, loss_, output_image_] = sess.run([train_batch, loss, output_image], feed_dict)
    print loss_
    print i
    if i % 10 == 0:
        f, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(output_image_[0, :, :], cmap='Greys_r')
        ax2.imshow(inp_[0], cmap='Greys_r')
        f.savefig('./res.png')
    i += 1