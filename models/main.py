import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from SpatialTransformerNetwork import SpatialTransformerNetwork
import sys

if len(sys.argv) < 2:
    raise Exception('Command must be specified as \'show_projection\' or \'train\'.')

mnist_cluttered = np.load('./mnist_sequence1_sample_5distortions5x5.npz')

# convert labels to one-hot
def to_onehot(length, i):
    zeros = np.zeros(length)
    zeros[i] = 1
    return zeros

X_train = np.reshape(mnist_cluttered['X_train'], (10000, 40, 40))
y_train = mnist_cluttered['y_train']
indices = range(len(X_train))

def get_batch(size):
    idx = np.random.choice(indices, size)
    return (X_train[idx, :, :], [to_onehot(10, label) for label in y_train[idx, :]])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

batch_size = 50 if sys.argv[1] == 'train' else 1
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[batch_size, 40, 40])
y_ = tf.placeholder(tf.float32, shape=[batch_size, 10])


# spatial layer
spatial_layer_node = SpatialTransformerNetwork(x, (40, 40), (40, 40))
spatial_layer = tf.reshape(spatial_layer_node.output, [-1, 40, 40, 1])
# first conv layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(spatial_layer, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# second conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# fully connected layer
W_fc1 = weight_variable([10*10*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropping
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# learning
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver({'loc_w1': spatial_layer_node.loc_w1,
                        'loc_b1': spatial_layer_node.loc_b1,
                        'loc_w2': spatial_layer_node.loc_w2,
                        'loc_b2': spatial_layer_node.loc_b2,
                        'W_conv1': W_conv1,
                        'b_conv1': b_conv1,
                        'W_conv2': W_conv2,
                        'b_conv2': b_conv2,
                        'W_fc1': W_fc1,
                        'b_fc1': b_fc1,
                        'W_fc2': W_fc2,
                        'b_fc2': b_fc2})


def train():
    best_accuracy = 0.0
    for i in range(100000000):
        batch = get_batch(50)
        if i%100 == 0:
            localizers = sess.run(spatial_layer_node.localizers, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            if train_accuracy > best_accuracy or train_accuracy == 1.0:
                print "Storing new best accuracy..."
                saver.save(sess, './spatial_transformer_weights')
                best_accuracy = train_accuracy
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

def show_projection():
    saver.restore(sess, './spatial_transformer_weights')
    images, labels = get_batch(1000)
    for image in images:
        f, [ax1, ax2] = plt.subplots(1, 2)
        feed_dict = {x: np.reshape(image, (1, 40, 40))}
        [localizers, reconstructed] = sess.run([spatial_layer_node.localizers, spatial_layer], feed_dict)
        print localizers[0, :, :]
        ax1.imshow(np.reshape(image, (40, 40)))
        ax2.imshow(np.reshape(reconstructed, (40, 40)))
        f.show()
        raw_input()
        f.clf()


if len(sys.argv) < 2:
    raise Exception('Command must be specified as \'show_projection\' or \'train\'.')
if sys.argv[1] == 'show_projection':
    show_projection()
elif sys.argv[1] == 'train':
    train()
else:
    raise Exception('Command must be specified as \'show_projection\' or \'train\'.')