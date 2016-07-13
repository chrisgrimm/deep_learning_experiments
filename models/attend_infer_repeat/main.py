import tensorflow as tf
import numpy as np
from AIRAugmented import AIR
from data_generation import get_batch
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data")
import cv2

sess = tf.Session()

batch_size = 100


air = AIR(sess, 40, 40, 30, 2, batch_size)
sess.run(tf.initialize_all_variables())
#air.restore()

num_batches = 10000000

def placeDigitInPosition(digit, image, x, y):
    start_x, end_x = x-5, x+5
    start_y, end_y = y-5, y+5
    image = image.copy()
    image[start_x:end_x, start_y:end_y] = digit
    return image


def walkingDigit(digit, step, num_digits):
    canvas = np.zeros((40, 40))
    digit = cv2.resize(digit, (10, 10))
    px = [20 for x in range(num_digits)]
    py = [20 for x in range(num_digits)]
    while True:
        res = canvas
        for i in range(num_digits):
            px[i] = np.clip(px[i] + np.random.randint(-step, step+1), 5, 35)
            py[i] = np.clip(py[i] + np.random.randint(-step, step+1), 5, 35)
            res = placeDigitInPosition(digit, res, px[i], py[i])
        yield res


def create_batch_generator(batch_size, num_digits):
    return [walkingDigit(mnist.train.images[i,:].reshape((28, 28)), 5, 2) for i in range(batch_size)]

def sample_batch(generator):
    return [next(gener).flatten() for gener in generator]

generator = create_batch_generator(100)

for i in range(1, num_batches):
    print i
    batch = sample_batch(generator)
    loss = air.train_batch(batch, i)
    print loss
    if i % 1000 == 0 and i != 0 :
        air.save()
    if i % 100 == 0:
        air.visualize_result(batch, str(i))
