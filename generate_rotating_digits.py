import numpy as np
import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def placeDigitInPosition(digit, image, x, y):
    start_x, end_x = x-5, x+5
    start_y, end_y = y-5, y+5
    image = image.copy()
    print start_x, end_x, start_y, end_y
    image[start_x:end_x, start_y:end_y] = digit
    return image


def walkingDigit(digit, step):
    canvas = np.zeros((40, 40))
    digit = cv2.resize(digit, (10, 10))
    px, py = 20, 20
    while True:
        px = np.clip(px + np.random.randint(-step, step+1), 5, 35)
        py = np.clip(py + np.random.randint(-step, step+1), 5, 35)
        yield placeDigitInPosition(digit, canvas, px, py)

mnist = input_data.read_data_sets('MNIST_data')
plt.ion()
digit = mnist.train.images[0, :].reshape((28, 28))
for img in walkingDigit(digit, 5):
    plt.figure()
    plt.imshow(img)
    plt.draw()
    time.sleep(0.1)
    plt.close()


