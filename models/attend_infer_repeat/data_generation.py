import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data')


def combineTwo(image1, image2):
    final_size = 40
    x1 = np.random.randint(0, 4)
    x2 = np.random.randint(14, 20)
    y1 = np.random.randint(0, 15)
    y2 = np.random.randint(0, 15)
    s1 = 0.3*np.random.rand() + 0.3
    s2 = 0.3*np.random.rand() + 0.3
    move = lambda x, y: np.float32(np.mat([[1.0, 0, x], [0, 1.0, y]]))
    img1 = cv2.resize(np.reshape(image1, (28,28)),(final_size, final_size), interpolation = cv2.INTER_CUBIC)
    img1 = cv2.warpAffine(img1, np.float32([[s1, 0, 0], [0, s1, 0]]), (final_size, final_size))
    img1 = cv2.warpAffine(img1, move(x1, y1), (final_size, final_size))
    img2 = cv2.resize(np.reshape(image2, (28,28)),(final_size, final_size), interpolation = cv2.INTER_CUBIC)
    img2 = cv2.warpAffine(img2, np.float32([[s2, 0, 0], [0, s2, 0]]), (final_size, final_size))
    img2 = cv2.warpAffine(img2, move(x2, y2), (final_size, final_size))
    return cv2.add(img1, img2)

def get_batch(batch_size):
    idx1 = np.random.choice(range(len(mnist.train.images)), batch_size)
    idx2 = np.random.choice(range(len(mnist.train.images)), batch_size)
    images1 = mnist.train.images[idx1, :]
    labels1 = mnist.train.labels[idx1]
    images2 = mnist.train.images[idx2, :]
    labels2 = mnist.train.labels[idx2]
    images = []
    labels = []
    zeros_image = np.zeros((28, 28), dtype=np.float32)
    for (image1, image2, label1, label2) in zip(images1, images2, labels1, labels2):
        if random.randint(0, 1) == 0:
            images.append(np.reshape(combineTwo(image1, image2), [1600]))
        else:
            images.append(np.reshape(combineTwo(image1, zeros_image), [1600]))
        label = np.zeros(10)
        label[label1] = 1
        label[label2] = 1
        labels.append(label)
        leak = 0
        images[-1][images[-1] >= 0.5] = 1 - leak
        images[-1][images[-1] < 0.5] = leak
    return images