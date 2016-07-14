import tensorflow as tf
import numpy as np
from objectDetector import AIR
from AIRAugmented import AIR
from data_generation import get_batch
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data")
import cv2
from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt
import sys, time, numpy as np
import random
import pickle
import os
import cv2

ale = ALEInterface()
ale.setInt('random_seed', 123)

def get_screen_array():
    w, h = ale.getScreenDims()
    data = np.zeros(w * h * 3, dtype=np.uint8)
    ale.getScreen(data)
    image = data.reshape((h, w, 3))
    #image = cv2.resize(image, (84, 110))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    gray = gray[10:67, :55]
    return gray

def choose_action():
    return random.choice(ale.getMinimalActionSet())


def take_sequence(rom_path, step, n):
    ale.loadROM(rom_path)
    w, h = ale.getScreenDims()
    screenshots = np.zeros((n,57*55))
    prev_shot = np.zeros((57, 55))
    j = 0
    for i in range(step*n):
        if ale.game_over():
            ale.loadROM(rom_path)
        ale.act(choose_action())
        if i % step == 0:
            screen_shot = get_screen_array()
            temp = 1 * (np.abs(screen_shot - prev_shot) > 0)
            prev_shot = screen_shot
            cv2.imshow("SDfs", temp)
            cv2.waitKey(50)
            print np.median(temp)
            screenshots[j, :] = temp.flatten()
            j += 1
    return screenshots

rom_path = './roms/pong.bin'

sess = tf.Session()

batch_size = 500


air = AIR(sess, 57, 55, 30, 3, batch_size)
sess.run(tf.initialize_all_variables())
#print "RESTORING!!!"
#air.restore()
#print "RESTORed!!!"

num_batches = 10000000

for i in range(1, num_batches):
    print i
    batch = take_sequence(rom_path, 1, batch_size)
    np.random.shuffle(batch)

air = AIR(sess, 40, 40, 30, 2, batch_size)
sess.run(tf.initialize_all_variables())
#air.restore()

num_batches = 10000000


for i in range(1, num_batches):
    print i
    batch = sample_batch(generator)
    loss = air.train_batch(batch, i)
    print loss
    if i % 1000 == 0 and i != 0 :
        air.save()
    if i % 100 == 0:
        air.visualize_result(batch, str(i))
