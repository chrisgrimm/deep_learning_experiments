import numpy as np
import random

class RollingBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.cursor = 0

    def add(self, element):
        if len(self.buffer) < self.capacity:
            self.buffer.append(element)
        else:
            self.buffer[self.cursor] = element
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, number):
        return random.sample(self.buffer, number)