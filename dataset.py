import math
import random
import torch

"""
1. Define the Dataset class
2. Define the ground truth function that needs to be learnt

NOTE:
The dataset arragement is like this:
train_set =>
[[x, y], [x, y], ...]
[[v], [v], ...]
"""


class Dataset:
    def __init__(self, count, half_width=1):
        self.train_set = [[], []]
        self.test_set = [[], []]
        for i in range(count):
            x = random.uniform(-half_width, half_width)
            y = random.uniform(-half_width, half_width)
            self.train_set[0].append([x, y])
            self.train_set[1].append([self.function(x, y)])
            x = random.uniform(-half_width, half_width)
            y = random.uniform(-half_width, half_width)
            self.test_set[0].append([x, y])
            self.test_set[1].append([self.function(x, y)])

    def get_train_set(self):
        return torch.tensor(self.train_set[0]), torch.tensor(self.train_set[1])

    def get_test_set(self):
        return torch.tensor(self.test_set[0]), torch.tensor(self.test_set[1])

    @staticmethod
    def function(x, y):
        # return x**2 + y**2
        return math.sin(10 * x) + math.cos(10 * y)
