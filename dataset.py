import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sampler import poisson_disk


"""
1. Define the Dataset class
2. Define the ground truth function that needs to be learnt

NOTE:
The dataset arragement is like this:
train_set =>
[[x, y], [x, y], ...]
[[v], [v], ...]

If you want to generate fixed dataset:
    gen = poisson_disk(r=0.05, k=100, span=[[-1, 1], [-1, 1]])
    train_set = Dataset(gen).save_train_set("datasets/[X2Y2]")
"""


def gaussian_function(points, mu1=0, mu2=0, sigma=1):
    x = points[:, 0]
    y = points[:, 1]
    two_sigma2 = 2 * sigma * sigma
    return (
        1
        / (np.pi * two_sigma2)
        * np.exp(-((x - mu1) ** 2 + (y - mu2) ** 2) / two_sigma2)
    )


def ackley_function(points, a, b, c, d):
    x = points[:, 0]
    y = points[:, 1]
    exp1 = np.exp(-b * np.sqrt(1 / d * (x**2 + y**2)))
    exp2 = 1 / d * (np.cos(c * x) + np.cos(c * y))
    return -a * exp1 - exp2 + a + np.exp(1)


def x2y2_function(points):
    x = points[:, 0]
    y = points[:, 1]
    return (x) ** 2 + (y) ** 2


class Dataset:
    # gen is a sampler instance
    def __init__(self, gen, fun):
        samples = gen.sample()
        self.train_set = [
            samples,
            self.function(samples[:, 0], samples[:, 1])
            .reshape(-1, 1)
            .astype(np.float32),
        ]
        self.fun = fun

    def get_train_set(self):
        return self.train_set

    def save_train_set(self):
        np.save("datasets/" + self.fun + "-P.npy", self.train_set[0])
        np.save("datasets/" + self.fun + "-V.npy", self.train_set[1])

    @staticmethod
    def load_train_set(fun):
        train_set = [
            np.load("datasets/" + fun + "-P.npy"),
            np.load("datasets/" + fun + "-V.npy"),
        ]
        return train_set

    def function(self, x, y):
        if self.fun == "Ackley":
            return ackley_function(x, y, 20, 0.2, 2 * np.pi, 2)
        elif self.fun == "Gaussian":
            return gaussian_function(x, y, 0, 0, 0.5)
        elif self.fun == "X2Y2":
            return x2y2_function(x, y)
        else:
            raise ValueError(f"Function {self.fun} is not supported.")


# Generate fixed dataset
# Dataset(
#     poisson_disk(0.01, k=100, span=[[-1, 1], [-1, 1]], bounded=True)
# ).save_train_set()
