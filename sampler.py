import random
import numpy as np
import matplotlib.pyplot as plt

# from scipy.stats.qmc import PoissonDisk

"""
The axis model used:
 y
 ^
 |    .(1, 1)
 |
 o ------> x

...
row=1, col=0	row=1, col=1	row=1, col=2	...
row=0, col=0	row=0, col=1	row=0, col=2	...

Usage:
    a = uniform().sample()
    print(a)
    a = poisson_disk().sample()
    plt.scatter(a[:, 0], a[:, 1])
    plt.show()
"""


class uniform:
    def __init__(self, count=1000, span=[[0, 1], [0, 1]]):
        self.count = count
        self.span = span

    def sample(self):
        x = np.random.uniform(self.span[0][0], self.span[0][1], self.count)
        y = np.random.uniform(self.span[1][0], self.span[1][1], self.count)
        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)
        return np.concatenate((x, y), axis=1)


class grid:
    def __init__(self, res=[5, 5], span=[[0, 1], [0, 1]]):
        self.res = res
        self.span = span

    def sample(self):
        x = np.linspace(self.span[0][0], self.span[0][1], self.res[0])
        y = np.linspace(self.span[1][0], self.span[1][1], self.res[1])
        X, Y = np.meshgrid(x, y)
        X = np.expand_dims(X, 2)
        Y = np.expand_dims(Y, 2)
        return np.concatenate((X, Y), axis=2).reshape(-1, 2)


class poisson_disk:
    def __init__(self, r=0.01, k=100, span=[[0, 1], [0, 1]]):
        self.r = r
        self.k = k
        self.two_pi = 2 * np.pi
        self.box_side = r / np.sqrt(2)
        self.activate = []
        self.samples = []
        self.index = {}

        self.width = span[0][1] - span[0][0]
        self.height = span[1][1] - span[1][0]
        self.offsets = np.array([span[0][0], span[1][0]])
        self.cols = np.ceil(self.width / self.box_side).astype(np.int32)
        self.rows = np.ceil(self.height / self.box_side).astype(np.int32)

        for i in range(self.rows):
            for j in range(self.cols):
                self.index[i * self.cols + j] = -1

    # which box (x, y) locates in
    def get_box_id(self, x, y):
        i = np.ceil(y / self.box_side).astype(np.int32) - 1
        j = np.ceil(x / self.box_side).astype(np.int32) - 1
        return i, j, i * self.cols + j

    # sample points in an annulus whose center is (x, y) with (inner, outer) = (r, 2r)
    # a.k.a draw candidate samples around (x, y)
    def sample_annulus(self, x, y):
        rho = np.random.uniform(self.r, 2 * self.r, self.k)
        theta = np.random.uniform(0, self.two_pi, self.k)
        xx = rho * np.cos(theta) + x
        yy = rho * np.sin(theta) + y
        condition = (xx >= 0) & (xx <= self.width) & (yy >= 0) & (yy <= self.height)
        xx = xx[condition]
        yy = yy[condition]
        return xx, yy

    # calculate distance between 2 points
    def calc_dist(self, x1, y1, x2, y2):
        diffx = x1 - x2
        diffy = y1 - y2
        return np.sqrt(diffx**2 + diffy**2)

    # check if (x, y) could be a valid sample
    # valid sample: far away from all other drawn samples with distance >= r
    def check(self, x, y):
        i, j, _ = self.get_box_id(x, y)
        for u in range(-2, 2):
            for v in range(-2, 2):
                ii = i + u
                jj = j + v
                if ii >= 0 and ii < self.rows and jj >= 0 and jj < self.cols:
                    idx = ii * self.cols + jj
                    if not self.index[idx] == -1:
                        point = self.samples[self.index[idx]]
                        dist = self.calc_dist(point[0], point[1], x, y)
                        if dist < self.r:
                            return False
        return True

    # launch poisson disk sampling
    # idea: we have some samples already drawn on a plane, we want to draw another sample
    # and we want this sample to far away from other samples with distance larger than r
    def sample(self):
        start_x = np.random.uniform(0, self.width)
        start_y = np.random.uniform(0, self.height)
        self.samples.append([start_x, start_y])
        _, _, box_id = self.get_box_id(start_x, start_y)
        self.index[box_id] = 0
        self.activate.append(0)
        while True:
            idx = random.randint(0, len(self.activate) - 1)
            activate_id = self.activate[idx]
            point = self.samples[activate_id]
            k_samples = self.sample_annulus(point[0], point[1])
            Flag = False
            for cnt, sample_x in enumerate(k_samples[0]):
                sample_y = k_samples[1][cnt]
                flag = self.check(sample_x, sample_y)
                if flag:
                    _, _, box_id = self.get_box_id(sample_x, sample_y)
                    self.index[box_id] = len(self.samples)
                    self.activate.append(len(self.samples))
                    self.samples.append([sample_x, sample_y])
                    Flag = True

            # we did not get new samples from current sample, so we remove it
            if not Flag:
                self.activate.pop(idx)

            # when there is no activate samples (the samples that can be potentially used to get new samples)
            if len(self.activate) == 0:
                break
        result = np.array(self.samples) + self.offsets
        return result.astype(np.float32)
