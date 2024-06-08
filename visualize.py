import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from dataset import *

"""
1. Visualize the implicit learnt mapping of the network
2. Compare it against the ground truth mapping
"""

net = torch.load("siren_latest.ckpt")  # load the latest model

# prepare the input based on [-1, 1] x [-1, 1]
input = []
half_width, step = 1, 0.05
gt = []
grid_x = int(2 * half_width / step)
for i in np.arange(-half_width, half_width, step):
    for j in np.arange(-half_width, half_width, step):
        input.append([float(i), float(j)])
        gt.append(Dataset.function(i, j))

# evaluate the model
output = net(torch.tensor(input)).detach().numpy()

# draw the evaluated 2D function
fig = plt.figure()
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection="3d")

# create mesh grid from scratch
X = np.array([i[0] for i in input]).reshape((grid_x, grid_x))
Y = np.array([i[1] for i in input]).reshape((grid_x, grid_x))
Z = np.array([i[0] for i in output]).reshape((grid_x, grid_x))
gt = np.array(gt).reshape((grid_x, grid_x))

ax0.set_xlabel("X")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")
ax0.set_title("INR")

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("GT")

surf0 = ax0.plot_surface(X, Y, Z, cmap="viridis")
surf1 = ax1.plot_surface(X, Y, gt, cmap="viridis")
plt.show()
