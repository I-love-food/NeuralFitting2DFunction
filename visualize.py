import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from dataset import *
from sampler import poisson_disk

"""
1. Visualize the implicit learnt mapping of the network
2. Compare it against the ground truth mapping
"""
net = torch.load(
    "models\[Ackley]siren_latest-2024-06-16 19-42-44-100.ckpt"
)  # load the latest model
if fix_dataset:
    train_set = Dataset.load_train_set("datasets/" + function_name)
else:
    gen = poisson_disk(r=0.05, k=100, span=[[-1, 1], [-1, 1]])
    train_set = Dataset(gen).get_train_set()

input = torch.tensor(train_set[0])
count = len(input)
gt = train_set[1].reshape(count)
# evaluate the model
output = net(input).detach().numpy()
# draw the evaluated 2D function
fig = plt.figure()
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection="3d")

Z = output.reshape(count)
print(Z)

ax0.set_xlabel("X")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")
ax0.set_title("INR")

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("GT")

ax0.plot_trisurf(input[:, 0], input[:, 1], Z, cmap="viridis")
ax1.plot_trisurf(input[:, 0], input[:, 1], gt, cmap="viridis")
plt.tight_layout()
plt.show()
