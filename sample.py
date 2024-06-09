import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from dataset import *
import networkx as nx
import sys
import copy

"""
# show the mesh
plt.figure()
plt.triplot(mesh, "go-")
plt.title("Delaunay Triangulation")
plt.show()
# how to convert torch.tensor to numpy
xxx.detach().numpy()
"""

# sample range
lower_left = [-1, -1]
upper_right = [1, 1]

# sample resolution
step = 0.1
grid_x = (upper_right[0] - lower_left[0]) / step
grid_y = (upper_right[1] - lower_left[1]) / step

samples = []
for i in np.arange(lower_left[0], upper_right[0], step):
    for j in np.arange(lower_left[1], upper_right[1], step):
        samples.append([i, j])
samples = np.array(samples, dtype=np.float32)
mesh = mtri.Triangulation(samples[:, 0], samples[:, 1])
# convert the triangle mesh to graph
# Direct Graph? Undirected Graph?
# How to save the Graph? Matrix? Link-list?
# Undirected Graph & Link-list
"""
The Graph:
v0 -> va, vb, ...
v1 -> vc, vd, ...
...
I can use networkx to refactor this part in the future.
"""
graph = {}


def connect(va, vb):
    if graph.get(va) == None:
        graph[va] = []
    graph[va].append(vb)


for t in mesh.triangles:
    connect(t[0], t[1])
    connect(t[1], t[0])
    connect(t[0], t[2])
    connect(t[2], t[0])
    connect(t[1], t[2])
    connect(t[2], t[1])


# Do the test from one point
# In this case, I will
# 1. load the model which has been fitted to some f(x, y)
# 2. choose graph.key[0] as the starting point
# 3. trace a "streamline" from starting point
net = torch.load("siren_latest-100.ckpt")  # load the latest model

# I will not use gradient in this function,
# just compare the function value (forward the net) to determine the next step
def get_val(v):
    coord = np.array(samples[v])
    eval = net(torch.tensor(coord)).detach().numpy()
    return eval[0]


vis = np.zeros(len(graph))


def trace_on_mesh(u, cur_val, path):
    if vis[u] == 1:
        return
    vis[u] = 1
    path.append([u, cur_val])
    next_vals = []
    for next_v in graph[u]:
        next_vals.append(get_val(next_v))
    next_node = -1
    minv = sys.float_info.max
    for i, next_val in enumerate(next_vals):
        if next_val < minv:
            minv = next_val
            next_node = i
    if minv < cur_val:
        trace_on_mesh(next_node, next_vals[next_node], path)
    else:
        # u is a min point
        return


start = list(graph.keys())[0]
path = []

trace_on_mesh(start, get_val(start), path)

# plot function f(x, y) represented by the net based on the mesh
input = []
for i in range(len(graph)):
    input.append([mesh.x[i], mesh.y[i]])
input = np.array(input, dtype=np.float32)
output = net(torch.tensor(input)).detach().numpy().reshape(grid_x, grid_y)
X = copy.deepcopy(mesh.x).reshape(grid_x, grid_y)
Y = copy.deepcopy(mesh.y).reshape(grid_x, grid_y)
# fig = plt.figure()
# ax0 = fig.add_subplot(121, projection="3d")
# surf0 = ax0.plot_surface(mesh.x, mesh.y, , cmap="viridis")
# surf1 = ax1.plot_surface(X, Y, gt, cmap="viridis")
# plt.tight_layout()
# plt.show()
