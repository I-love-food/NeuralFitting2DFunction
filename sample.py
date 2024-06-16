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
from sampler import poisson_disk

"""
# show the mesh
plt.figure()
plt.triplot(mesh, "go-")
plt.title("Delaunay Triangulation")
plt.show()
# how to convert torch.tensor to numpy
xxx.detach().numpy()
"""
# freeze the dataset to make debug easier
if fix_dataset:
    train_set = Dataset.load_train_set("datasets/" + function_name)
else:
    gen = poisson_disk(r=0.05, k=100, span=[[-1, 1], [-1, 1]])
    train_set = Dataset(gen).get_train_set()

input = train_set[0]

mesh = mtri.Triangulation(input[:, 0], input[:, 1])
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
net = torch.load(
    "models\[Ackley]siren_latest-2024-06-16 19-42-44-100.ckpt"
)  # load the latest model

# I will not use gradient in this function,
# just compare the function value (forward the net) to determine the next step
def get_val(v):
    coord = np.array(input[v])
    eval = net(torch.tensor(coord)).detach().numpy()
    return eval[0]


def trace_on_mesh_recursive(u, path):
    cur_val = get_val(u)
    path.append([u, cur_val])
    next_node = -1
    minv = sys.float_info.max
    for i, next_v in enumerate(graph[u]):
        val = get_val(next_v)
        if val < minv:
            minv = val
            next_node = i
    if minv < cur_val:
        trace_on_mesh_recursive(graph[u][next_node], path)
    else:
        # u is a min point
        return


def trace_on_mesh_iterative(u, path):
    while True:
        cur_val = get_val(u)
        path.append([u, cur_val])
        minv = sys.float_info.max
        next_node = -1
        for i, next_v in enumerate(graph[u]):
            val = get_val(next_v)
            if val < minv:
                minv = val
                next_node = i
        if minv < cur_val:
            u = graph[u][next_node]
        else:
            break


start = list(graph.keys())[223]  # choose center point
path = []

trace_on_mesh_iterative(start, path)

# plot function f(x, y) represented by the net based on the mesh
input = []
for i in range(len(graph)):
    input.append([mesh.x[i], mesh.y[i]])
input = np.array(input, dtype=np.float32)
output = net(torch.tensor(input)).detach().numpy()
X = copy.deepcopy(mesh.x)
Y = copy.deepcopy(mesh.y)

# draw the inplicit surface based on the sampled points
fig = plt.figure()
ax0 = fig.add_subplot(111, projection="3d")
surf0 = ax0.plot_trisurf(
    X.flatten(), Y.flatten(), output.flatten(), cmap="viridis", alpha=0.3
)

# draw the trace line--"integral line"
XL = []
YL = []
ZL = []
for point in path:
    coord = input[point[0]]
    XL.append(coord[0])
    YL.append(coord[1])
    ZL.append(point[1])

ax0.plot(XL, YL, ZL, color="blue", marker="*", markersize=8)

plt.tight_layout()
plt.show()
