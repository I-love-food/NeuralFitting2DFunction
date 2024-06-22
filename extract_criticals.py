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
    train_set = Dataset.load_train_set()
else:
    gen = poisson_disk(r=0.05, k=100, span=[[-1, 1], [-1, 1]])
    train_set = Dataset(gen).get_train_set()

net = torch.load(f"models\{function_name}-latest.ckpt")  # load the latest model

input = train_set[0]
values = net(torch.tensor(train_set[0])).detach().numpy().flatten()

mesh = mtri.Triangulation(input[:, 0], input[:, 1])
vnum = len(mesh.x)

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
        graph[va] = {}
    graph[va][vb] = True


for t in mesh.triangles:
    connect(t[0], t[1])
    connect(t[1], t[0])
    connect(t[0], t[2])
    connect(t[2], t[0])
    connect(t[1], t[2])
    connect(t[2], t[1])

# traverse each vertex in the graph, then determine which category it falls into
def find(x, P):
    if P[x] == x:
        return x
    else:
        P[x] = find(P[x], P)
        return P[x]


def merge(x, y, P):
    a = find(x, P)
    b = find(y, P)
    if not a == b:
        P[b] = a


def get_lower_link_cpn(u):
    nu = list(graph[u].keys())
    val_u = values[u]
    lower_link = []
    for i in range(len(nu)):
        v = nu[i]
        val_v = values[v]
        if val_v < val_u:
            lower_link.append(i)

    count = len(lower_link)

    P = []
    for i in range(count):
        P.append(i)

    for i in range(count):
        for j in range(i + 1, count):
            va, vb = nu[lower_link[i]], nu[lower_link[j]]
            if graph[va].get(vb) != None:
                merge(i, j, P)

    comp_id = []
    for i in range(count):
        comp_id.append(find(i, P))
    comp_id.sort()
    comp_id = set(comp_id)
    return len(comp_id)


def get_upper_link_cpn(u):
    nu = list(graph[u].keys())
    val_u = values[u]
    upper_link = []
    for i in range(len(nu)):
        v = nu[i]
        val_v = values[v]
        if val_v > val_u:
            upper_link.append(i)

    count = len(upper_link)

    P = []
    for i in range(count):
        P.append(i)

    for i in range(count):
        for j in range(i + 1, count):
            va, vb = nu[upper_link[i]], nu[upper_link[j]]
            if graph[va].get(vb) != None:
                merge(i, j, P)

    comp_id = []
    for i in range(count):
        comp_id.append(find(i, P))
    comp_id.sort()
    comp_id = set(comp_id)
    return len(comp_id)


cpns = {}
for u in graph.keys():
    llk, ulk = get_lower_link_cpn(u), get_upper_link_cpn(u)
    cpns[u] = (llk, ulk)
    print(f"(Lk-, Lk+) of vertex {u} connected component = {llk, ulk}")


def fetch_criticals(cpns):
    min_points, min_value = [], []
    max_points, max_value = [], []
    saddle_points, saddle_value = [], []

    for i in range(vnum):
        llk_cpn = cpns[i][0]
        ulk_cpn = cpns[i][1]
        if llk_cpn == 0 and ulk_cpn == 1:
            # min
            min_points.append(input[i])
            min_value.append(values[i])
        elif ulk_cpn == 0 and llk_cpn == 1:
            # max
            max_points.append(input[i])
            max_value.append(values[i])
        elif ulk_cpn == 2 and llk_cpn == 2:
            # saddle
            saddle_points.append(input[i])
            saddle_value.append(values[i])

    min_info = np.array(min_points), np.array(min_value)
    max_info = np.array(max_points), np.array(max_value)
    saddle_info = np.array(saddle_points), np.array(saddle_value)
    return min_info, max_info, saddle_info


critical_info = fetch_criticals(cpns)

fig = plt.figure()
ax0 = fig.add_subplot(111, projection="3d")
surf0 = ax0.plot_trisurf(
    mesh.x,
    mesh.y,
    values,
    cmap="viridis",
    alpha=0.1,
    edgecolors="black",
    linewidths=0.1,
)


ax0.scatter(
    critical_info[0][0][:, 0],
    critical_info[0][0][:, 1],
    critical_info[0][1],
    color="blue",
)
ax0.scatter(
    critical_info[1][0][:, 0],
    critical_info[1][0][:, 1],
    critical_info[1][1],
    color="red",
)
ax0.scatter(
    critical_info[2][0][:, 0],
    critical_info[2][0][:, 1],
    critical_info[2][1],
    color="green",
)

plt.show()
