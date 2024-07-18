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

samples = np.load("samples/6041-[[-5, 5], [-5, 5]].npy").astype(
    np.float32
)  # n x 2 list
net = torch.load("models/my_model_volcano")
inputs = torch.tensor(samples, requires_grad=True)
outputs = net(inputs)
values = outputs.detach().numpy().flatten()
mesh = mtri.Triangulation(samples[:, 0], samples[:, 1])
# plt.triplot(mesh, "bo-", alpha=0.5)  # 'bo-' 表示蓝色圆点和线段
# plt.show()
vnum = len(samples)


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


def get_lower_link(u):  # return the lower link for this vertex: [[], [], ...]
    nu = list(graph[u].keys())
    val_u = values[u]
    lower_link = []  # this is the vertex on the lower link for vertex u
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

    LK = {}

    for i in range(count):
        set_id = find(i, P)
        if LK.get(set_id) == None:
            LK[set_id] = []
        LK[set_id].append(nu[lower_link[i]])

    return LK


def get_upper_link(u):
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

    UK = {}

    for i in range(count):
        set_id = find(i, P)
        if UK.get(set_id) == None:
            UK[set_id] = []
        UK[set_id].append(nu[upper_link[i]])

    return UK


links = {}
for u in graph.keys():
    llk, ulk = get_lower_link(u), get_upper_link(u)
    links[u] = (llk, ulk)


def fetch_criticals(links):
    min_points = []
    max_points = []
    saddle_points = []
    monkey_saddle_points = []

    for i in range(vnum):
        llk_ccpn = len(links[i][0].keys())
        ulk_ccpn = len(links[i][1].keys())
        if llk_ccpn == 0 and ulk_ccpn == 1:
            # min
            min_points.append(i)
        elif ulk_ccpn == 0 and llk_ccpn == 1:
            # max
            max_points.append(i)
        elif ulk_ccpn == 2 and llk_ccpn == 2:
            # saddle
            saddle_points.append(i)
        elif ulk_ccpn == 3 and llk_ccpn == 3:
            monkey_saddle_points.append(i)
            print("find a monkey saddle")

    return min_points, max_points, saddle_points, monkey_saddle_points


min_points, max_points, saddle_points, monkey_saddle_points = fetch_criticals(links)


def trace_min(u, path):
    while True:
        path.append(u)
        minv = sys.float_info.max
        next_node = -1
        nu = list(graph[u].keys())
        for v in nu:
            if values[v] < minv:
                minv = values[v]
                next_node = v
        if minv < values[u]:
            u = next_node
        else:
            break


def trace_max(u, path):
    while True:
        path.append(u)
        maxv = -sys.float_info.max
        next_node = -1
        nu = list(graph[u].keys())
        for v in nu:
            if values[v] > maxv:
                maxv = values[v]
                next_node = v
        if maxv > values[u]:
            u = next_node
        else:
            break


# I think, in the future, maybe critical points needs to be refined,
# the correct way to trace via gradient is from the correct critical point
def eval_model(vertex):
    inputs = torch.tensor(vertex, requires_grad=True)
    out = net(inputs)
    out.backward()
    return inputs.grad.numpy(), out.detach().numpy()


# Important: u is the actual 2D position in space
def trace_min_numerical(u, lr=0.01, max_iter=50):
    path, vals = [], []
    for _ in range(max_iter):
        grad, val = eval_model(u)
        vals.append(val)
        u_next = u - lr * grad
        u = u_next
        path.append(u)
    return np.array(path), np.array(vals).flatten()


def trace_max_numerical(u, lr=0.01, max_iter=50):
    u = np.array(u)
    path, vals = [], []
    for _ in range(max_iter):
        grad, val = eval_model(u)
        vals.append(val)
        u_next = u + lr * grad
        u = u_next
        path.append(u)
    return np.array(path), np.array(vals).flatten()


# Trace integral lines from here (starting from saddles)
intline_D = {}
for u in saddle_points:
    LK, UK = links[u]
    intline_D[u] = [[], []]  # min path x 2, max path x 2
    # enumerate lower links connected components/wedges
    for _, cp in LK.items():
        idx = min(cp, key=lambda x: values[x])
        path = []
        trace_min(idx, path)
        intline_D[u][0].append(path)

    # enumerate upper links connected components/wedges
    for _, cp in UK.items():
        idx = max(cp, key=lambda x: values[x])
        path = []
        trace_max(idx, path)
        intline_D[u][1].append(path)

for u in monkey_saddle_points:
    LK, UK = links[u]
    intline_D[u] = [[], []]  # min path x 3, max path x 3
    # enumerate lower links connected components/wedges
    for _, cp in LK.items():
        idx = min(cp, key=lambda x: values[x])
        path = []
        trace_min(idx, path)
        intline_D[u][0].append(path)

    # enumerate upper links connected components/wedges
    for _, cp in UK.items():
        idx = max(cp, key=lambda x: values[x])
        path = []
        trace_max(idx, path)
        intline_D[u][1].append(path)

# visualization
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

saddle_vertices = np.array(list(map(lambda x: samples[x], saddle_points)))
saddle_values = np.array(list(map(lambda x: values[x], saddle_points)))
monkey_saddle_vertices = np.array(list(map(lambda x: samples[x], monkey_saddle_points)))
monkey_saddle_values = np.array(list(map(lambda x: values[x], monkey_saddle_points)))

ax0.scatter(
    saddle_vertices[:, 0],
    saddle_vertices[:, 1],
    saddle_values,
    color="orange",
    s=50,
    marker="P",
    label="saddle",
)

ax0.scatter(
    monkey_saddle_vertices[:, 0],
    monkey_saddle_vertices[:, 1],
    monkey_saddle_values,
    color="aqua",
    s=100,
    marker="P",
    label="monkey_saddle",
)


max_vertices = np.array(list(map(lambda x: samples[x], max_points)))
max_values = np.array(list(map(lambda x: values[x], max_points)))

ax0.scatter(
    max_vertices[:, 0],
    max_vertices[:, 1],
    max_values,
    color="red",
    s=50,
    marker="^",
    label="max",
)

min_vertices = np.array(list(map(lambda x: samples[x], min_points)))
min_values = np.array(list(map(lambda x: values[x], min_points)))

ax0.scatter(
    min_vertices[:, 0],
    min_vertices[:, 1],
    min_values,
    color="blue",
    s=50,
    marker="v",
    label="min",
)

fig = plt.figure()
ax1 = fig.add_subplot(111)  # 2D version critical points
ax1.scatter(
    saddle_vertices[:, 0],
    saddle_vertices[:, 1],
    color="orange",
    marker="P",
    s=50,
    label="saddle",
)
ax1.scatter(
    max_vertices[:, 0], max_vertices[:, 1], color="red", marker="^", s=50, label="max"
)
ax1.scatter(
    min_vertices[:, 0], min_vertices[:, 1], color="blue", marker="v", s=50, label="min"
)

ax1.scatter(
    monkey_saddle_vertices[:, 0],
    monkey_saddle_vertices[:, 1],
    color="aqua",
    s=100,
    marker="P",
    label="monkey_saddle",
    alpha=1.0,
)

# using mesh grid to show the underlying value field
ct = ax1.tricontourf(
    mesh, values, levels=50, cmap="viridis", antialiased=True, zorder=-1
)
plt.colorbar(ct)


# 画一个梯度场
grads = []
for i in samples:
    grad, _ = eval_model(i)
    # grad = grad / np.linalg.norm(grad)
    grads.append(grad)

grads = np.array(grads)
ax1.quiver(samples[:, 0], samples[:, 1], grads[:, 0], grads[:, 1], color="black")

for saddle, paths in intline_D.items():
    min_path = paths[0]  # min path
    max_path = paths[1]  # max path
    for path in min_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax0.plot(vs[:, 0], vs[:, 1], vals, color="blue", linestyle="--")
        ax1.plot(vs[:, 0], vs[:, 1], color="blue", linestyle="--")
    for path in max_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax0.plot(vs[:, 0], vs[:, 1], vals, color="red")
        ax1.plot(vs[:, 0], vs[:, 1], color="red")
ax0.legend()
ax1.legend(loc="lower right")
# get the gradient of all of the critical points
# use a 3d bar plot to show the gradient of each kinds of critical points
# min points
grad_len = {"min": [], "max": [], "saddle": []}
for p in min_points:
    vertex = np.array(samples[p])
    grad, _ = eval_model(vertex)
    grad_length = np.linalg.norm(grad)
    grad_len["min"].append(grad_length)
    # print(grad_length)

for p in max_points:
    vertex = np.array(samples[p])
    grad, _ = eval_model(vertex)
    grad_length = np.linalg.norm(grad)
    grad_len["max"].append(grad_length)

for p in saddle_points:
    vertex = np.array(samples[p])
    grad, _ = eval_model(vertex)
    grad_length = np.linalg.norm(grad)
    grad_len["saddle"].append(grad_length)

fig = plt.figure()
ax3 = fig.add_subplot(111, projection="3d")


ax3.bar3d(
    max_vertices[:, 0],
    max_vertices[:, 1],
    0,
    0.02,
    0.02,
    grad_len["max"],
    color="red",
    label="max",
)

ax3.bar3d(
    min_vertices[:, 0],
    min_vertices[:, 1],
    0,
    0.02,
    0.02,
    grad_len["min"],
    color="blue",
    label="min",
)

ax3.bar3d(
    saddle_vertices[:, 0],
    saddle_vertices[:, 1],
    0,
    0.02,
    0.02,
    grad_len["saddle"],
    color="black",
    label="saddle",
)
plt.legend()
plt.show()


# ！！！！！2D 情况下 field value colormap
# 目前最重要的：优化调整critical points
# monkey saddle: locally 增加一些sample，目前尝试全部重新triangulation / PL2 paper上的处理方法
# junction point: 在连续情况下 junction point只会出现在critical points, 希望junction point尽可能接近critical
# point / 用triangular surface paper的方法
# 模型训练没训练好 (有可能) 导致numerical line trace 出现问题
# 其他 criticals, 也可以尝试用one-neighbour来细化 --> 从而实现对critical的优化
# 比如maximun，他有一个one-star, 然后找one-star里面每一个三角形的质心，
# 看maximum往哪边移动，在那个新到达的star中接着做refine
# 增大一下训练范围， 其实就在原本边界上采样就行了
