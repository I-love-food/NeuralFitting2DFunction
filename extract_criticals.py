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
samples = np.load("samples/6254.npy").astype(np.float32)  # n x 2 list
net = torch.load("models/my_model_ackley")
inputs = torch.tensor(samples)
values = net(inputs).detach().numpy().flatten()
mesh = mtri.Triangulation(samples[:, 0], samples[:, 1])
plt.triplot(mesh, "bo-", alpha=0.5)  # 'bo-' 表示蓝色圆点和线段
plt.show()
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


# a = list(graph.keys())
# a.sort()
# print(a)

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
    # print(
    #     f"(Lk-, Lk+) of vertex {u} connected component = {len(llk.keys()), len(ulk.keys())}"
    # )


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
        if maxv > values[u]:  # 可能漏掉一些，和邻居相等的情况
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
def trace_min_numerical(u, lr=0.03, max_iter=50):
    path, vals = [], []
    for _ in range(max_iter):
        grad, val = eval_model(u)
        vals.append(val)
        u_next = u - lr * grad
        u = u_next
        path.append(u)
    return np.array(path), np.array(vals).flatten()


def trace_max_numerical(u, lr=0.03, max_iter=50):
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

# intline_N = {}

# for saddle, paths in intline_D.items():
#     min_path = paths[0]  # min path
#     max_path = paths[1]  # max path
#     intline_N[saddle] = [[[], []], [[], []]]  # 2 min paths, 2 max paths
#     for i, path in enumerate(min_path):
#         for u in path:
#             vertex = np.array(input[u])
#             down = trace_min_numerical(vertex)
#             up = trace_max_numerical(vertex)
#             nline_p = np.concatenate(
#                 (np.flip(down[0]), vertex.reshape(1, -1), up[0]), axis=0
#             )
#             nline_v = np.concatenate(
#                 (np.flip(down[1]), np.array([values[u]]), up[1]), axis=0
#             )
#             intline_N[saddle][0][i].append((nline_p, nline_v))

#     for i, path in enumerate(max_path):
#         for u in path:
#             vertex = np.array(input[u])
#             down = trace_min_numerical(vertex)
#             up = trace_max_numerical(vertex)
#             nline_p = np.concatenate(
#                 (np.flip(down[0]), vertex.reshape(1, -1), up[0]), axis=0
#             )
#             nline_v = np.concatenate(
#                 (np.flip(down[1]), np.array([values[u]]), up[1]), axis=0
#             )
#             intline_N[saddle][1][i].append((nline_p, nline_v))


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

ax0.scatter(
    saddle_vertices[:, 0],
    saddle_vertices[:, 1],
    saddle_values,
    color="black",
    s=50,
    marker="P",
    label="saddle",
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

# ax1 = fig.add_subplot(111)  # 2D version critical points
# ax1.scatter(
#     saddle_vertices[:, 0],
#     saddle_vertices[:, 1],
#     color="black",
#     marker="P",
#     s=50,
#     label="saddle",
# )
# ax1.scatter(
#     max_vertices[:, 0], max_vertices[:, 1], color="red", marker="^", s=50, label="max"
# )
# ax1.scatter(
#     min_vertices[:, 0], min_vertices[:, 1], color="blue", marker="v", s=50, label="min"
# )


for saddle, paths in intline_D.items():
    min_path = paths[0]  # min path
    max_path = paths[1]  # max path
    for path in min_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax0.plot(vs[:, 0], vs[:, 1], vals, color="blue")
        # ax1.plot(vs[:, 0], vs[:, 1], color="blue")
    for path in max_path:
        new_path = [saddle, *path]
        vs = np.array(list(map(lambda x: samples[x], new_path)))
        vals = np.array(list(map(lambda x: values[x], new_path)))
        ax0.plot(vs[:, 0], vs[:, 1], vals, color="red")
        # ax1.plot(vs[:, 0], vs[:, 1], color="red")


# colors = cm.viridis
# colors = colors(np.linspace(0, 1, 100))
# # draw numerical lines for vertex i
# for saddle, nlines in intline_N.items():
#     # print(nlines)
#     for nline_p, nline_v in nlines[0][0]:  # neumerical lines for discrete min line
#         # print("points = ", nline_p)
#         # print("values = ", nline_v)
#         ax0.plot(
#             nline_p[:, 0],
#             nline_p[:, 1],
#             nline_v,
#             color="cyan",
#         )

#         ax0.scatter(
#             nline_p[0:2, 0],
#             nline_p[0:2, 1],
#             nline_v[0:2],
#             c=np.linspace(0, 1, 2),
#             cmap="viridis",
#         )

#         break
#     break


# get the gradient of all of the critical points
# use a 3d bar plot to show the gradient of each kinds of critical points
# min points
grad_len = {"min": [], "max": [], "saddle": []}
for p in min_points:
    vertex = np.array(samples[p])
    grad, _ = eval_model(vertex)
    grad_length = np.linalg.norm(grad)
    grad_len["min"].append(grad_length)

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
# color
# 2D visualization
# verify gradients at critical points, threshhold close to 0
# refine critical points, gradient magnitude 下降方向，尽可能接近0， refine 位置
# refine grid, add samples, 在哪些地方加
# refine integral line traced from saddles
# QMS visualize

# saddle, wedge, 上升最大，下降最大，trace, saddle 有4个wedge (connected component)
# next step: critical points + **integral curves**
# numerical lines

# integral line vs numerical integral line
# refinement 的策略，critical point等等
# rd1: sparse, rd2: add samples, ...
# integral curve碰到一起如何处理(1. follow paper, junction point 2. 可以增加sample处理)


# 另外一大类方法（不同于2-manifolds paper） discrete morse理论


# Date: 2024.7.1
# 把 boundary上的点加入三角化的点
# 边界上的奇怪的三角形
# 2D 版本的line， critical
# saddle, 圈十字、、、标准表示方法
# 在saddle处reroute
# monkeysaddle 必须refine，仅在离散情况下出现
# 和一篇paper撞车了
# 尽可能少的sample来获得尽可能高质量的complex
# DMT里面有一个discrete gradient
# saddle desending 流向min, 可能两条descending 跑到同一个min上
# Sim C Piecewise constant
# piecewise linear 和 DMT基于的域不太一样

# ＰＬ的优缺点：更容易refine， max point是一个点，可解释性强一些
# DMT 的优缺点：不太好refine，max point 在三角形上，关于dmt的refine
# DMT: pairing, v-path, integral line和 v-path * (待做),
# 如果后续要refine，注意pairing变化的问题 (pairing决定了critical point，critical定义在cell上)，corner case少
# 用带mesh的数据 去做verification

# !critical point的refine(这个可能也要增加sample)，得到一个比较满意的critical point set
# junction point refinement/reroute

# numerical line trace问题
# ！！！！！离散情况下critical point的gradient和真实的gradient的比较
# ！！！！！2D的cp, intline的vis
# 将一些目前做了哪些事情，做一个演示，roadmap，做一个更加细密网格下的结果
# 可以复用给DMT的部分(3D的时候再说)
