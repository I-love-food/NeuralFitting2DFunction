import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.quiver import quiver


def f(x, y):
    # return x**2 + y**2
    return np.sin(x) + np.sin(y)


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
X1, Y1 = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
Z = f(X, Y)
Z1 = f(X1, Y1)

Fx = np.gradient(Z1, axis=1)  # 对x的偏导数
Fy = np.gradient(Z1, axis=0)  # 对y的偏导数

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制原始函数的等高线图
CS = ax.contourf(X, Y, f(X, Y), levels=50, cmap="viridis")

# 绘制梯度向量场
Q = ax.quiver(
    X1,
    Y1,
    Fx,
    Fy,
    # linewidths=2,  # 箭头线宽
    # headwidth=3,  # 箭头头宽度
    # headlength=1.5,
    scale=30,
)

# 添加颜色条
plt.colorbar(CS, ax=ax, label="f(x, y) = x^2 + y^2")
plt.colorbar(Q, ax=ax, label="Gradient Magnitude")

# 设置图形标题和轴标签
ax.set_title("Vector Field of the Gradient of f(x, y) = sin(x) + sin(y)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# 显示图形
plt.show()


# edelsbrunner rerouting
# Timo, Follow 原函数 gradient: <prerequisite PL的critical 尽可能精确>
# volario, Follow PL函数 gradient: Topology Diagrams of Scalar Fields in Scientific Visualisation

# 一种潜在方法：尽可能follow edge，碰见冲突，follow 原+函数gradient

# next step: <prerequisite PL的critical 尽可能精确>


# 论文scope、claim什么contribution
# 主要工作：INR -> MSC 或者 **sampling based approach for data recover MSC**
# 论文写作时间：2月，最好尽快有一个草稿

# related work是最容易开始的，survay leterature.
# motivation

# related work, intro, (method, result), comparison (传统的方法需要先离散化再做，我们的更快 比如说).

# 论文contribution
# critical refine的策略
# integral line refine策略 (Timo, ...)
# 尽快的获得MSC，快
