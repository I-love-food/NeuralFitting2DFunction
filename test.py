import numpy as np
import matplotlib.pyplot as plt

# 定义空间范围
x = np.linspace(-3.0, 3.0, 1000)
y = np.linspace(-3.0, 3.0, 1000)

# 生成网格数据
X, Y = np.meshgrid(x, y)

# 定义二维函数，例如：z = x^2 + y^2
Z = X**2 + Y**2

# 绘制等高线图
plt.contourf(X, Y, Z, cmap="viridis", antialiased=True)  # 使用contourf填充等高线之间的颜色

# 添加颜色条
plt.colorbar()

# 显示图形
plt.show()
