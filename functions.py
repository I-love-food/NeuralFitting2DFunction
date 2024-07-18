import numpy as np


def gaussian_function(points, mu1=0, mu2=0, sigma=1):
    x = points[:, 0]
    y = points[:, 1]
    two_sigma2 = 2 * sigma * sigma
    return (
        1
        / (np.pi * two_sigma2)
        * np.exp(-((x - mu1) ** 2 + (y - mu2) ** 2) / two_sigma2)
    )


def ackley_function(points, a=20, b=0.2, c=2 * np.pi, d=2):
    x = points[:, 0]
    y = points[:, 1]
    exp1 = np.exp(-b * np.sqrt(1 / d * (x**2 + y**2)))
    exp2 = 1 / d * (np.cos(c * x) + np.cos(c * y))
    return -a * exp1 - exp2 + a + np.exp(1)


def volcano_function(points):
    x = points[:, 0]
    y = points[:, 1]
    x2y2 = (x) ** 2 + (y) ** 2
    sine = np.sin(x2y2)
    w = 1 - np.power(1.5, (-x2y2))
    return 0.5 * (sine - sine * w)
