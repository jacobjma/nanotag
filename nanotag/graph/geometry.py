import numpy as np


def regular_polygon(n, sidelength=1):
    points = np.zeros((n, 2))

    i = np.arange(n, dtype=np.int32)
    A = sidelength / (2 * np.sin(np.pi / n))

    points[:, 0] = A * np.sin(i * 2 * np.pi / n)
    points[:, 1] = A * np.cos(-i * 2 * np.pi / n)
    return points


def polygon_area(points):
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
