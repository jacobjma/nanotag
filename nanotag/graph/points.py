import numba as nb
import numpy as np
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist


@nb.jit(nopython=True)
def point_in_polygon(point, polygon):
    n = len(polygon)
    inside = False
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def points_in_polygon(points, polygon, return_indices=False):
    if return_indices:
        return np.array([i for i, point in enumerate(points) if point_in_polygon(point, polygon)])
    else:
        return np.array([point for point in points if point_in_polygon(point, polygon)])


def points_in_box(positions, size, corner=(0, 0), margin=0):
    mask = ((positions[:, 0] >= -margin) & (positions[:, 1] >= -margin) &
            (positions[:, 0] < size[0] + margin) & (positions[:, 1] < size[1] + margin))
    return mask


def greedy_assign(points1, points2, cutoff=np.inf):
    d = cdist(points1, points2)
    d[d > cutoff] = 1e12
    assignment1, assignment2 = linear_sum_assignment(d)
    valid = d[assignment1, assignment2] < cutoff
    return assignment1[valid], assignment2[valid]