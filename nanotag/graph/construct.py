import numpy as np
from scipy.spatial import KDTree


def order_adjacency_clockwise(points, adjacency, counter_clockwise=False):
    for node, adjacent in adjacency.items():
        centered = points[adjacent] - points[node]
        order = np.arctan2(centered[:, 0], centered[:, 1])
        adjacency[node] = [x for _, x in sorted(zip(order, adjacent), reverse=counter_clockwise)]
    return adjacency


def knn_graph(points, k, clockwise=False):
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    adjacency = {i: indices[i, 1:].tolist() for i in range(len(indices))}

    if clockwise:
        adjacency = order_adjacency_clockwise(points, adjacency)

    return adjacency
