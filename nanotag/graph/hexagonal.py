import bisect

import numpy as np
from nanotag.graph.construct import knn_graph
from nanotag.graph.geometry import regular_polygon
from nanotag.graph.rmsd import rmsd_qcp


def prioritized_greedy_two_coloring(adjacency, priority):
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    first_node = np.argmin(priority)

    queue = [(priority[first_node], first_node)]

    while queue:
        _, node = queue.pop(0)
        neighbors = np.array(adjacency[node])

        if len(neighbors) == 0:
            continue

        neighbors = neighbors[labels[neighbors] == -1]
        labels[neighbors] = labels[node] == 0

        for neighbor in neighbors:
            bisect.insort(queue, (priority[neighbor], neighbor))

    return labels


def adjacency_segments(adjacency, points):
    segments = []
    for i, adjacent in adjacency.items():
        segment = points[adjacent]
        segment = np.vstack([[points[i]], segment])
        segment = segment - segment.mean(axis=0, keepdims=True)
        order = np.argsort(np.arctan2(segment[1:, 0], segment[1:, 1]))
        segment[1:] = segment[1:][order]
        segments.append(segment)
    return segments


def assign_sublattice(adjacency, points, lattice_constant):
    segments = adjacency_segments(adjacency, points)
    template = np.vstack(([[0., 0.]], regular_polygon(3))) * lattice_constant

    rmsd = rmsd_qcp(template, segments)

    return prioritized_greedy_two_coloring(adjacency, rmsd)
