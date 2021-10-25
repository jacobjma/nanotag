import numpy as np

def bfs(adjacency, start):
    visited = [False] * (max(adjacency) + 1)

    queue = []
    queue.append(start)
    visited[start] = True
    order = []

    while queue:
        s = queue.pop(0)
        order.append(s)
        for i in adjacency[s]:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
    return order, visited


def connected_components(adjacency):
    to_start = set(adjacency.keys())
    components = []
    while len(to_start) > 0:
        n = to_start.pop()
        order, visited = bfs(adjacency, n)

        to_start = to_start - set(order)
        components.append(order)

    return components


def traverse_left_most_outer(points, adjacency, counter_clockwise=True):
    left_most = np.where((points[:, 0] == np.min(points[:, 0])))[0]
    left_bottom_most = left_most[np.argmin(points[left_most, 1])]

    adjacent = adjacency[left_bottom_most]
    angles = np.arctan2(points[adjacent][:, 1] - points[left_bottom_most, 1],
                        points[adjacent][:, 0] - points[left_bottom_most, 0])

    if counter_clockwise:
        edge = (left_bottom_most, adjacent[np.argmin(angles)])
    else:
        edge = (left_bottom_most, adjacent[np.argmax(angles)])

    outer_path = [edge]
    while (edge != outer_path[0]) or (len(outer_path) == 1):
        next_adjacent = np.array(adjacency[edge[1]])
        j = next_adjacent[(np.nonzero(next_adjacent == edge[0])[0][0] - 1) % len(next_adjacent)]
        edge = (edge[1], j)
        outer_path.append(edge)

    return [edge[0] for edge in outer_path]