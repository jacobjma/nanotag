from collections import defaultdict


def adjacency_to_edges(adjacency):
    edges = []
    for i, adjacent in adjacency.items():
        for j in adjacent:
            edges.append([i, j])
    return edges


def faces_to_adjacency(faces, num_nodes):
    adjacency = defaultdict(set)
    for face in faces:
        for i in range(len(face)):
            adjacency[face[i]].add(face[i - 1])
            adjacency[face[i - 1]].add(face[i])

    return {i: list(adjacency[i]) for i in range(num_nodes)}


def faces_to_edges(faces):
    edges = set()
    for face in faces:
        if len(face) < 2:
            continue

        for i in range(len(face)):
            edges.add(frozenset({face[i - 1], face[i]}))

    return [list(edge) for edge in edges]
