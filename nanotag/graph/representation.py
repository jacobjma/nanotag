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


def faces_to_quad_edge(faces):
    quad_edge = defaultdict(lambda: list([None, None]))
    for i, face in enumerate(faces):
        for j in range(len(face)):
            quad_edge[(face[j - 1], face[j])][0] = i
            quad_edge[(face[j], face[j - 1])][-1] = i
    return quad_edge


def edges_to_adjacency(edges):
    adjacency = defaultdict(list)
    for edge in edges:
        edge = list(edge)
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
    return adjacency
