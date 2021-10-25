def subgraph_adjacency(node_indices, adjacency, relabel=True):
    if relabel:
        backward = {i: node_index for i, node_index in enumerate(node_indices)}
        forward = {value: key for key, value in backward.items()}
        node_indices = set(node_indices)
        adjacency = [set(adjacency[backward[i]]).intersection(node_indices) for i in range(len(node_indices))]
        return {i: [forward[adjacent] for adjacent in set(adjacency[backward[i]]).intersection(node_indices)] for i in
                range(len(node_indices))}
    else:
        node_indices = set(node_indices)
        return {i: list(set(adjacency[i]).intersection(node_indices)) for i in node_indices}