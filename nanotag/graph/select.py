def grow(indices, adjacency, remove_initial=False):
    new_indices = [item for i in indices for item in adjacency[i]] + list(indices)
    if remove_initial:
        return list(set(new_indices) - set(indices))
    else:
        return list(set(new_indices))
