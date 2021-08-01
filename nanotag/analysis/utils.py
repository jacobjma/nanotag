def is_position_inside_image(positions, shape, margin=0):
    mask = ((positions[:, 0] >= -margin) & (positions[:, 1] >= -margin) &
            (positions[:, 0] < shape[0] + margin) & (positions[:, 1] < shape[1] + margin))
    return mask