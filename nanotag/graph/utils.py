import numpy as np

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def check_clockwise(polygon):
    clockwise = False
    signed_area = 0.
    for i in range(len(polygon)):
        signed_area += polygon[i - 1, 0] * polygon[i, 1] - polygon[i, 0] * polygon[i - 1, 1]
    if signed_area > 0.:
        clockwise = True
    return clockwise


def bounding_box_from_points(points, margin=0):
    return np.array([[np.min(points[:, 0]) - margin, np.max(points[:, 0]) + margin],
                     [np.min(points[:, 1]) - margin, np.max(points[:, 1]) + margin]])
