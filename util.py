import numpy as np


def get_rotation_matrix_2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# search for when function = 0
# assum min < 0, max > 0
def binary_search(minx, maxx, function, max_depth=8, depth=0):
    x = (minx + maxx) / 2.0
    if depth == max_depth:
        return x
    out = function(x)
    print(out)
    if out > 0:
        return binary_search(minx=minx, maxx=(maxx + x) / 2.0, function=function, max_depth=max_depth, depth=depth + 1)
    else:
        return binary_search(minx=(minx + x) / 2.0, maxx=maxx, function=function, max_depth=max_depth, depth=depth + 1)