import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon


def add_edges_to_mpl_plot(points, edges, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()
    line_collection = LineCollection(points[edges], **kwargs)
    ax.add_collection(line_collection)
    ax.autoscale()
    return line_collection


def add_polygons_to_mpl_plot(polygons, facecolors=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()

    patches = []
    for polygon in polygons:
        patches.append(Polygon(polygon, closed=True))

    patch_collection = PatchCollection(patches, facecolors=facecolors, **kwargs)
    ax.add_collection(patch_collection)
    ax.autoscale()
    return patch_collection


def get_colors_from_cmap(c, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if vmin is None:
        vmin = np.nanmin(c)

    if vmax is None:
        vmax = np.nanmax(c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    c = np.array(c, dtype=float)

    valid = np.isnan(c) == 0
    colors = np.zeros((len(c), 4))
    colors[valid] = cmap(norm(c[valid]))

    return colors