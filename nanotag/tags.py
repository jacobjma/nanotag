import numpy as np
from bqplot import Scatter
from matplotlib.colors import rgb2hex
from traitlets import HasTraits, observe, Bool, Instance, directional_link, List, Int, Dict

from nanotag.utils import Array, link, get_colors_from_cmap
from nanotag.artists import PointArtist


class PointTags(HasTraits):
    x = Array(np.zeros((0,)), check_equal=False)
    y = Array(np.zeros((0,)), check_equal=False)
    labels = Array(np.zeros((0,)), allow_none=True, check_equal=False)
    visible = Bool(True)

    def __init__(self, **kwargs):
        self._artist = PointArtist(tags=self)
        super().__init__(**kwargs)

    @property
    def artist(self):
        return self._artist

    def add_tags(self, x, y, labels):
        x = np.concatenate((self.x, x))
        y = np.concatenate((self.y, y))
        labels = np.concatenate((self.labels, labels))

        self.x = x
        self.y = y
        self.labels = labels

    def delete_tags(self, indices):
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.labels = np.delete(self.labels, indices)

    def serialize(self):
        return {'type': 'point-tags', 'x': self.x.tolist(), 'y': self.y.tolist(), 'labels': self.labels.tolist()}

    @classmethod
    def from_serialized(cls, serialized):
        if not serialized['type'] == 'point-tags':
            raise RuntimeError()

        return cls(x=serialized['x'], y=serialized['y'], labels=serialized['labels'])


def tags_from_serialized(serialized):
    if serialized['type'] == 'point-tags':
        return PointTags.from_serialized(serialized)
    else:
        raise RuntimeError()
