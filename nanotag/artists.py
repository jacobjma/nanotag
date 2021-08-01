import numpy as np
from bqplot import Scatter
from matplotlib.colors import rgb2hex
from traitlets import HasTraits, observe, Bool, directional_link, Any
from traitlets import link

from nanotag.utils import link, get_colors_from_cmap


class PointArtist(HasTraits):
    tags = Any(allow_none=True)
    visible = Bool(True)

    def __init__(self, **kwargs):
        self._mark = Scatter(x=np.zeros((0,)), y=np.zeros((0,)), colors=['red'])
        self._mark.enable_move = True

        self._artist_links = []

        super().__init__(**kwargs)

    @observe('tags')
    def _observe_point_tags(self, change):
        for artist_link in self._artist_links:
            artist_link.unlink()

        self._artist_links = []
        # self._artist_links.append(link((self, 'visible'), (self.mark, 'visible')))
        self._artist_links.append(link((self.tags, 'x'), (self.mark, 'x'), check_broken=False))
        self._artist_links.append(link((self.tags, 'y'), (self.mark, 'y'), check_broken=False))
        labels_link = directional_link((change['new'], 'labels'), (self.mark, 'colors'),
                                       transform=self.labels_to_colors)
        self._artist_links.append(labels_link)

    @property
    def mark(self):
        return self._mark

    def add_to_canvas(self, canvas):
        self._mark.scales = {'x': canvas.x_scale, 'y': canvas.y_scale}
        if not self.mark in canvas.figure.marks:
            canvas.figure.marks = [self.mark] + canvas.marks

    def remove_from_canvas(self, canvas):
        marks = canvas.figure.marks
        marks = [mark for mark in marks if mark is not self.mark]
        canvas.figure.marks = marks

    def labels_to_colors(self, labels):
        colors = get_colors_from_cmap(labels, cmap='tab10', vmin=0, vmax=8)
        colors = [rgb2hex(color) for color in colors]
        return colors
