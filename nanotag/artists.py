import contextlib

import numpy as np
from bqplot import LinearScale, ColorScale, Scatter
from bqplot_image_gl import ImageGL
from traitlets import HasTraits, observe, default, List, link, Float, Unicode, Bool, validate

from nanotag.utils import Array


class Artist(HasTraits):
    visible = Bool()

    def _add_to_canvas(self, canvas):
        raise NotImplementedError()

    @property
    def limits(self):
        raise NotImplementedError()


class ImageArtist(Artist):
    image = Array(check_equal=False)
    extent = List(allow_none=True)
    power = Float(1.)
    color_scheme = Unicode('Greys')
    autoadjust_colorscale = Bool(True)

    def __init__(self, **kwargs):
        self._color_scale = ColorScale(colors=['black', 'white'], min=0, max=1)

        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'),
                  'image': self._color_scale}

        self._mark = ImageGL(image=np.zeros((0, 0)), scales=scales)

        link((self._mark, 'visible'), (self, 'visible'))
        super().__init__(**kwargs)

    @default('extent')
    def _default_extent(self):
        return None

    @default('image')
    def _default_image(self):
        return np.zeros((0, 0, 3))

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale,
                  'image': self._color_scale}

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    def update_image(self, *args):
        image = self.image
        if self.power != 1:
            image = image ** self.power

        image = np.swapaxes(image, 0, 1)

        if self.extent is None:
            with self._mark.hold_sync():
                self._mark.x = [-.5, image.shape[0] - .5]
                self._mark.y = [-.5, image.shape[1] - .5]
                self._mark.image = image
        else:
            self._mark.image = image

        if (not self.image.size == 0) & self.autoadjust_colorscale:
            with self._mark.hold_sync():
                self._mark.scales['image'].min = float(image.min())
                self._mark.scales['image'].max = float(image.max())

    @observe('image')
    def _observe_image(self, *args):
        self.update_image(*args)

    @observe('extent')
    def _observe_extent(self, change):
        if self.image.size == 0:
            return

        sampling = ((change['new'][0][1] - change['new'][0][0]) / self.image.shape[0],
                    (change['new'][1][1] - change['new'][1][0]) / self.image.shape[1])

        self._mark.x = [value - .5 * sampling[0] for value in change['new'][0]]
        self._mark.y = [value - .5 * sampling[1] for value in change['new'][1]]

    @property
    def display_sampling(self):
        if self.image.size == 0:
            return

        extent_x = self._mark.x[1] - self._mark.x[0]
        extent_y = self._mark.y[1] - self._mark.y[0]
        pixel_extent_x = self.image.shape[0]
        pixel_extent_y = self.image.shape[1]
        return (extent_x / pixel_extent_x, extent_y / pixel_extent_y)

    def position_to_index(self, position):
        sampling = self.display_sampling
        px = int(np.round((position[0] - self._mark.x[0]) / sampling[0] - .5))
        py = int(np.round((position[1] - self._mark.y[0]) / sampling[1] - .5))
        return [px, py]

    def indices_to_position(self, indices):
        sampling = self.display_sampling
        px = (indices[0] + .5) * sampling[0] + self._mark.x[0]
        py = (indices[1] + .5) * sampling[1] + self._mark.y[0]
        return [px, py]

    @property
    def limits(self):
        if (self.extent is None) or (self.display_sampling is None):
            return [(-.5, self.image.shape[0] - .5), (-.5, self.image.shape[1] - .5)]

        return [tuple([l - .5 * s for l in L]) for L, s in zip(self.extent, self.display_sampling)]


class ScatterArtist(HasTraits):
    points = Array(np.array((0, 2)), check_equal=False)
    labels = Array()
    visible = Bool(True)
    _updating = False

    def __init__(self, **kwargs):
        self._mark = Scatter(x=np.zeros((1,)), y=np.zeros((1,)), colors=['red'])

        def observe_x_and_y(*args):
            if self._updating:
                return
            with self._busy_updating():
                min_length = min(len(self._mark.x), len(self._mark.y))
                x = self._mark.x[:min_length]
                y = self._mark.y[:min_length]
                self.points = np.array([x, y]).T

        self._mark.observe(observe_x_and_y, ('x', 'y'))

        super().__init__(**kwargs)
        link((self, 'visible'), (self._mark, 'visible'))

    @validate('points')
    def _validate_points(self, proposal):
        points = np.array(proposal['value'])

        if points.size == 0:
            return np.zeros((0, 2))

        if len(points.shape) == 1:
            points = points[None]

        return points


    @contextlib.contextmanager
    def _busy_updating(self):
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    @observe('points')
    def _observe_points(self, *args):
        if self._updating:
            return
        with self._busy_updating():
            with self._mark.hold_sync():
                if len(self.points) > 0:
                    self._mark.y = self.points[:, 0]
                    self._mark.x = self.points[:, 1]
                else:
                    self._mark.y = np.zeros((0,))
                    self._mark.x = np.zeros((0,))

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale}

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    @property
    def limits(self):
        if len(self._mark.x) > 0:
            x_lim = (self._mark.x.min(), self._mark.x.max())
        else:
            x_lim = (0., 0.)

        if len(self._mark.x) > 0:
            y_lim = (self._mark.y.min(), self._mark.y.max())
        else:
            y_lim = (0., 0.)

        return [x_lim, y_lim]

    @observe('labels')
    def _observe_labels(self, change):
        with self._mark.hold_sync():
            colors = get_colors_from_cmap(self.labels, cmap='tab10', vmin=0, vmax=8)
            colors = [rgb2hex(color) for color in colors]
            self._mark.colors = colors
#
# class ScatterArtist(HasTraits):
#     points = Array()
#     color = Any()
#     visible = Bool()
#
#     def __init__(self, colors=None, **kwargs):
#         if colors is None:
#             colors = ['red']
#
#         color_scale = ColorScale(scheme='plasma')
#
#         scales = {'x': LinearScale(allow_padding=False),
#                   'y': LinearScale(allow_padding=False, orientation='vertical'),
#                   'color': color_scale,
#                   'size': LinearScale(min=0, max=1),
#                   }
#         mark = Scatter(x=np.zeros((1,)), y=np.zeros((1,)), scales=scales, colors=colors)
#
#         self._mark = mark
#
#         super().__init__(**kwargs)
#         link((self, 'color'), (mark, 'color'))
#         link((self._mark, 'visible'), (self, 'visible'))
#
#     @observe('color')
#     def _observe_color(self):
#         self._mark.color = self.color
#
#     @observe('x')
#     def _observe_x(self, change):
#         self._mark.x = self.x
#
#     @observe('y')
#     def _observe_y(self, change):
#         self._mark.y = self.y
#
#     def _add_to_canvas(self, canvas):
#         scales = {'x': canvas.figure.axes[0].scale,
#                   'y': canvas.figure.axes[1].scale}
#
#         try:
#             scales['color'] = self._mark.scales['color']
#         except KeyError:
#             pass
#
#         try:
#             scales['size'] = self._mark.scales['size']
#         except KeyError:
#             pass
#
#         self._mark.scales = scales
#         canvas.figure.marks = [self._mark] + canvas.figure.marks
#
#     @property
#     def limits(self):
#         return [(self.x.min(), self.x.max()), (self.y.min(), self.y.max())]
#
#     # @default('color')
#     # def _default_colors(self):
#     #     return ['red']
