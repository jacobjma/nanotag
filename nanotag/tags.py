from collections import defaultdict

import ipywidgets as widgets
import numpy as np
from bqplot import Scatter, ColorScale, LinearScale
from traitlets import Bool, Int, observe, default, Dict, Unicode

from nanotag.layout import VBox
from nanotag.utils import Array, link
from bqplot.colorschemes import CATEGORY10


class PointTags(VBox):
    x = Array(np.zeros((0,)), check_equal=False)
    y = Array(np.zeros((0,)), check_equal=False)

    data_overlay = Unicode()
    color_scheme = Unicode('plasma')
    visible = Bool(True)

    def __init__(self, data_fields=None, **kwargs):
        self._mark = Scatter(x=np.zeros((0,)), y=np.zeros((0,)),
                             scales=
                             {'color': ColorScale(scheme='plasma'),
                              'opacity': LinearScale(min=0, max=1)})
        self._mark.enable_move = True

        self._data_overlay_dropdown = widgets.Dropdown(description='Overlay', options=data_fields)
        self._color_scheme_dropdown = widgets.Dropdown(description='Colors', options=[
            'category',
            'viridis',
            'plasma',
            'inferno',
            'magma'])

        self._visible_checkbox = widgets.Checkbox(description='Visible')

        super().__init__(children=[self._data_overlay_dropdown,
                                   self._color_scheme_dropdown,
                                   self._visible_checkbox], **kwargs)

        link((self, 'x'), (self._mark, 'x'), check_broken=False)
        link((self, 'y'), (self._mark, 'y'), check_broken=False)
        link((self, 'visible'), (self._mark, 'visible'), check_broken=False)

        if data_fields is not None:
            self.add_traits(**{data_field: Array(np.zeros((0,))) for data_field in data_fields})
            self.observe(lambda *args: self._set_color(), tuple(data_fields) + ('data_overlay',))
            self.data_overlay = data_fields[0]

        self._data_fields = data_fields

        link((self, 'visible'), (self._visible_checkbox, 'value'))
        link((self, 'data_overlay'), (self._data_overlay_dropdown, 'value'))
        link((self, 'color_scheme'), (self._color_scheme_dropdown, 'value'))

    @observe('color_scheme')
    def _observe_color_scheme(self, change):
        if change['new'] == 'category':
            self.color_scale.colors = CATEGORY10
            self.color_scale.scheme = ''
        else:
            self.color_scale.colors = []
            self.color_scale.scheme = change['new']

    @property
    def empty(self):
        return not any((len(self.x), len(self.y)))

    @property
    def mark(self):
        return self._mark

    @property
    def color_scale(self):
        return self._mark.scales['color']

    @property
    def opacity_scale(self):
        return self._mark.scales['opacity']

    # @observe('color_scale')
    # def _observe_color_scale(self, change):
    #     scales = {key: scale for key, scale in self._mark.scales.items()}
    #     scales.update({'color': change['new']})
    #     self._mark.scales = scales

    def _set_color(self):
        self._mark.color = getattr(self, self.data_overlay)

    def add_to_canvas(self, canvas):
        self._mark.scales = {'x': canvas.x_scale, 'y': canvas.y_scale, 'color': self.color_scale,
                             'opacity': self.opacity_scale}
        if not self.mark in canvas.figure.marks:
            canvas.figure.marks = [self.mark] + canvas.marks

    def remove_from_canvas(self, canvas):
        marks = canvas.figure.marks
        marks = [mark for mark in marks if mark is not self.mark]
        canvas.figure.marks = marks

    def add_tags(self, x, y, data_fields=None):
        x = np.concatenate((self.x, x))
        y = np.concatenate((self.y, y))

        self.x = x
        self.y = y

        if data_fields is not None:
            for data_field, data in data_fields.items():
                data = np.concatenate((getattr(self, data_field), data))
                setattr(self, data_field, data)

    def delete_tags(self, indices):

        if (len(self.x) - len(indices)) == 0:
            self.x = np.zeros((0,))
            self.y = np.zeros((0,))
            for data_field in self._data_fields:
                setattr(self, data_field, np.zeros((0,)))

            return

        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)

        for data_field in self._data_fields:
            data = np.delete(getattr(self, data_field), indices)
            setattr(self, data_field, data)

    def reset(self):
        with self.hold_trait_notifications():
            self.x = np.zeros((0,))
            self.y = np.zeros((0,))

            for data_field in self._data_fields:
                setattr(self, data_field, np.zeros((0,)))

    def set_scales(self, scales):
        scales.update({'color': self.color_scale, 'opacity': self.opacity_scale})
        self._mark.scales = scales

    def serialize(self):
        serialized = {'x': self.x.tolist(), 'y': self.y.tolist()}
        serialized.update({data_field: getattr(self, data_field).tolist() for data_field in self._data_fields})
        return serialized

    def from_serialized(self, serialized):
        with self.hold_trait_notifications():
            self.x = serialized['x']
            self.y = serialized['y']

            for data_field in self._data_fields:
                try:
                    setattr(self, data_field, serialized[data_field])
                except KeyError:
                    pass


class PointTagSeries(widgets.VBox):
    index = Int(0)
    series = Dict()
    color_scheme = Unicode('plasma')

    def __init__(self, data_fields=None, **kwargs):

        self._point_tags = PointTags(data_fields=data_fields)

        super().__init__(children=[self._point_tags], **kwargs)

        link((self, 'color_scheme'), (self.point_tags, 'color_scheme'))

    @property
    def point_tags(self):
        return self._point_tags

    @property
    def mark(self):
        return self.point_tags.mark

    @property
    def x(self):
        return self.point_tags.x

    @property
    def y(self):
        return self.point_tags.y

    @property
    def empty(self):
        if not self.point_tags.empty:
            return False

        return all([self.empty_entry(index) for index in self.series.keys()])

    def empty_entry(self, index):
        if not index in self.series.keys():
            return True
        # print([value for value in self.series[index].values()])
        return not any([len(value) for value in self.series[index].values()])

    @default('point_tags')
    def _default_point_tags(self):
        return PointTags()

    @default('series')
    def _default_series(self):
        new_entry = lambda: {'x': np.zeros((0,)), 'y': np.zeros((0,))}
        return defaultdict(new_entry)

    @observe('index')
    def _observe_index(self, change):
        self.update_series(change['old'])
        self.update_current(change['new'])

    def update_series(self, index):
        # if self.point_tags.empty:
        #    return
        self.series[index] = self.point_tags.serialize()

    def update_current(self, index):
        if self.empty_entry(index):
            self.point_tags.reset()
            return

        self.point_tags.from_serialized(self.series[index])

    def add_tags(self, x, y, data_fields=None):
        self.point_tags.add_tags(x, y, data_fields)
        self.update_series(self.index)

    def delete_tags(self, indices):
        self.point_tags.delete_tags(indices)
        self.update_series(self.index)

    def set_scales(self, scales):
        self.point_tags.set_scales(scales)

    def reset(self):
        self.series = self._default_series()
        self.point_tags.reset()

    def serialize(self):
        self.update_series(self.index)
        # serialized = {}
        # for index in self.series.keys():
        #    serialized[index] = {key: value for key, value in self.series[index].items()}
        return self.series

    def from_serialized(self, serialized):
        # self.index = 0

        series = {}
        for index, entry in serialized.items():
            series[int(index)] = {key: np.array(value) for key, value in entry.items()}

        self.series = series
        self.update_current(self.index)
