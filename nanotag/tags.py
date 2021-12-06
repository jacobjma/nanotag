from collections import defaultdict

import ipywidgets as widgets
import numpy as np
from bqplot import ColorScale, LinearScale, ScatterGL
from bqplot.colorschemes import CATEGORY10, CATEGORY20
from traitlets import Bool, Int, observe, default, Dict, Unicode, Any, directional_link, List

from nanotag.graph.utils import flatten_list_of_lists
from nanotag.layout import VBox
from nanotag.utils import Array, link


class PointTags(VBox):
    x = Array(np.zeros((0,)), check_equal=False)
    y = Array(np.zeros((0,)), check_equal=False)

    default_size = Int(20)
    data_overlay = Unicode()
    color_scheme = Unicode('plasma')

    visible = Bool(True)

    def __init__(self, data_overlays=None, **kwargs):
        self._mark = ScatterGL(x=np.zeros((0,)), y=np.zeros((0,)),
                               scales=
                               {'color': ColorScale(scheme='plasma'),
                                'opacity': LinearScale(min=0, max=1)})

        self._color_scheme_dropdown = widgets.Dropdown(description='Colors', options=[
            'category',
            'viridis',
            'plasma',
            'inferno',
            'magma'])

        self._visible_checkbox = widgets.Checkbox(description='Visible')

        box = [self._visible_checkbox, self._color_scheme_dropdown]

        if data_overlays is not None:
            self._data_overlay_dropdown = widgets.Dropdown(description='Overlay', options=data_overlays)
            box += [self._data_overlay_dropdown]
        else:
            self._data_overlay_dropdown = None

        super().__init__(children=box, **kwargs)

        link((self, 'x'), (self._mark, 'x'), check_broken=False)
        link((self, 'y'), (self._mark, 'y'), check_broken=False)
        link((self, 'default_size'), (self._mark, 'default_size'))
        link((self, 'visible'), (self._mark, 'visible'), check_broken=False)

        if data_overlays is not None:
            self.add_traits(**{data_overlay: Array(np.zeros((0,))) for data_overlay in data_overlays})
            self.observe(lambda *args: self._set_color(), tuple(data_overlays) + ('data_overlay',))
            self.data_overlay = data_overlays[0]
            link((self, 'data_overlay'), (self._data_overlay_dropdown, 'value'))

        self._data_overlays = data_overlays

        link((self, 'visible'), (self._visible_checkbox, 'value'))
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
    def marks(self):
        return [self._mark]

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

    # def add_to_canvas(self, canvas):
    #     self._mark.scales = {'x': canvas.x_scale, 'y': canvas.y_scale, 'color': self.color_scale,
    #                          'opacity': self.opacity_scale}
    #     if not self.mark in canvas.figure.marks:
    #         canvas.figure.marks = [self.mark] + canvas.marks
    #
    # def remove_from_canvas(self, canvas):
    #     marks = canvas.figure.marks
    #     marks = [mark for mark in marks if mark is not self.mark]
    #     canvas.figure.marks = marks

    def add_tags(self, x, y, data_overlays=None):
        x = np.concatenate((self.x, x))
        y = np.concatenate((self.y, y))

        self.x = x
        self.y = y

        if data_overlays is not None:
            for data_overlay, data in data_overlays.items():
                data = np.concatenate((getattr(self, data_overlay), data))
                setattr(self, data_overlay, data)

    def delete_tags(self, indices):

        if (len(self.x) - len(indices)) == 0:
            self.x = np.zeros((0,))
            self.y = np.zeros((0,))
            for data_overlay in self._data_overlays:
                setattr(self, data_overlay, np.zeros((0,)))

            return

        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)

        for data_overlay in self._data_overlays:
            try:
                data = np.delete(getattr(self, data_overlay), indices)
                setattr(self, data_overlay, data)
            except IndexError:
                pass

    def reset(self):
        with self.hold_trait_notifications():
            self.x = np.zeros((0,))
            self.y = np.zeros((0,))

            if self._data_overlays is None:
                return

            for data_overlay in self._data_overlays:
                setattr(self, data_overlay, np.zeros((0,)))

    def set_scales(self, scales):
        scales = {key: scale for key, scale in scales.items()}
        scales.update({'color': self.color_scale, 'opacity': self.opacity_scale})
        self._mark.scales = scales

    def serialize(self):
        serialized = {'x': self.x.tolist(), 'y': self.y.tolist()}

        if self._data_overlays is not None:
            serialized.update(
                {data_overlay: getattr(self, data_overlay).tolist() for data_overlay in self._data_overlays})
        return serialized

    def from_serialized(self, serialized):


        with self.hold_trait_notifications():
            self.x = serialized['x']
            self.y = serialized['y']

            if self._data_overlays is None:
                return

            for data_overlay in self._data_overlays:
                try:
                    setattr(self, data_overlay, serialized[data_overlay])
                except KeyError:
                    pass


class GraphTags(VBox):
    graph = Any()

    data_overlay = Unicode()
    edge_data_overlay = Unicode()

    nodes_color_scheme = Unicode('plasma')
    nodes_visible = Bool(True)
    edge_width = Int(1)
    visible = Bool(True)

    def __init__(self, data_overlays=None, edge_data_overlays=None, **kwargs):
        x_scale = LinearScale()
        y_scale = LinearScale()

        self._visible_checkbox = widgets.Checkbox(description='Visible')
        self._nodes_color_scheme_dropdown = widgets.Dropdown(description='Colors', options=[
            'category',
            'viridis',
            'plasma',
            'inferno',
            'magma'])

        self._edge_color_scheme_dropdown = widgets.Dropdown(description='Colors', options=[
            'category',
            'viridis',
            'plasma',
            'inferno',
            'magma'])

        self._edge_color_scale = ColorScale(scheme='plasma')
        self._edge_mark = ScatterGL(x=np.zeros((0,)), y=np.zeros((0,)),
                                    colors=['red'],
                                    scales={'x': x_scale, 'y': y_scale, 'color': self._edge_color_scale})

        self._nodes_color_scale = ColorScale(scheme='plasma')
        self._nodes_mark = ScatterGL(x=np.zeros((0,)), y=np.zeros((0,)),
                                     scales={'x': x_scale, 'y': y_scale, 'color': self._nodes_color_scale})

        box = [self._visible_checkbox, self._nodes_color_scheme_dropdown]

        if data_overlays is not None:
            self._data_overlay_dropdown = widgets.Dropdown(description='Nodes overlay', options=data_overlays)
            self.add_traits(**{data_overlay: Array(np.zeros((0,))) for data_overlay in data_overlays})
            self.observe(lambda *args: self._set_nodes_color(), tuple(data_overlays) + ('data_overlay',))
            self.data_overlay = data_overlays[0]
            link((self, 'data_overlay'), (self._data_overlay_dropdown, 'value'))
            box = box + [self._data_overlay_dropdown]

        if edge_data_overlays is not None:
            self._edge_data_overlay_dropdown = widgets.Dropdown(description='Edge overlay', options=edge_data_overlays)
            self.add_traits(**{data_overlay: Array(np.zeros((0,))) for data_overlay in edge_data_overlays})
            self.observe(lambda *args: self._set_edge_color(), tuple(edge_data_overlays) + ('edges_data_overlay',))

            self.edge_data_overlay = edge_data_overlays[0]
            link((self, 'edge_data_overlay'), (self._edge_data_overlay_dropdown, 'value'))
            box = box + [self._edge_data_overlay_dropdown]

        super().__init__(children=box, **kwargs)

        link((self, 'visible'), (self._visible_checkbox, 'value'))
        link((self, 'edge_width'), (self._edge_mark, 'default_size'))

        def transform_visible(x):
            if self.nodes_visible and x:
                return True
            return False

        directional_link((self, 'visible'), (self._edge_mark, 'visible'))
        directional_link((self, 'visible'), (self._nodes_mark, 'visible'), transform=transform_visible)
        directional_link((self, 'nodes_visible'), (self._nodes_mark, 'visible'))

        link((self, 'nodes_color_scheme'), (self._nodes_color_scheme_dropdown, 'value'))

    def _set_nodes_color(self):
        self._nodes_mark.color = getattr(self, self.data_overlay)

    def _set_edge_color(self):
        self._edge_mark.color = np.repeat(getattr(self, self.edge_data_overlay), 5)

    @observe('nodes_color_scheme')
    def _observe_nodes_color_scheme(self, change):
        if change['new'] == 'category':
            self.nodes_color_scale.colors = CATEGORY20
            self.nodes_color_scale.scheme = ''
        else:
            self.nodes_color_scale.colors = []
            self.nodes_color_scale.scheme = change['new']

    @observe('graph')
    def _observe_graph(self, *args):
        edges = self.graph.points[self.graph.edges]

        x = edges[..., 0]
        y = edges[..., 1]

        edges_x = np.linspace(x[:, 0], x[:, 1], 10).T
        edges_y = np.linspace(y[:, 0], y[:, 1], 10).T

        self._nodes_mark.x = self.graph.points[:, 0]
        self._nodes_mark.y = self.graph.points[:, 1]
        self._edge_mark.x = edges_x.ravel()
        self._edge_mark.y = edges_y.ravel()

    @property
    def marks(self):
        return [self._edge_mark, self._nodes_mark]

    @property
    def nodes_color_scale(self):
        return self._nodes_color_scale

    @property
    def edge_color_scale(self):
        return self._nodes_color_scale

    def set_scales(self, scales):
        nodes_scales = {key: scale for key, scale in scales.items()}
        nodes_scales.update({'color': self.nodes_color_scale})
        self._nodes_mark.scales = nodes_scales

        edge_scales = {key: scale for key, scale in scales.items()}
        edge_scales.update({'color': self.edge_color_scale})
        self._edge_mark.scales = edge_scales


class ContourTags(VBox):
    points = List()

    # x = Array(np.zeros((0, 2)), check_equal=False)
    # y = Array(np.zeros((0, 2)), check_equal=False)

    data_overlay = Unicode(None, allow_none=True)
    color_scheme = Unicode('plasma')
    default_size = Int(1)
    padding = Int(5)
    width = Int(5)
    visible = Bool(True)

    def __init__(self, data_overlays=None, enable_move=True, **kwargs):
        x_scale = LinearScale()
        y_scale = LinearScale()
        self._visible_checkbox = widgets.Checkbox(description='Visible')
        self._mark = ScatterGL(x=np.zeros((0,)), y=np.zeros((0,)),
                               colors=['orange'],
                               scales={'x': x_scale, 'y': y_scale,
                                       'color': ColorScale(scheme='plasma'),
                                       'opacity': LinearScale(min=0, max=1)})

        self._color_scheme_dropdown = widgets.Dropdown(description='Colors', options=[
            'category',
            'viridis',
            'plasma',
            'inferno',
            'magma'])

        box = [self._visible_checkbox, self._color_scheme_dropdown]

        if data_overlays is not None:
            self.data_overlay = data_overlays[0]
            self._data_overlay_dropdown = widgets.Dropdown(description='Overlay', options=data_overlays)
            self.add_traits(**{data_overlay: Array(np.zeros((0,))) for data_overlay in data_overlays})
            self.observe(lambda *args: self._set_color(), tuple(data_overlays) + ('data_overlay',))

            link((self, 'data_overlay'), (self._data_overlay_dropdown, 'value'))
            box = box + [self._data_overlay_dropdown]

        super().__init__(children=box, **kwargs)

        self._data_overlays = data_overlays

        link((self, 'visible'), (self._visible_checkbox, 'value'))
        link((self, 'default_size'), (self._mark, 'default_size'))
        # link((self, 'x'), (self._mark, 'x'), check_broken=False)
        # link((self, 'y'), (self._mark, 'y'), check_broken=False)
        link((self, 'width'), (self._mark, 'default_size'))
        link((self, 'visible'), (self._mark, 'visible'), check_broken=False)
        link((self, 'color_scheme'), (self._color_scheme_dropdown, 'value'))

    @observe('color_scheme')
    def _observe_nodes_color_scheme(self, change):
        if change['new'] == 'category':
            self.color_scale.colors = CATEGORY20
            self.color_scale.scheme = ''
        else:
            self.color_scale.colors = []
            self.color_scale.scheme = change['new']

    @observe('padding')
    def _set_color(self, *args):
        if self.data_overlay is None:
            return

        if len(self.points) == 0:
            return

        data = getattr(self, self.data_overlay)
        if len(data) == 0:
            return

        contour_lengths = [len(points) for points in self.points]
        colors = np.zeros(sum(contour_lengths) * self.padding)

        i = 0
        for j, contour_length in enumerate(contour_lengths):
            colors[i:i + contour_length * self.padding] = data[j % len(data)]
            i += contour_length * self.padding

        self._mark.color = colors

    @observe('points', 'padding')
    def _observe_xy(self, *args):
        x = []
        y = []
        for points in self.points:
            x.append([np.linspace(points[i - 1][0], points[i][0], self.padding) for i in range(len(points))])
            y.append([np.linspace(points[i - 1][1], points[i][1], self.padding) for i in range(len(points))])

        x = np.array(flatten_list_of_lists(x))
        y = np.array(flatten_list_of_lists(y))

        self._mark.x = x.ravel()
        self._mark.y = y.ravel()

    @property
    def polygons(self):
        polygons = []
        for x, y in zip(self.x, self.y):
            polygons.append(np.array([x, y]).T)
        return polygons

    @property
    def marks(self):
        return [self._mark]

    @property
    def color_scale(self):
        return self._mark.scales['color']

    @property
    def opacity_scale(self):
        return self._mark.scales['opacity']

    def set_scales(self, scales):
        scales.update({})
        scales.update({'color': self.color_scale, 'opacity': self.opacity_scale})
        self._mark.scales = scales

    def reset(self):
        with self.hold_trait_notifications():
            self.x = np.zeros((0,))
            self.y = np.zeros((0,))

            if self._data_overlays is None:
                return

            for data_overlay in self._data_overlays:
                setattr(self, data_overlay, np.zeros((0,)))

    def serialize(self):
        serialized = {'points': [polygon.tolist() for polygon in self.points]}
        if self._data_overlays is not None:
            serialized.update(
                {data_overlay: getattr(self, data_overlay).tolist() for data_overlay in self._data_overlays})
        return serialized

    def from_serialized(self, serialized):
        with self.hold_trait_notifications():
            self.points = [np.array(polygon) for polygon in serialized['points']]

            if self._data_overlays is None:
                return

            for data_overlay in self._data_overlays:
                try:
                    setattr(self, data_overlay, serialized[data_overlay])
                except KeyError:
                    pass


class PointTagSeries(widgets.VBox):
    frame_index = Int(0)
    series = Dict()
    color_scheme = Unicode('plasma')

    def __init__(self, data_fields=None, enable_move=True, **kwargs):

        self._point_tags = PointTags(data_fields=data_fields, enable_move=enable_move)

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

        return all([self.empty_entry(frame_index) for frame_index in self.series.keys()])

    def empty_entry(self, frame_index):
        if not frame_index in self.series.keys():
            return True
        # print([value for value in self.series[index].values()])
        return not any([len(value) for value in self.series[frame_index].values()])

    @default('point_tags')
    def _default_point_tags(self):
        return PointTags()

    @default('series')
    def _default_series(self):
        new_entry = lambda: {'x': [], 'y': []}
        return defaultdict(new_entry)

    @observe('frame_index')
    def _observe_frame_index(self, change):
        self.update_series(change['old'])
        self.update_current(change['new'])

    @observe('series')
    def _observe_series(self, change):
        self.update_current(self.frame_index)

    def update_series(self, frame_index):
        # if self.point_tags.empty:
        #    return
        self.series[frame_index] = self.point_tags.serialize()

    def update_current(self, frame_index):
        if self.empty_entry(frame_index):
            self.point_tags.reset()
            return

        self.point_tags.from_serialized(self.series[frame_index])

    def add_tags(self, x, y, data_fields=None):
        self.point_tags.add_tags(x, y, data_fields)
        self.update_series(self.frame_index)

    def delete_tags(self, indices):
        self.point_tags.delete_tags(indices)
        self.update_series(self.frame_index)

    def set_scales(self, scales):
        self.point_tags.set_scales(scales)

    def reset(self):
        self.series = self._default_series()
        self.point_tags.reset()

    def serialize(self):
        self.update_series(self.frame_index)
        # serialized = {}
        # for index in self.series.keys():
        #    serialized[index] = {key: value for key, value in self.series[index].items()}
        return self.series

    def from_serialized(self, serialized):
        # self.index = 0

        series = {}
        for frame_index, entry in serialized.items():
            series[int(frame_index)] = {key: value for key, value in entry.items()}

        self.series = series
        self.update_current(self.frame_index)
