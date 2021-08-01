import ipywidgets as widgets
import numpy as np
from bqplot import Figure, Axis, LinearScale, Lines, OrdinalScale, Scatter, ColorScale
from ipyevents import Event
from traitlets import observe, List, Int, validate, Unicode, Instance, default, HasTraits

from nanotag.utils import link, Array


class EditTimelineTags(HasTraits):
    label = Int()

    def __init__(self, tags, **kwargs):
        self._tags = tags
        self._event = Event(watched_events=['click'])
        super().__init__(**kwargs)

    def activate(self, timeline):
        self._event.source = timeline

        def handle_event(event):
            x = timeline.pixel_to_domain(event['offsetX'])
            tags = self._tags
            if event['altKey'] is True:
                indices = [np.argmin(np.abs(tags.t - x))]
                tags.delete_tags(indices)
            else:
                tags.add_tags([int(np.round(x))], [self.label])

        self._event.on_dom_event(handle_event)

    def deactivate(self, canvas):
        self._event.reset_callbacks()


class TimelineTags(HasTraits):
    t = Array(np.zeros((0,)), check_equal=False)
    color_scale = Instance(ColorScale)
    labels = Array(np.zeros((0,)), check_equal=False)

    def __init__(self, row, default_t=None, **kwargs):
        self._mark = Scatter(x=np.zeros((0,)), y=np.zeros((0,)))
        self._mark.enable_move = True
        self._mark.restrict_x = True
        self._row = row

        if default_t is None:
            default_t = np.zeros((0,))

        self._default_t = default_t

        super().__init__(**kwargs)
        link((self, 'labels'), (self._mark, 'color'), check_broken=False)
        link((self, 't'), (self._mark, 'x'), check_broken=False)

    @property
    def empty(self):
        return not any((len(self.t), len(self.labels)))

    @default('color_scale')
    def _default_color_scale(self):
        return ColorScale()

    @validate('t')
    def _validate_t(self, proposal):
        return proposal['value'].astype(np.int64)

    @observe('t')
    def _observe_t(self, change):
        self._mark.y = [self._row] * len(change['new'])

    @property
    def mark(self):
        return self._mark

    def set_scales(self, scales):
        scales.update({'color': self.color_scale})
        self._mark.scales = scales

    def reset(self):
        self.t = self._default_t
        self.labels = np.zeros((0,), dtype=np.int64)

    def add_tags(self, t, labels):
        self.t = np.concatenate((self.t, t))
        self.labels = np.concatenate((self.labels, labels))

    def delete_tags(self, indices):
        self.t = np.delete(self.t, indices)
        self.labels = np.delete(self.labels, indices)

    def serialize(self):
        return {'name': self._row, 't': self.t.tolist(), 'labels': self.labels.tolist()}

    def from_serialized(self, serialized):
        self._row = serialized['name']
        self.t = serialized['t']
        self.labels = serialized['labels']


class Timeline(widgets.VBox):
    tags = List()
    frame_index = Int(0)
    num_frames = Int()

    def __init__(self, fig_margin=None, width=450, **kwargs):
        x_scale = LinearScale(allow_padding=False, min=0)
        y_scale = OrdinalScale(allow_padding=False)

        link((self, 'num_frames'), (x_scale, 'max'))

        scales = {'x': x_scale, 'y': y_scale}

        x_axis = Axis(scale=scales['x'])
        y_axis = Axis(scale=scales['y'], orientation='vertical', grid_lines='none')

        fig_margin = fig_margin or {'top': 0, 'bottom': 30, 'left': 60, 'right': 0}

        self._figure = Figure(scales=scales, axes=[x_axis, y_axis], fig_margin=fig_margin)

        self._figure.layout.height = f'{50 + (10 * 1)}px'
        self._figure.layout.width = f'{width}px'

        linear_y_scale = LinearScale(allow_padding=False)
        self._index_indicator = Lines(x=[0, 0], y=[0, 1],
                                      scales={'x': x_scale, 'y': linear_y_scale},
                                      colors=['lime'])

        self._next_frame_button = widgets.Button(description='Next frame')
        self._next_frame_button.on_click(lambda *args: self.next_frame())

        self._previous_frame_button = widgets.Button(description='Previous frame')
        self._previous_frame_button.on_click(lambda *args: self.previous_frame())

        super().__init__(children=[self._figure,
                                   widgets.HBox(
                                       [self._previous_frame_button, self._next_frame_button])
                                   ], **kwargs)

        event = Event(source=self._figure, watched_events=['mousemove', 'mousedown'])
        event.on_dom_event(self._handle_event)

        self._figure.marks = [self._index_indicator]

    @property
    def figure(self):
        return self._figure

    @property
    def marks(self):
        return self.figure.marks

    @property
    def x_axis(self):
        return self.figure.axes[0]

    @property
    def y_axis(self):
        return self.figure.axes[1]

    @property
    def x_scale(self):
        return self.x_axis.scale

    @property
    def y_scale(self):
        return self.y_axis.scale

    def _handle_event(self, event):
        if event['buttons']:
            self.frame_index = int(np.round(self.pixel_to_domain(event['offsetX'])))

    @validate('frame_index')
    def _validate_frame_index(self, change):
        value = change['value']
        value = max(value, 0)
        value = min(value, self.num_frames - 1)
        return value

    @observe('frame_index')
    def _observe_frame_index(self, change):
        self._index_indicator.x = [self.frame_index] * 2

    def pixel_to_domain(self, x):
        pixel_margin_width = self.figure.fig_margin['left'] + self.figure.fig_margin['right']
        pixel_width = int(self.figure.layout.width[:-2]) - pixel_margin_width
        domain_width = self.x_scale.max - self.x_scale.min
        x = domain_width / pixel_width * (x - self.figure.fig_margin['left']) + self.x_scale.min
        return x

    def _update_scatter(self):
        if len(self.data) == 0:
            return

        for key, values in self.data.items():
            self._summaries_scatter[key].x = np.arange(len(values), dtype=np.float64)
            self._summaries_scatter[key].y = [key] * len(values)
            self._summaries_scatter[key].color = values

    @observe('tags')
    def _observe_tags(self, change):

        for tags in change['new']:
            tags.set_scales({'x': self.x_scale, 'y': self.y_scale})

        self.y_scale.domain = [tags._row for tags in change['new']]
        self.figure.marks = [self._index_indicator] + [tags.mark for tags in change['new']]

    @observe('data')
    def _observe_data(self, change):
        self._update_scatter()

    def next_frame(self):
        self.frame_index = self.frame_index + 1

    def previous_frame(self):
        self.frame_index = self.frame_index - 1
