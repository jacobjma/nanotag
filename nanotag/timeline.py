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
    data_overlay = Unicode()
    color_scale = Instance(ColorScale)

    def __init__(self, row, data_fields, enable_move=True, **kwargs):
        self._mark = Scatter(x=np.zeros((0,)), y=np.zeros((0,)), marker='rectangle')
        self._mark.enable_move = enable_move
        self._mark.restrict_x = True
        self._row = row

        # if default_t is None:
        #    default_t = np.zeros((0,))

        # self._default_t = default_t

        self._data_fields = data_fields

        super().__init__(**kwargs)

        if data_fields is not None:
            self.add_traits(**{data_field: Array(np.zeros((0,))) for data_field in data_fields})
            self.observe(lambda *args: self._set_color(), tuple(data_fields) + ('data_overlay',))
            self.data_overlay = data_fields[0]

        # link((self, 'labels'), (self._mark, 'color'), check_broken=False)
        link((self, 't'), (self._mark, 'x'), check_broken=False)

    @property
    def empty(self):
        return len(self.t) == 0

    @default('color_scale')
    def _default_color_scale(self):
        return ColorScale()

    @property
    def mark(self):
        return self._mark

    def set_scales(self, scales):
        scales.update({'color': self.color_scale})
        self._mark.scales = scales

    @observe('t')
    def _observe_t(self, change):
        self._mark.y = [self._row] * len(change['new'])

    def _set_color(self):
        self._mark.color = getattr(self, self.data_overlay)

    def reset(self):
        self.t = []
        for data_field in self._data_fields:
            setattr(self, data_field, np.zeros((0,)))

    def add_tags(self, t, labels):
        self.t = np.concatenate((self.t, t))
        self.labels = np.concatenate((self.labels, labels))

    def delete_tags(self, indices):
        self.t = np.delete(self.t, indices)
        self.labels = np.delete(self.labels, indices)

    def serialize(self):
        serialized = {}

        for data_field in self._data_fields:
            data = getattr(self, data_field)
            serialized = {i: {} for i in range(len(data))}
            for i, value in enumerate(data):
                serialized[i][data_field] = value

        return serialized
        # return {'t': self.t.tolist()}

    def from_serialized(self, serialized):
        for data_field in self._data_fields:
            data = [serialized[frame_index][data_field] for frame_index in serialized.keys()]
            setattr(self, data_field, data)

        self.t = np.array([int(key) for key in serialized.keys()])

        # self.t = serialized['t']
        # self.labels = serialized['labels']


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


        fig_margin = fig_margin or {'top': 0, 'bottom': 30, 'left': 5, 'right': 0}

        self._figure = Figure(scales=scales, axes=[x_axis, y_axis], fig_margin=fig_margin)

        self._figure.layout.height = f'{50 + (20 * 1)}px'
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
        self.figure.marks = [tags.mark for tags in change['new']] + [self._index_indicator]

    @observe('data')
    def _observe_data(self, change):
        self._update_scatter()

    def next_frame(self):
        self.frame_index = self.frame_index + 1

    def previous_frame(self):
        self.frame_index = self.frame_index - 1


class TimelineTagEditor(widgets.HBox):

    def __init__(self, tags, timeline):

        def set_last_frame(*args):
            if len(tags.t) > 0:
                start = tags.t[0]
            else:
                start = 0

            tags.t = list(range(start, timeline.frame_index + 1))

        def set_first_frame(*args):
            if len(tags.t) > 0:
                end = tags.t[-1] + 1
            else:
                end = timeline.frame_index + 1
            tags.t = list(range(timeline.frame_index, end))

        def toggle_frame(*args):
            if timeline.frame_index in list(tags.t):
                tags.t = np.delete(tags.t, np.where(tags.t == timeline.frame_index)[0])

            else:
                tags.t = np.sort(np.concatenate((tags.t, [timeline.frame_index]))).astype(int)

        def toggle_all(*args):
            if len(tags.t) > 0:
                tags.t = []
            else:
                tags.t = range(0, timeline.num_frames)

        layout = widgets.Layout(width='100px')

        last_frame_button = widgets.Button(description='Last frame', layout=layout)
        last_frame_button.on_click(set_last_frame)
        first_frame_button = widgets.Button(description='First frame', layout=layout)
        first_frame_button.on_click(set_first_frame)
        toggle_frame_button = widgets.Button(description='Toggle frame', layout=layout)
        toggle_frame_button.on_click(toggle_frame)
        toggle_all_button = widgets.Button(description='Toggle all', layout=layout)
        toggle_all_button.on_click(toggle_all)

        super().__init__(children=[first_frame_button, toggle_frame_button, last_frame_button, toggle_all_button])
