import ipywidgets as widgets
import numpy as np
from bqplot import Figure, Axis, LinearScale, Lines, OrdinalScale, Scatter, ColorScale
from ipyevents import Event
from traitlets import observe, List, Int, validate, Unicode, Instance, default, HasTraits, Float, Bool

from nanotag.utils import link, Array
from bqplot import Hist, Bars


class Histogram(widgets.HBox):

    adjust_x_scale = Bool(True)
    min = Float(0.)
    max = Float(1.)
    bins = Int(10)
    sample = Array(np.zeros((0,)), check_equal=False)

    def __init__(self, markers=None, fig_margin=None, label=None, width=450, height=450, **kwargs):
        x_scale = LinearScale(allow_padding=False)
        y_scale = LinearScale(allow_padding=False)

        scales = {'x': x_scale, 'y': y_scale}

        x_axis = Axis(scale=scales['x'])
        y_axis = Axis(scale=scales['y'], orientation='vertical')

        fig_margin = fig_margin or {'top': 0, 'bottom': 30, 'left': 30, 'right': 0}

        min_aspect_ratio = width / height
        max_aspect_ratio = width / height

        self._figure = Figure(scales=scales,
                              axes=[x_axis, y_axis],
                              min_aspect_ratio=min_aspect_ratio,
                              max_aspect_ratio=max_aspect_ratio,
                              fig_margin=fig_margin)

        self._figure.layout.height = f'{height}px'
        self._figure.layout.width = f'{width}px'

        self._mark = Bars(x=np.zeros((0,)), y=np.zeros((0,)), scales={'x': x_scale, 'y': y_scale})

        if label is not None:
            self._mark.display_legend = True
            self._mark.labels = [label]
            self._figure.legend_style = {'stroke-width': 0}
            self._figure.legend_location = 'top-left'

        super().__init__(children=[self._figure], **kwargs)

        if markers is not None:
            self.add_traits(**{marker: Float(0.) for marker in markers})
            self.observe(lambda *args: self._observe_markers(), tuple(markers))

        self._markers = markers

        event = Event(source=self._figure, watched_events=['mousemove', 'mousedown'])
        event.on_dom_event(self._handle_event)

        linear_y_scale = LinearScale(allow_padding=False)
        self._index_indicator = Lines(x=[[0, 0]] * len(markers), y=[[0, 1]] * len(markers),
                                      scales={'x': x_scale, 'y': linear_y_scale},
                                      colors=['lime'])

        self._figure.marks = [self._mark, self._index_indicator]

        self._observe_min_max_bins(None)
        self._current_marker = None

        link((self, 'min'), (x_scale, 'min'))
        link((self, 'max'), (x_scale, 'max'))

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

    @observe('min', 'max', 'bins')
    def _observe_min_max_bins(self, change):
        self._mark.x = np.linspace(self.min, self.max, self.bins)

        if len(self.sample) > 0:
            h = np.histogram(self.sample, bins=self.bins, range=(self.min, self.max))
            self._mark.y = h[0]

    @observe('sample')
    def _observe_sample(self, change):
        if len(change['new']) == 0:
            return

        with self.hold_trait_notifications():
            values = change['new']

            if self.adjust_x_scale:
                self.min = float(min(values))
                self.max = float(max(values))

            h = np.histogram(values, bins=self.bins, range=(self.min, self.max))

            self._mark.y = h[0]

    def _handle_event(self, event):

        if event['buttons']:
            x = self.pixel_to_domain(event['offsetX'])

            if event['event'] == 'mousedown':
                i = np.argmin([np.abs(getattr(self, marker) - x) for marker in self._markers])
                self._current_marker = self._markers[i]

            if self._current_marker is not None:
                setattr(self, self._current_marker, x)

    def pixel_to_domain(self, x):
        pixel_margin_width = self.figure.fig_margin['left'] + self.figure.fig_margin['right']
        pixel_width = int(self.figure.layout.width[:-2]) - pixel_margin_width
        domain_width = self.x_scale.max - self.x_scale.min
        x = domain_width / pixel_width * (x - self.figure.fig_margin['left']) + self.x_scale.min
        return x

    def _observe_markers(self):
        x = []
        for marker in self._markers:
            x += [[getattr(self, marker)] * 2]
        self._index_indicator.x = x



