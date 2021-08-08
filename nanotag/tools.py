from abc import abstractmethod, ABC

from bqplot.interacts import BrushSelector, PanZoom
from ipyevents import Event
from scipy.spatial import KDTree
import numpy as np


class Tool(ABC):

    @abstractmethod
    def activate(self, canvas):
        pass

    @abstractmethod
    def deactivate(self, canvas):
        pass


class Action(ABC):

    @abstractmethod
    def activate(self, canvas):
        pass


class ResetView(Action):

    def activate(self, canvas):
        canvas.reset()


class BoxZoomTool(Tool):

    def activate(self, canvas):
        brush_selector = BrushSelector()
        brush_selector.x_scale = canvas.x_scale
        brush_selector.y_scale = canvas.y_scale

        def box_zoom(change):
            selected_x = brush_selector.selected_x
            selected_y = brush_selector.selected_y

            if (selected_x is None) or (selected_y is None):
                return

            canvas.x_scale.min = min(selected_x[0], selected_x[1])
            canvas.x_scale.max = max(selected_x[0], selected_x[1])
            canvas.y_scale.min = min(selected_y[0], selected_y[1])
            canvas.y_scale.max = max(selected_y[0], selected_y[1])

            canvas.adjust_equal_axes()

            brush_selector.selected_x = None
            brush_selector.selected_y = None
            canvas.figure.interaction = None
            canvas.figure.interaction = brush_selector

        brush_selector.observe(box_zoom, 'brushing')
        canvas.figure.interaction = brush_selector

    def deactivate(self, canvas):
        canvas.figure.interaction = None


class PanZoomTool(Tool):

    def activate(self, canvas):
        panzoom = PanZoom(scales={'x': [canvas.x_scale], 'y': [canvas.y_scale]})
        canvas.figure.interaction = panzoom

    def deactivate(self, canvas):
        canvas.figure.interaction = None


class EditPointTags(Tool):

    def __init__(self, tags, data_fields=None):
        self._tags = tags

        if data_fields is None:
            data_fields = {}

        self._data_fields = data_fields
        self._event = Event(watched_events=['click'])

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    def activate(self, canvas):
        self._event.source = canvas

        def handle_event(event):
            x, y = canvas.pixel_to_domain(event['offsetX'], event['offsetY'])
            tags = self._tags

            if event['altKey'] is True:
                positions = np.array([tags.x, tags.y]).T
                distances, indices = KDTree(positions).query([[x, y]])
                tags.delete_tags(indices)
            else:
                tags.add_tags([x], [y], {key: [value] for key, value in self._data_fields.items()})

        self._event.on_dom_event(handle_event)

    def deactivate(self, canvas):
        self._event.reset_callbacks()
