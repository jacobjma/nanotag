from abc import abstractmethod, ABC

import ipywidgets as widgets
import numpy as np
from bqplot import ScatterGL
from bqplot.interacts import BrushSelector, PanZoom
from ipyevents import Event
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from traitlets import Int, Float, link, observe


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


class SelectPointTags(widgets.HBox):
    selected = Int(None, allow_none=True)

    def __init__(self, tags):
        self._tags = tags
        self._selected_for_move = None
        self._event = None

        super().__init__()

        self._mark = None

    @observe('selected')
    def _new_selected(self, *args):
        if self._mark is None:
            return

        self._mark.x = [self._tags.x[self.selected]]
        self._mark.y = [self._tags.y[self.selected]]

    def activate(self, canvas):
        self._event = Event(source=canvas, watched_events=['click', 'mousemove', 'mouseup', 'mousedown'])
        self._selected_for_move = None
        self._selecting = False

        self._mark = ScatterGL(x=np.zeros((0,)), y=np.zeros((0,)),
                               scales={'x': canvas.x_scale, 'y': canvas.y_scale},
                               colors=['white'], default_size=140)
        canvas.figure.marks = canvas._image_marks + [self._mark] + canvas._tags_marks

        def get_closest(mouse_x, mouse_y):

            positions = np.array([self._tags.x, self._tags.y]).T
            distances, indices = KDTree(positions).query([[mouse_x, mouse_y]])

            return int(indices[0])

        def handle_select(event):
            mouse_x, mouse_y = canvas.pixel_to_domain(event['offsetX'], event['offsetY'])

            if event['type'] == 'mousedown':
                self._selecting = True

            if event['type'] in ('mousedown', 'mousemove'):
                if self._selecting:
                    self.selected = get_closest(mouse_x, mouse_y)

            if event['type'] == 'mouseup':
                self._selecting = False

        self._event.on_dom_event(handle_select)

    def deactivate(self, canvas):
        self._selected = None
        marks = [mark for mark in canvas.figure.marks if mark is not self._mark]
        canvas.figure.marks = marks

        if self._event is not None:
            self._event.close()
            self._event = None


class EditPointTags(widgets.HBox):

    def __init__(self, tags, data_overlays=None):
        self._tags = tags
        self._selected_for_move = None
        self._event = None
        self._data_overlays = data_overlays

        super().__init__()

    def activate(self, canvas):
        self._event = Event(source=canvas, watched_events=['click', 'mousemove', 'mouseup', 'mousedown'])
        self._selected_for_move = None

        def get_closest(mouse_x, mouse_y):

            positions = np.array([self._tags.x, self._tags.y]).T
            distances, indices = KDTree(positions).query([[mouse_x, mouse_y]])

            return int(indices[0])

        def handle_add_delete(event):
            mouse_x, mouse_y = canvas.pixel_to_domain(event['offsetX'], event['offsetY'])

            if event['type'] == 'mousedown':
                self._selected_for_move = get_closest(mouse_x, mouse_y)

            if (event['type'] == 'mousemove') and (self._selected_for_move is not None):
                x = self._tags.x.copy()
                y = self._tags.y.copy()
                x[self._selected_for_move] = mouse_x
                y[self._selected_for_move] = mouse_y
                self._tags.x = x
                self._tags.y = y

            if event['type'] == 'click':
                self._selected_for_move = None

                if event['ctrlKey'] is True:
                    self._tags.add_tags([mouse_x], [mouse_y], self._data_overlays)
                if event['altKey'] is True:
                    i = get_closest(mouse_x, mouse_y)
                    self._tags.delete_tags([i])
                else:
                    pass

            if event['type'] == 'mouseup':
                self._selected_for_move = None

        self._event.on_dom_event(handle_add_delete)

    def deactivate(self, canvas):

        if self._event is not None:
            self._event.close()
            self._event = None


class BrushContourTags(widgets.VBox):
    radius = Float(20)
    n = Int(6)

    def __init__(self, tags):
        self._tags = tags
        self._event = None
        self._brushing = False

        brush_size_slider = widgets.FloatSlider(description='Brush size', min=.1, max=30, value=10)

        link((self, 'radius'), (brush_size_slider, 'value'))

        box = [brush_size_slider]
        super().__init__(children=box)
        self.layout.border = 'solid 1px'

    def activate(self, canvas):
        self._event = Event(source=canvas, watched_events=['click', 'mousemove', 'mouseup', 'mousedown'], wait=20)
        self._brushing = False

        def handle_event(event):

            def new_polygon(center):
                s = np.linspace(0, np.pi * 2, self.n, endpoint=False)
                circle_x = np.cos(s) * self.radius + center[0]
                circle_y = np.sin(s) * self.radius + center[1]
                return Polygon([(x, y) for x, y in zip(circle_x, circle_y)])

            def set_tags(polygons):
                if hasattr(polygons, 'geoms'):
                    polygons = polygons.geoms
                else:
                    polygons = [polygons]

                points = [np.array(polygon.exterior.coords) for polygon in polygons]
                self._tags.points = points

            def add_polygon(center):
                polygons = [Polygon(polygon) for polygon in self._tags.points]
                polygons += [new_polygon(center)]
                polygons = unary_union(polygons)
                set_tags(polygons)

            def remove_polygon(center):
                polygons = [Polygon(polygon) for polygon in self._tags.points]

                polygons = unary_union(polygons)
                polygons = polygons.difference(new_polygon(center))
                set_tags(polygons)

            if event['type'] == 'mousedown':
                self._brushing = True
                mouse_x, mouse_y = canvas.pixel_to_domain(event['offsetX'], event['offsetY'])

                if event['altKey'] is True:
                    remove_polygon((mouse_x, mouse_y))
                else:
                    add_polygon((mouse_x, mouse_y))

            elif event['type'] == 'mousemove':
                if self._brushing:
                    mouse_x, mouse_y = canvas.pixel_to_domain(event['offsetX'], event['offsetY'])

                    if event['altKey'] is True:
                        remove_polygon((mouse_x, mouse_y))
                    else:
                        add_polygon((mouse_x, mouse_y))


            elif event['type'] == 'mouseup':
                self._brushing = False

        self._event.on_dom_event(handle_event)

    def deactivate(self, canvas):
        if self._event is not None:
            self._event.close()
            self._event = None
