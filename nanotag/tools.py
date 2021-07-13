from abc import abstractmethod, ABC

from bqplot.interacts import BrushSelector, PanZoom


class Tool(ABC):

    @abstractmethod
    def activate(self, canvas):
        pass


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

            canvas._enforce_scale_lock()

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
        panzoom = PanZoom()
        panzoom.scales = {'x': [canvas.x_scale], 'y': [canvas.y_scale]}
        canvas.figure.interaction = panzoom

    def deactivate(self, canvas):
        canvas.figure.interaction = None
