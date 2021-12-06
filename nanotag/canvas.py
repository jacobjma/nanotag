import contextlib

import ipywidgets as widgets
from bqplot import Figure, Axis, LinearScale
from traitlets import observe, Float, Unicode, Dict, List

from nanotag.utils import link


class ToolBox(widgets.HBox):
    tools = Dict()
    button_width = Unicode()

    def __init__(self, canvas, **kwargs):
        self._canvas = canvas
        self._widgets = None
        self._active_tool = None
        super().__init__(children=[], **kwargs)
        self.layout.border = 'solid 1px'

    @property
    def canvas(self):
        return self._canvas

    def activate_action(self, action):
        self.tools[action].activate(self.canvas)

    def toggle_tool(self, tool):
        self._widgets[tool].value = not self._widgets[tool].value

    @observe('tools')
    def _observe_tools(self, change):

        def toggle_tool(change):
            if self._active_tool is not None:
                if self._active_tool != change['owner'].description:
                    self._widgets[self._active_tool].value = False

                self.tools[self._active_tool].deactivate(self.canvas)

            if change['owner'].value is True:
                self.tools[change['owner'].description].activate(self.canvas)

            self._active_tool = change['owner'].description

        def activate_action(owner):
            self.activate_action(owner.description)

        self._widgets = {}
        for key, value in change['new'].items():
            if hasattr(value, 'deactivate'):
                widget = widgets.ToggleButton(value=False, description=key)
                widget.observe(toggle_tool, 'value')

            elif hasattr(value, 'activate'):
                widget = widgets.Button(description=key)
                widget.on_click(activate_action)

            else:
                raise RuntimeError()

            self._widgets[key] = widget

        with self.hold_trait_notifications():
            self.children = list(self._widgets.values())
            self._adjust_button_width()

    @observe('button_width')
    def _observe_button_width(self, change):
        self._adjust_button_width()

    def _adjust_button_width(self):
        for child in self.children:
            child.layout.width = self.button_width


class Canvas(widgets.HBox):
    images = List()
    sampling = Float(1.)
    tags = List()
    tools = Dict()
    tool = Unicode(None, allow_none=True)

    def __init__(self, fig_margin=None, width=450, height=450, **kwargs):
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

        super().__init__(children=[self._figure], **kwargs)

        self._image_mark = None
        self._tag_marks = None

        self._toolbox = ToolBox(self)
        link((self, 'tools'), (self._toolbox, 'tools'))

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

    @property
    def image_mark(self):
        return self._image_mark

    @property
    def toolbox(self):
        return self._toolbox

    def pixel_to_domain(self, x, y):
        pixel_margin_width = self.figure.fig_margin['left'] + self.figure.fig_margin['right']
        pixel_margin_height = self.figure.fig_margin['top'] + self.figure.fig_margin['bottom']

        pixel_width = int(self.figure.layout.width[:-2]) - pixel_margin_width
        pixel_height = int(self.figure.layout.height[:-2]) - pixel_margin_height

        domain_width = self.x_scale.max - self.x_scale.min
        domain_height = self.y_scale.max - self.y_scale.min

        x = domain_width / pixel_width * (x - self.figure.fig_margin['left']) + self.x_scale.min
        y = domain_height / pixel_height * (y - self.figure.fig_margin['top']) + self.y_scale.min
        y = - y + self.y_scale.max + self.y_scale.min
        return x, y

    def adjust_equal_axes(self):
        if None in (self.x_scale.min, self.x_scale.min, self.y_scale.min, self.y_scale.max):
            return

        extent = max(self.x_scale.max - self.x_scale.min, self.y_scale.max - self.y_scale.min) / 2
        x_center = (self.x_scale.min + self.x_scale.max) / 2
        y_center = (self.y_scale.min + self.y_scale.max) / 2

        with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
            self.x_scale.min = x_center - extent
            self.x_scale.max = x_center + extent
            self.y_scale.min = y_center - extent
            self.y_scale.max = y_center + extent

    def reset(self):
        if self.images is None:
            return

        with contextlib.ExitStack() as stack:

            for mark in [image.mark for image in self.images] + [self.x_scale, self.y_scale]:
                stack.enter_context(mark.hold_sync())

            x_limits = self.images[0].x_limits
            y_limits = self.images[0].y_limits

            self.x_scale.min = x_limits[0]
            self.x_scale.max = x_limits[1]
            self.y_scale.min = y_limits[0]
            self.y_scale.max = y_limits[1]

    @observe('images')
    def _observe_image(self, change):
        self._image_marks = []
        for image in change['new']:
            image.set_scales({'x': self.x_scale, 'y': self.y_scale})
            self._image_marks.append(image.mark)

        marks = self._image_marks
        if self._tag_marks is not None:
            marks = marks + self._tag_marks

        self.figure.marks = marks

    @observe('tags')
    def _observe_tags(self, change):
        for tags in change['new']:
            tags.set_scales({'x': self.x_scale, 'y': self.y_scale})

        self._tags_marks = []
        for tags in change['new']:
            self._tags_marks += tags.marks

        if self._image_marks is not None:
            marks = self._image_marks + self._tags_marks
        else:
            marks = self._tags_marks

        self.figure.marks = marks

    @observe('tool')
    def _observe_tool(self, change):
        if change['new'] is not None:
            self.tools[change['new']].deactivate(self)

        if change['new'] is None:
            self.figure.interaction = None
        else:
            self.tools[change['new']].activate(self)
