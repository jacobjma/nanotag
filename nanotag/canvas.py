import ipywidgets as widgets
import numpy as np
from bqplot import Figure, Axis, LinearScale, ColorScale, Lines, OrdinalScale, Scatter
from bqplot_image_gl import ImageGL
from ipyevents import Event
from traitlets import observe, Float, Unicode, Dict, List, Int, validate, Callable

from nanotag.tools import Tool, Action
from nanotag.utils import Array, link


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
            if isinstance(value, Tool):
                widget = widgets.ToggleButton(value=False, description=key)
                widget.observe(toggle_tool, 'value')

            elif isinstance(value, Action):
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
    image = Array(check_equal=False)
    sampling = Float(1.)
    tags = Dict()
    tools = Dict()
    tool = Unicode(None, allow_none=True)

    def __init__(self, fig_margin=None, width=450, height=450, **kwargs):
        x_scale = LinearScale(allow_padding=False)
        y_scale = LinearScale(allow_padding=False, reverse=True)

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

        image_scales = {**scales, 'image': ColorScale(colors=['black', 'white'])}

        self._image_mark = ImageGL(image=np.zeros((0, 0)), scales=image_scales)

        super().__init__(children=[self._figure], **kwargs)

        link((self._image_mark, 'image'), (self, 'image'), check_broken=False)

        self._figure.marks = [self._image_mark]

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
        with self.image_mark.hold_sync(), self.x_scale.hold_sync(), self.y_scale.hold_sync():
            x_limits = [-.5 * self.sampling, (self.image.shape[0] - .5) * self.sampling]
            y_limits = [-.5 * self.sampling, (self.image.shape[1] - .5) * self.sampling]

            self.x_scale.min = x_limits[0]
            self.x_scale.max = x_limits[1]
            self.y_scale.min = y_limits[0]
            self.y_scale.max = y_limits[1]

            self.image_mark.x = x_limits
            self.image_mark.y = y_limits

    @observe('image')
    def _observe_image(self, change):
        #self.reset()

        if (not self.image.size == 0):  # & self.autoadjust_colorscale:
            with self.image_mark.hold_sync():
                self.image_mark.scales['image'].min = float(self.image.min())
                self.image_mark.scales['image'].max = float(self.image.max())

    @observe('tags')
    def _observe_tags(self, change):
        for key, tag in change['new'].items():
            tag.artist.add_to_canvas(self)

        try:
            for key, tag in change['old'].items():
                tag.artist.remove_from_canvas(self)

        except AttributeError:
            pass

    @observe('tool')
    def _observe_tool(self, change):
        if change['new'] is not None:
            self.tools[change['new']].deactivate(self)

        if change['new'] is None:
            self.figure.interaction = None
        else:
            self.tools[change['new']].activate(self)


class Timeline(widgets.VBox):
    data = Dict()
    tags = List()

    frame_index = Int(0)
    num_frames = Int()

    update_func = Callable()

    def __init__(self, color_scales, fig_margin=None, height=70, width=450, **kwargs):
        x_scale = LinearScale(allow_padding=False, min=0)
        y_scale = OrdinalScale(allow_padding=False, domain=list(color_scales.keys()))

        link((self, 'num_frames'), (x_scale, 'max'))

        scales = {'x': x_scale, 'y': y_scale}

        x_axis = Axis(scale=scales['x'])
        y_axis = Axis(scale=scales['y'], orientation='vertical', grid_lines='none')

        fig_margin = fig_margin or {'top': 0, 'bottom': 30, 'left': 30, 'right': 0}

        self._figure = Figure(scales=scales, axes=[x_axis, y_axis], fig_margin=fig_margin)

        self._figure.layout.height = f'{50 + (10 * len(color_scales))}px'
        self._figure.layout.width = f'{width}px'

        index_indicator_y_scale = LinearScale(allow_padding=False)
        self._index_indicator = Lines(x=[0, 0], y=[0, 1],
                                      scales={'x': x_scale, 'y': index_indicator_y_scale},
                                      colors=['lime'])

        self._next_frame_button = widgets.Button(description='Next frame')
        self._next_frame_button.on_click(lambda *args: self.next_frame())

        self._previous_frame_button = widgets.Button(description='Previous frame')
        self._previous_frame_button.on_click(lambda *args: self.previous_frame())

        #self._update_button = widgets.Button(description='Update')

        super().__init__(children=[self._figure,
                                   widgets.HBox(
                                       [self._previous_frame_button, self._next_frame_button])
                                   ], **kwargs)

        event = Event(source=self._figure, watched_events=['mousemove', 'mousedown'])
        event.on_dom_event(self._handle_event)

        self._scatter = {}
        for entry, color_scale in color_scales.items():
            self._scatter[entry] = Scatter(x=[], y=[],
                                           scales={'x': self.x_scale,
                                                   'y': self.y_scale,
                                                   'color': color_scale
                                                   },
                                           )

        self._figure.marks = list(self._scatter.values()) + [self._index_indicator]

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
            self._scatter[key].x = np.arange(len(values), dtype=np.float64)
            self._scatter[key].y = [key] * len(values)
            self._scatter[key].color = values

    @observe('data')
    def _observe_data(self, change):
        self._update_scatter()

    def next_frame(self):
        self.frame_index = self.frame_index + 1

    def previous_frame(self):
        self.frame_index = self.frame_index - 1

# class Canvas(HasTraits):
#     image = Dict()
#     tools = Dict()
#     tool = Unicode('None')
#     figure = Instance(Figure)
#     title = Unicode()
#     x_label = Unicode()
#     y_label = Unicode()
#     x_limits = List()
#     y_limits = List()
#     lock_scale = Bool(True, allow_none=True)
#     _tool_artists = List()
#
#     def __init__(self,
#                  x_scale=None,
#                  y_scale=None,
#                  x_axis=None,
#                  y_axis=None,
#                  height=450.,
#                  width=450.,
#                  fig_margin=None,
#                  **kwargs):
#
#         x_scale = x_scale or LinearScale(allow_padding=False)
#         y_scale = y_scale or LinearScale(allow_padding=False)
#
#         scales = {'x': x_scale, 'y': y_scale}
#
#         x_axis = x_axis or Axis(scale=scales['x'])
#         y_axis = y_axis or Axis(scale=scales['y'], orientation='vertical')
#
#         fig_margin = fig_margin or {'top': 0, 'bottom': 50, 'left': 50, 'right': 0}
#
#         min_aspect_ratio = width / height
#         max_aspect_ratio = width / height
#
#         figure = Figure(scales=scales,
#                         axes=[x_axis, y_axis],
#                         min_aspect_ratio=min_aspect_ratio,
#                         max_aspect_ratio=max_aspect_ratio,
#                         fig_margin=fig_margin)
#
#         figure.layout.height = f'{height}px'
#         figure.layout.width = f'{width}px'
#
#         super().__init__(figure=figure, **kwargs)
#         link((self, 'x_label'), (x_axis, 'label'))
#         link((self, 'y_label'), (y_axis, 'label'))
#
#     @property
#     def widget(self):
#         whitespace = widgets.HBox([])
#         whitespace.layout.width = f'{self.figure.fig_margin["left"]}px'
#
#         title = widgets.HTML(value=f"<p style='font-size:16px;text-align:center'> {self.title} </p>")
#         title.layout.width = f'{float(self.figure.layout.width[:-2]) - self.figure.fig_margin["left"]}px'
#         return widgets.VBox([widgets.HBox([whitespace, title]), self.figure])
#
#     @property
#     def x_axis(self):
#         return self.figure.axes[0]
#
#     @property
#     def y_axis(self):
#         return self.figure.axes[1]
#
#     @property
#     def x_scale(self):
#         return self.x_axis.scale
#
#     @property
#     def y_scale(self):
#         return self.y_axis.scale
#
#     @observe('artists', '_tool_artists')
#     def _observe_artists(self, change):
#         self._update_marks()
#
#     def _update_marks(self):
#         self.figure.marks = []
#
#         for key, artist in self.artists.items():
#             artist._add_to_canvas(self)
#
#         for artist in self._tool_artists:
#             artist._add_to_canvas(self)
#
#     def _enforce_scale_lock(self, adjust_x=True, adjust_y=True):
#         if not self.lock_scale:
#             return
#
#         if None in (self.x_scale.min, self.x_scale.min, self.y_scale.min, self.y_scale.max):
#             return
#
#         extent = max(self.x_scale.max - self.x_scale.min, self.y_scale.max - self.y_scale.min) / 2
#         x_center = (self.x_scale.min + self.x_scale.max) / 2
#         y_center = (self.y_scale.min + self.y_scale.max) / 2
#
#         with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
#             if adjust_x:
#                 self.x_scale.min = x_center - extent
#                 self.x_scale.max = x_center + extent
#
#             if adjust_y:
#                 self.y_scale.min = y_center - extent
#                 self.y_scale.max = y_center + extent
#
#     @observe('tool')
#     def _observe_tool(self, change):
#         if change['old'] != 'None':
#             self.tools[change['old']].deactivate(self)
#
#         if change['new'] == 'None':
#             self.figure.interaction = None
#         else:
#             self.tools[change['new']].activate(self)
#
#     @property
#     def toolbar(self):
#         tool_names = ['None'] + list(self.tools.keys())
#
#         tool_selector = widgets.ToggleButtons(options=tool_names, value=self.tool)
#         tool_selector.style.button_width = '60px'
#
#         link((tool_selector, 'value'), (self, 'tool'))
#
#         whitespace = widgets.HBox([])
#         whitespace.layout.width = f'{self.figure.fig_margin["left"]}px'
#
#         reset_button = widgets.Button(description='Reset', layout=widgets.Layout(width='80px'))
#         reset_button.on_click(lambda _: self.adjust_limits_to_artists())
#
#         return widgets.HBox([whitespace, widgets.HBox([tool_selector]), reset_button])
#
#     def reset(self):
#         self.adjust_limits_to_artists()
#
#     @observe('x_limits')
#     def _observe_x_limits(self, change):
#         if change['new'] is None:
#             self.x_scale.min = None
#             self.x_scale.max = None
#         else:
#             self.x_scale.min = change['new'][0]
#             self.x_scale.max = change['new'][1]
#
#     @observe('y_limits')
#     def _observe_y_limits(self, change):
#         if change['new'] is None:
#             self.y_scale.min = None
#             self.y_scale.max = None
#         else:
#             self.y_scale.min = change['new'][0]
#             self.y_scale.max = change['new'][1]
#
#     @property
#     def visibility_checkboxes(self):
#         checkboxes = []
#         for key, artist in self.artists.items():
#             checkbox = widgets.Checkbox(value=True, description=key, indent=False, layout=widgets.Layout(width='90%'))
#             link((checkbox, 'value'), (artist, 'visible'))
#             checkboxes.append(checkbox)
#         return widgets.VBox(checkboxes)
#
#     def adjust_labels_to_artists(self):
#         x_labels = [artist.x_label for artist in self.artists.values() if artist.x_label is not None]
#         if len(x_labels) > 0:
#             self.x_label = x_labels[0]
#
#         y_labels = [artist.y_label for artist in self.artists.values() if artist.y_label is not None]
#         if len(y_labels) > 0:
#             self.y_label = y_labels[0]
#
#     def adjust_limits_to_artists(self, adjust_x=True, adjust_y=True, *args):
#         xmin = np.min([artist.limits[0][0] for artist in self.artists.values()])
#         xmax = np.max([artist.limits[0][1] for artist in self.artists.values()])
#         ymin = np.min([artist.limits[1][0] for artist in self.artists.values()])
#         ymax = np.max([artist.limits[1][1] for artist in self.artists.values()])
#
#         with self.x_scale.hold_trait_notifications(), self.y_scale.hold_trait_notifications():
#             if adjust_x:
#                 self.x_scale.min = float(xmin)
#                 self.x_scale.max = float(xmax)
#
#             if adjust_y:
#                 self.y_scale.min = float(ymin)
#                 self.y_scale.max = float(ymax)
#
#             self._enforce_scale_lock()
