import ipywidgets as widgets
import numpy as np
from bqplot import ColorScale, LinearScale
from bqplot_image_gl import ImageGL
from scipy.ndimage import gaussian_filter
from traitlets import Int, observe, default, Float, Unicode, List, Bool, Dict, link
from traittypes import Array

from nanotag.layout import VBox
from nanotag.timeline import Timeline
from nanotag.utils import link
from skimage.draw import polygon2mask


class GaussianFilterSlider(VBox):
    sigma = Float(0.)

    def __init__(self, min=0, max=5, **kwargs):
        self._slider = widgets.FloatSlider(description='Gaussian filter', min=min, max=max, step=.5)
        link((self, 'sigma'), (self._slider, 'value'))

        super().__init__(children=[self._slider], **kwargs)

    def __call__(self, image):
        return gaussian_filter(image, sigma=float(self._slider.value))


class ImageSeries(VBox):
    num_frames = Int(0)
    frame_index = Int(0)
    sampling = Float(1.)
    color_scheme = Unicode('grey')
    shifts = Dict()

    filters = List()
    images = Array(check_equal=False)
    image = Array(check_equal=False)

    visible = Bool(True)
    autoadjust_scales = Bool(True)

    vmin = Float(0.)
    vmax = Float(1.)

    def __init__(self, **kwargs):

        scales = {'x': LinearScale(), 'y': LinearScale(), 'image': ColorScale(colors=['black', 'white'])}

        self._mark = ImageGL(image=np.zeros((0, 0)), scales=scales)

        link((self, 'image'), (self._mark, 'image'), check_broken=False)

        color_scheme_dropdown = widgets.Dropdown(
            options=['grey', 'viridis', 'plasma', 'inferno', 'magma'],
            value='grey',
            description='Color scheme:',
            style={'description_width': '144px'}
        )
        visible_checkbox = widgets.Checkbox(description='Visible')

        link((self, 'visible'), (visible_checkbox, 'value'))
        link((self, 'color_scheme'), (color_scheme_dropdown, 'value'))
        link((self, 'visible'), (self._mark, 'visible'))

        self._range_slider = widgets.FloatRangeSlider(description='Range slider',
                                                      min=0.,
                                                      max=1.,
                                                      step=.01,
                                                      value=[0, 1.])

        def range_slider_changed(change):
            self.vmin = change['new'][0]
            self.vmax = change['new'][1]

        self._range_slider.observe(range_slider_changed, 'value')

        super().__init__(children=[color_scheme_dropdown, visible_checkbox, self._range_slider], **kwargs)

    def autoadjust_range(self, mask=None):
        image_mask = np.zeros_like(self.image, dtype=bool)

        for polygon in mask:
            if len(polygon) > 0:
                image_mask += polygon2mask(self.image.shape, polygon[:, ::-1])

        image_mask = image_mask == 0
        min_value, ptp_value = self.image.min(), self.image.ptp()
        masked_min_value, masked_ptp_value = self.image[image_mask].min(), self.image[image_mask].ptp()
        self.vmin = float((masked_min_value - min_value) / ptp_value)
        self.vmax = float((masked_min_value + masked_ptp_value - min_value) / ptp_value)

    def build_timeline(self):
        timeline = Timeline()
        link((timeline, 'frame_index'), (self, 'frame_index'))
        link((self, 'num_frames'), (timeline, 'num_frames'))
        return timeline

    @property
    def mark(self):
        return self._mark

    @property
    def x_scale(self):
        return self.mark.scales['x']

    @property
    def y_scale(self):
        return self.mark.scales['y']

    @property
    def shift(self):
        try:
            return self.shifts[self.frame_index]
        except KeyError:
            return [0., 0.]

    @property
    def x_limits(self):
        return [-.5 * self.sampling + self.shift[0], (self.image.shape[0] - .5) * self.sampling + self.shift[0]]

    @property
    def y_limits(self):
        return [-.5 * self.sampling + self.shift[1], (self.image.shape[1] - .5) * self.sampling + self.shift[1]]

    def set_scales(self, scales):
        scales['image'] = self.mark.scales['image']
        self._mark.scales = scales

    @default('images')
    def _default_images(self):
        return np.zeros((0, 0, 0))

    @default('image')
    def _default_image(self):
        return np.zeros((0, 0))

    @property
    def color_scale(self):
        return self.mark.scales['image']

    def _update_image(self):
        image = self.images[min(self.frame_index, len(self.images) - 1)]
        for filt in self.filters:
            image = filt(image)
        self.image = image

    @observe('frame_index')
    def _observe_frame_index(self, change):
        self._update_image()

    @observe('color_scheme')
    def _observe_color_scale(self, change):
        if change['new'] == 'grey':
            self.color_scale.colors = ['black', 'white']
            self.color_scale.scheme = ''
        else:
            self.color_scale.colors = []
            self.color_scale.scheme = change['new']

    @observe('filters')
    def _observe_filters(self, change):

        for filt in change['new']:
            filt.observe(lambda *args: self._update_image())

    @observe('shift')
    def _observe_shift(self, *args):
        with self.mark.hold_sync():
            self.mark.x = self.x_limits
            self.mark.y = self.y_limits

    @observe('vmin', 'vmax')
    def _set_color_scale(self, *args):
        self._range_slider.value = [self.vmin, self.vmax]

        min_value = float(self.image.min())
        ptp_value = float(self.image.ptp())

        self.mark.scales['image'].min = min_value + self.vmin * ptp_value
        self.mark.scales['image'].max = min_value + self.vmax * ptp_value

    @observe('image')
    def _observe_image(self, change):
        if self.image.size == 0:
            return

        self._set_color_scale()

        if not self.autoadjust_scales:
            return

        with self.mark.hold_sync():
            self.mark.x = self.x_limits
            self.mark.y = self.y_limits

    @observe('images')
    def _observe_images(self, change):
        if len(self.images) == 0:
            return

        self.num_frames = len(change['new'])

        # if self.frame_index == 0:
        self._update_image()
        # else:
        #    self.frame_index = 0

        if not self.autoadjust_scales:
            return

        self.x_scale.min = self.x_limits[0]
        self.x_scale.max = self.x_limits[1]
        self.y_scale.min = self.y_limits[0]
        self.y_scale.max = self.y_limits[1]
