import ipywidgets as widgets
from scipy.ndimage import gaussian_filter
from traitlets import observe
from traittypes import Array


class GaussianFilterSlider(widgets.HBox):
    image_in = Array(check_equal=False)
    image_out = Array(check_equal=False)

    def __init__(self, min=0, max=5, **kwargs):
        self._slider = widgets.FloatSlider(description='Gaussian filter', min=min, max=max)

        super().__init__(children=[self._slider], **kwargs)
        self._slider.observe(self.update_image)

        self.layout.border = 'solid 1px'

    def update_image(self, *args):
        self.image_out = self(self.image_in)

    @observe('image_in')
    def _observe_image_in(self, *args):
        self.update_image()

    def __call__(self, image):
        return gaussian_filter(image, sigma=float(self._slider.value))
