import ipywidgets as widgets
from scipy.ndimage import gaussian_filter
from traitlets import observe
from traittypes import Array


class GaussianFilterSlider(widgets.FloatSlider):
    image_in = Array(check_equal=False)
    image_out = Array(check_equal=False)

    def __init__(self, description='Gaussian filter', min=0., max=5, **kwargs):
        super().__init__(description=description, min=min, max=max, **kwargs)
        self.observe(self.update_image)

    def update_image(self, *args):
        self.image_out = self(self.image_in)

    @observe('image_in')
    def _observe_image_in(self, *args):
        self.update_image()

    def __call__(self, image):
        return gaussian_filter(image, sigma=float(self.value))
