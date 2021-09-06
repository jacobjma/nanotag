import glob
import json
import os

import ipywidgets as widgets
import numpy as np
from skimage.io import imread
from traitlets import List, Unicode, Int, observe, Dict, default, directional_link, Bool, validate, Callable
from traittypes import Array

from nanotag.layout import VBox
from nanotag.utils import link, md5_digest


def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files


class IntSliderWithButtons(widgets.HBox):
    value = Int(0)
    min = Int(0)
    max = Int(0)

    def __init__(self, **kwargs):
        self._slider = widgets.IntSlider(**kwargs)

        self._slider.layout.width = '300px'

        link((self._slider, 'value'), (self, 'value'))
        link((self._slider, 'min'), (self, 'min'))
        link((self._slider, 'max'), (self, 'max'))

        previous_button = widgets.Button(description='Previous')
        previous_button.layout.width = '100px'
        next_button = widgets.Button(description='Next')
        next_button.layout.width = '100px'

        next_button.on_click(lambda args: self.next_item())
        previous_button.on_click(lambda args: self.previous_item())

        super().__init__([self._slider, previous_button, next_button])

    def next_item(self):
        self._slider.value += 1

    def previous_item(self):
        self._slider.value -= 1


class Text(widgets.Text):

    def __init__(self, *args, **kwargs):
        super().__init__(style={'description_width': '144px'}, *args, **kwargs)
        self.layout.width = '452px'


class NanotagData(VBox):
    root_directory = Unicode()
    analysis_folder = Unicode()
    read_file = Unicode()
    write_file = Unicode()

    analysis_path = Unicode()

    identifier = Unicode(allow_none=True)
    data = Dict()

    def __init__(self, tags, **kwargs):
        self._root_directory_text = Text(description='Root directory')

        self._analysis_folder_text = widgets.Text()
        self._analysis_folder_button = widgets.Button(description='Analysis')

        self._read_file_text = widgets.Text()
        self._read_file_button = widgets.Button(description='Read data')

        self._write_file_text = widgets.Text()
        self._write_file_button = widgets.Button(description='Write data')

        link((self, 'root_directory'), (self._root_directory_text, 'value'))
        link((self, 'analysis_folder'), (self._analysis_folder_text, 'value'))
        link((self, 'write_file'), (self._write_file_text, 'value'))
        link((self, 'read_file'), (self._read_file_text, 'value'))

        self._read_file_button.on_click(lambda *args: self.read_data())
        self._write_file_button.on_click(lambda *args: self.write_data())

        self._tags = tags

        super().__init__(children=[self._root_directory_text,
                                   self._analysis_folder_text,
                                   widgets.HBox([self._read_file_button, self._read_file_text]),
                                   widgets.HBox([self._write_file_button, self._write_file_text]),
                                   ], **kwargs)

    def retrieve_tags(self, identifier):

        for key, tags in self._tags.items():
            if tags.empty:
                continue

            if not identifier in self.data.keys():
                self.data[identifier] = {}

            self.data[identifier][key] = tags.serialize()

    def send_tags(self):
        for tags in self._tags.values():
           tags.reset()

        #if identifier in self.data.keys():

        d = self.data[list(self.data.keys())[0]]

        for key, data in d.items():
            if not key in self._tags.keys():
                continue

            self._tags[key].from_serialized(data)

    @observe('root_directory', 'analysis_folder', 'read_file')
    def _new_path(self, *args):
        self.analysis_path = os.path.join(self.root_directory, self.analysis_folder, self.read_file)

    @observe('analysis_path')
    def _observe_identifier(self, change):
        self.read_data()

        #self.retrieve_tags(change['old'])
        #self.send_tags(change['new'])
        #print(self.analysis_path)
        #try:
        #    self.selected_data = self.data[change['new']]
        #except KeyError:
        #    pass

    def write_data(self):
        self.retrieve_tags(self.identifier)
        with open(os.path.join(self.root_directory, self._write_file_text.value), 'w') as f:
            json.dump(self.data, f)

    def read_data(self):
        try:
            with open(self.analysis_path, 'r') as f:
                self.data = json.load(f)

            self.send_tags()
        except FileNotFoundError:
            self.data = {}

        # try:
        #     self.selected_data = self._data[self.identifier]
        # except KeyError:
        #     pass


class ImageFileCollection(VBox):
    root_directory = Unicode()
    filter = Unicode()

    file_index = Int(0)
    paths = List()
    path = Unicode()
    filename = Unicode()
    relative_path = Unicode()

    # hashes = List()
    # hash = Unicode(allow_none=True)

    images = Array(check_equal=False)
    num_frames = Int(0)

    def __init__(self, image_series=None, **kwargs):
        self._filters_text = Text(description='Filter')
        self._file_select = widgets.Dropdown(options=[''], layout=widgets.Layout(width='300px'), value='')
        self._find_button = widgets.Button(description='Find files')
        self._previous_button = widgets.Button(description='Previous file')
        self._next_button = widgets.Button(description='Next file')

        super().__init__(children=
                         [widgets.HBox([self._filters_text]),
                          widgets.HBox([self._find_button, self._file_select]),
                          widgets.HBox([self._previous_button, self._next_button])
                          ], **kwargs)

        link((self, 'filter'), (self._filters_text, 'value'))
        link((self, 'paths'), (self._file_select, 'options'))
        link((self, 'path'), (self._file_select, 'value'))

        self._find_button.on_click(lambda x: self.load_paths())
        self._previous_button.on_click(lambda *args: self.previous_file())
        self._next_button.on_click(lambda *args: self.next_file())

        self._image_series = image_series

    def load_paths(self):
        self.paths = glob.glob(os.path.join(self.root_directory, self._filters_text.value), recursive=True)[:15]
        # self.hashes = [md5_digest(path) for path in self.paths]

    @default('images')
    def _default_images(self):
        return np.zeros((0, 0, 0))

    @default('paths')
    def _default_paths(self, *args):
        return ['']

    @observe('root_directory')
    def _observe_filter(self, *args):
        self.load_paths()

    @validate('path')
    def _validate_path(self, proposal):

        if proposal['value'] is None:
            return ''

        return proposal['value']

    @observe('path')
    def _observe_path(self, *args):
        if self.path is None:
            return

        try:
            images = imread(self.path)
        except ValueError:
            images = np.zeros((0, 0, 0))

        if len(images.shape) == 2:
            images = images[None]

        assert len(images.shape) == 3

        if (images.shape[-1] == 4) or (images.shape[-1] == 3):
            images = np.rollaxis(images, -1)

        self.images = images
        self.num_frames = len(images)

        self.relative_path = os.path.relpath(self.path, self.root_directory)
        # self.filename = os.path.split()
        # self.hash = md5_digest(self.path)

        if self._image_series is not None:
            self._image_series.images = images

    def _current_file_index(self):
        try:
            return self._file_select.options.index(self.path)
        except ValueError:
            return None

    @property
    def num_files(self):
        return len(self._file_select.options)

    def next_file(self):
        i = self._current_file_index()
        if i is None:
            return

        self._file_select.value = self._file_select.options[(i + 1) % self.num_files]

    def previous_file(self):
        i = self._current_file_index()
        if i is None:
            return

        self._file_select.value = self._file_select.options[(i - 1) % self.num_files]


class Summary(VBox):
    key = Unicode()
    update_func = Callable()

    current_data = Dict()
    data = Dict()

    current_path = Unicode()
    path = Unicode()

    current_write_file = Unicode(allow_none=True)
    write_file = Unicode(allow_none=True)

    def __init__(self, **kwargs):
        write_current_button = widgets.Button(description='Write current')
        write_button = widgets.Button(description='Write')

        current_file_text = widgets.Text()
        file_text = widgets.Text()

        self._summary_text = widgets.Textarea(layout=widgets.Layout(width='452px', height='500px'))

        # update_button.on_click(lambda *args: self._update())
        write_current_button.on_click(lambda *args: self._write_current_data())
        write_button.on_click(lambda *args: self._write_data())

        super().__init__(children=[self._summary_text,
                                   widgets.VBox([
                                       widgets.HBox([write_current_button, current_file_text]),
                                       widgets.HBox([write_button, file_text])
                                   ])],
                         **kwargs)

        def summary_transform(summary):
            return json.dumps(summary, indent=4)

        directional_link((self, 'current_data'), (self._summary_text, 'value'), transform=summary_transform)

        link((self, 'write_file'), (file_text, 'value'))
        link((self, 'current_write_file'), (current_file_text, 'value'))

    def update(self):
        self.current_data, self.data = self.update_func()

    def _write_current_data(self):
        with open(os.path.join(self.current_path, self.current_write_file), 'w') as f:
            json.dump(self.current_data, f)

    def _write_data(self):
        with open(os.path.join(self.path, self.write_file), 'w') as f:
            json.dump(self.data, f)
