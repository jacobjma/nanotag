import glob
import json
import os

import ipywidgets as widgets
import numpy as np
from skimage.io import imread
from traitlets import List, Unicode, Int, observe, Dict, Callable, default, Instance, directional_link
from traittypes import Array

from nanotag.tags import tags_from_serialized
from nanotag.utils import link


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


class RootDir(widgets.VBox):
    root_dir = Unicode()

    def __init__(self, **kwargs):
        self._root_dir_text = widgets.Text(description='Root dir.')
        super().__init__(children=[self._root_dir_text], **kwargs)
        self.layout.border = 'solid 1px'
        link((self, 'root_dir'), (self._root_dir_text, 'value'))


class ImageFileCollection(widgets.VBox):
    root_dir = Unicode()
    filter = Unicode()

    file_index = Int(0)
    paths = List()
    path = Unicode(allow_none=True)

    frame_index = Int(0)
    relative_path = Unicode(allow_none=True)

    images = Array(check_equal=False)
    image = Array(check_equal=False)

    num_frames = Int(0)

    def __init__(self, **kwargs):

        self._filters_text = widgets.Text(description='Filter')
        self._file_select = widgets.Dropdown(options=[], layout=widgets.Layout(width='max-content'))
        self._find_button = widgets.Button(description='Find files')
        self._previous_button = widgets.Button(description='Previous file')
        self._next_button = widgets.Button(description='Next file')

        super().__init__(children=
                         [widgets.HBox([self._filters_text]),
                          widgets.HBox([self._find_button, self._file_select]),
                          widgets.HBox([self._previous_button, self._next_button])
                          ], **kwargs)

        self.layout.border = 'solid 1px'

        link((self, 'filter'), (self._filters_text, 'value'))
        link((self, 'paths'), (self._file_select, 'options'))
        link((self._file_select, 'value'), (self, 'path'))

        self._find_button.on_click(lambda x: self._load_paths())
        self._previous_button.on_click(lambda *args: self.previous_file())
        self._next_button.on_click(lambda *args: self.next_file())

    def _load_paths(self):
        self.paths = glob.glob(os.path.join(self.root_dir, self._filters_text.value))

    @default('images')
    def _default_images(self):
        return np.zeros((0, 0, 0))

    @default('image')
    def _default_image(self):
        return np.zeros((0, 0))

    @observe('path')
    def _observe_path(self, *args):
        if self.path is None:
            return

        images = imread(self.path)

        if len(images.shape) == 2:
            images = images[None]

        assert len(images.shape) == 3

        self.images = images
        self.relative_path = os.path.relpath(self.path, self.root_dir)

        if self.frame_index == 0:
            self._set_image()
        else:
            self.frame_index = 0

        self.num_frames = len(images)

    def _set_image(self):

        if self.images is None:
            return

        self.image = self.images[self.frame_index]

    @observe('frame_index')
    def _observe_current_frame_index(self, change):
        self._set_image()

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


class NanotagData(widgets.VBox):
    root_dir = Unicode()

    data = Dict()
    data_item = Dict()

    default_tags = Callable()

    identifier = Unicode()

    tags = List()

    frame_index = Int(0)
    frame_tags = Dict()

    write_file = Unicode()
    read_file = Unicode()

    frame_summaries = Dict()

    def __init__(self, **kwargs):
        self._read_file_text = widgets.Text()
        self._read_file_button = widgets.Button(description='Read')

        self._write_file_text = widgets.Text()
        self._write_file_button = widgets.Button(description='Write')

        super().__init__(children=[widgets.HBox([self._read_file_button, self._read_file_text]),
                                   widgets.HBox([self._write_file_button, self._write_file_text]),
                                   ], **kwargs)

        self.layout.border = 'solid 1px'

        link((self, 'write_file'), (self._write_file_text, 'value'))
        link((self, 'read_file'), (self._read_file_text, 'value'))

        self._write_file_button.on_click(lambda *args: self.write_data())
        self._read_file_button.on_click(lambda *args: self.read_data())

    def serialize(self):
        serialized = {}
        for identifier, data in self.data.items():
            serialized[identifier] = {}
            for key, values in data.items():

                if key == 'tags':
                    values = [{key: value.serialize() for key, value in frame_tags.items()} for frame_tags in values]

                serialized[identifier][key] = values
        return serialized

    def from_serialized(self, serialized):

        loaded_data = {}
        for identifier, data in serialized.items():
            loaded_data[identifier] = {}
            for key, values in data.items():

                if key == 'tags':
                    values = [{key: tags_from_serialized(tags) for key, tags in frame.items()} for frame in values]

                loaded_data[identifier][key] = values

        return loaded_data

    def write_data(self):
        with open(os.path.join(self.root_dir, self._write_file_text.value), 'w') as f:
            json.dump(self.serialize(), f)

    def read_data(self):
        with open(os.path.join(self.root_dir, self._read_file_text.value), 'r') as f:
            self.data = self.from_serialized(json.load(f))

    @observe('data')
    def _observe_data(self, change):
        self._set_data_item()

    def _set_frame_tags(self):
        if len(self.tags) <= self.frame_index:
            self.tags = self.tags + [self.default_tags() for _ in range(self.frame_index - len(self.tags) + 1)]
            self.data[self.identifier]['tags'] = self.tags
            # {'name': 'tags', 'old': [], 'new': [{'points': < nanotag.tags.PointTags object at
            #                                    0x14eb59460 >}], 'owner': TagSummaries(), 'type': 'change'}
            # self.notify_change({'name':'tags', 'new':self.tags, 'type':'change'})

        self.frame_tags = self.tags[self.frame_index]

    @observe('frame_index')
    def _observe_frame_index(self, change):
        self._set_frame_tags()

    def _set_data_item(self):
        if not self.identifier in self.data.keys():
            self.data[self.identifier] = {'tags': []}

        self.data_item = self.data[self.identifier]

    def _set_tags(self):
        self.tags = self.data_item['tags']

        if self.frame_index == 0:
            self._set_frame_tags()
        else:
            self.frame_index = 0

    @observe('data_item')
    def _observe_data_item(self, change):
        self._set_tags()

    @observe('identifier')
    def _observe_identifier(self, change):
        self._set_data_item()


class NanotagDataSummaries(widgets.VBox):
    data = Dict()

    frame_summary_funcs = Dict()
    summary_func = Callable()

    frame_summaries = Dict()
    summary = Dict()

    def __init__(self, **kwargs):
        self._update_button = widgets.Button(description='Update summaries')
        self._write_summary_button = widgets.Button(description='Write summary')
        self._summary_text_area = widgets.Textarea(
            description='Summary:',
            layout=widgets.Layout(width='600px')
        )

        super().__init__(children=[widgets.HBox([self._update_button, self._write_summary_button]),
                                   self._summary_text_area], **kwargs)


        def summary_transform(summary):
            return str(summary)

        directional_link((self, 'summary'), (self._summary_text_area, 'value'), transform=summary_transform)

        self._update_button.on_click(lambda *args: self.update())
        self._write_summary_button.on_click(lambda *args: self.write_summary())
        self.layout.border = 'solid 1px'

    def write_summary(self):
        pass

    def update(self):
        self._update_frame_summaries()
        self._update_summary()

    def _update_summary(self):
        self.summary = self.summary_func(self.data, self.frame_summaries)

    def _update_frame_summaries(self):
        if len(self.data) == 0:
            return

        frame_summaries = {key: [] for key in self.frame_summary_funcs.keys()}
        for key, summary_func in self.frame_summary_funcs.items():
            frame_summaries[key] = summary_func(self.data)

        self.frame_summaries = frame_summaries

    @observe('data')
    def _observe_tags(self, change):
        self._update_frame_summaries()
