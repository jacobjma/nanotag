import contextlib
import hashlib
import os
import warnings

import matplotlib.cm
import matplotlib.colors
import numpy as np
from traitlets import TraitError, Undefined
from traitlets.traitlets import _validate_link
from traittypes import SciType, Empty


def md5_digest(file, buf_size=6553610):
    md5 = hashlib.md5()

    with open(file, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def redirected_path(root, path, inserted_folder):
    x = os.path.relpath(path, root)
    x = os.path.join(root, inserted_folder, x)
    return x


def label_to_index_generator(labels):
    if len(labels) == 0:
        return []

    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(0, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]


class link(object):
    updating = False

    def __init__(self, source, target, transform=None, check_broken=False):
        _validate_link(source, target)
        self.source, self.target = source, target
        self._transform, self._transform_inv = (
            transform if transform else (lambda x: x,) * 2)

        self.link()
        self.check_broken = check_broken

    def link(self):
        try:
            setattr(self.target[0], self.target[1],
                    self._transform(getattr(self.source[0], self.source[1])))

        finally:
            self.source[0].observe(self._update_target, names=self.source[1])
            self.target[0].observe(self._update_source, names=self.target[1])

    @contextlib.contextmanager
    def _busy_updating(self):
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update_target(self, change):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))

            if not self.check_broken:
                return

            if getattr(self.source[0], self.source[1]) != change.new:
                raise TraitError(
                    "Broken link {}: the source value changed while updating "
                    "the target.".format(self))

    def _update_source(self, change):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1],
                    self._transform_inv(change.new))

            if not self.check_broken:
                return

            if getattr(self.target[0], self.target[1]) != change.new:
                raise TraitError(
                    "Broken link {}: the target value changed while updating "
                    "the source.".format(self))

    def unlink(self):
        self.source[0].unobserve(self._update_target, names=self.source[1])
        self.target[0].unobserve(self._update_source, names=self.target[1])


class Array(SciType):
    """A numpy array trait type."""

    info_text = 'a numpy array'
    dtype = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super(Array, self).validate(obj, value)
        try:
            r = np.asarray(value, dtype=self.dtype)
            if isinstance(value, np.ndarray) and r is not value:
                warnings.warn(
                    'Given trait value dtype "%s" does not match required type "%s". '
                    'A coerced copy has been created.' % (
                        np.dtype(value.dtype).name,
                        np.dtype(self.dtype).name))
            value = r
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(Array, self).validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value

        if not self.check_equal:
            obj._notify_trait(self.name, old_value, new_value)
            return

        if not np.array_equal(old_value, new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, check_equal=False, **kwargs):
        self.dtype = dtype
        if default_value is Empty:
            default_value = np.array(0, dtype=self.dtype)
        elif default_value is not None and default_value is not Undefined:
            default_value = np.asarray(default_value, dtype=self.dtype)
        self.check_equal = check_equal
        super(Array, self).__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return np.copy(self.default_value)


def get_colors_from_cmap(c, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if vmin is None:
        vmin = np.nanmin(c)

    if vmax is None:
        vmax = np.nanmax(c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    c = np.array(c, dtype=float)

    valid = np.isnan(c) == 0
    colors = np.zeros((len(c), 4))
    colors[valid] = cmap(norm(c[valid]))

    return colors
