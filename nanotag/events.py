from traitlets import HasTraits, Dict, observe
from ipyevents import Event


class KeyEvents:

    def __init__(self, source   ):
        self._event = Event(source=source, watched_events=['keydown'])
        self._event.on_dom_event(self._handle_event)
        self._callbacks = None

    def _handle_event(self, event):
        if self.callbacks is None:
            return

        for key, callback in self._callbacks.items():
            if event['key'] == key:
                callback()

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value):
        self._callbacks = value

    def reset_callbacks(self):
        self._event.reset_callbacks()


class ClickEvents:

    def __init__(self, source):
        self._event = Event(source=source, watched_events=['click'])
        self._event.on_dom_event(self._handle_event)
        self._callback = None

    def _handle_event(self, event):
        if self.callback is None:
            return
        self.callback((event['offsetX'], event['offsetY']))

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    def reset_callbacks(self):
        self._event.reset_callbacks()


# h = widgets.HTML('Event info')
# def handle_event(event):
#     lines = ['{}: {}'.format(k, v) for k, v in event.items()]
#     content = '<br>'.join(lines)
#     h.value = content
#
# event = Event(source=timeline, watched_events=['mousemove'])
# event.on_dom_event(handle_event)
# h