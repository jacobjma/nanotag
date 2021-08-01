import ipywidgets as widgets


class VBox(widgets.VBox):

    def __init__(self, children, **kwargs):
        super().__init__(children, **kwargs)

        self.layout.border = 'solid 1px'
