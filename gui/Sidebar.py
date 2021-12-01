from PyQt5.QtWidgets import QWidget, QDockWidget, QVBoxLayout

from .CameraParamsWidget import CameraParamsWidget
from .DiffRenderWidget import DiffRenderWidget


class Sidebar(QDockWidget):
    def __init__(self, main_window, camera_params: list,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._main_window = main_window

        self._dea_box = CameraParamsWidget(main_window, camera_params)
        self._diff_render_box = DiffRenderWidget(main_window)

        self._init_layout()

        self.width = self.sizeHint().width()
        self.height = self.sizeHint().height()

    def _init_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self._dea_box)
        layout.addWidget(self._diff_render_box)

        multi_widget = QWidget()
        multi_widget.setLayout(layout)

        self.setFloating(False)
        self.setWidget(multi_widget)
