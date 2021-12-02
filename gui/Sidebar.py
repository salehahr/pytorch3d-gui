from PyQt5.QtWidgets import QWidget, QDockWidget, QVBoxLayout

from .CameraParamsWidget import CameraParamsWidget
from .DiffRenderWidget import DiffRenderWidget
from .Panes import Panes


class Sidebar(QDockWidget):
    def __init__(self, graphics: QWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dea_box = CameraParamsWidget(graphics)
        self._diff_render_box = DiffRenderWidget(graphics.get_pane(Panes.RENDER))

        self._init_layout()

        self.width: int = self.sizeHint().width()
        self.height: int = self.sizeHint().height()

    def _init_layout(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._dea_box)
        layout.addWidget(self._diff_render_box)

        multi_widget = QWidget()
        multi_widget.setLayout(layout)

        self.setFloating(False)
        self.setWidget(multi_widget)
