from PyQt5.QtCore import Qt
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

        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.setWidget(multi_widget)

        for c in multi_widget.children():
            if isinstance(c, QVBoxLayout):
                continue

            c.setFixedHeight(c.sizeHint().height())

        layout.setAlignment(Qt.AlignTop)