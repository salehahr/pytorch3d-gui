import os

from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QMainWindow, QToolBar, QLabel

from .Sidebar import Sidebar
from .Graphics import Graphics
from .Panes import Panes

obj_filepath = '/graphics/scratch/schuelej/sar/pytorch3d-gui/data/cow.obj'
obj_filename = os.path.basename(obj_filepath)
device_str = 'cuda:0'


class Viewer(QMainWindow):
    def __init__(self):
        super(Viewer, self).__init__()

        # default mesh on startup
        self.image_size = 256
        self.filepath: str = obj_filepath
        self.filename = obj_filename

        # ui elements
        self._sidebar: Optional[QWidget] = None
        self._status_bar: Optional[QWidget] = None
        self._graphics: Optional[QWidget] = None
        self._target_pane: Optional[QWidget] = None
        self._render_pane: Optional[QWidget] = None
        self._init_ui()

        # camera params
        self.prev_pos = (0, 0)

        self.resize(self.sizeHint())
        self.show()

    def _init_ui(self) -> None:
        # self._init_menu_bar()
        # self._init_tool_bar()

        self._init_graphics()
        self.setCentralWidget(self._graphics)

        self._init_sidebar()
        self._init_status_bar()

        self.setWindowTitle(f"Viewer - {self.filepath}")

    def _init_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        label = 'TEST'
        menu_bar.addMenu(label)

    def _init_tool_bar(self) -> None:
        file_tool_bar = QToolBar("File")
        file_tool_bar.addWidget(QLabel("File: "))
        file_tool_bar.addWidget(QLabel(f'{self.filename}'))

        self._file_tool_bar = self.addToolBar(file_tool_bar)

    def _init_sidebar(self) -> None:
        self._sidebar = Sidebar(self._graphics, '')
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar)

    def _init_status_bar(self) -> None:
        self._status_bar = self.statusBar()
        self._status_bar.showMessage(self._graphics.camera_params_string)

    def _init_graphics(self) -> None:
        self._graphics = Graphics(self.image_size, parent=self)

        self._target_pane = self._graphics.get_pane(Panes.TARGET)
        self._render_pane = self._graphics.get_pane(Panes.RENDER)

        self._target_pane.load_mesh(obj_filepath)

    def mousePressEvent(self, e):
        self.prev_pos = (e.x(), e.y())

    def mouseMoveEvent(self, e):
        if not self.is_loaded:
            return

        dist, elev, azim = self.camera_params

        # Adjust rotation speed
        azim = azim + (self.prev_pos[0] - e.x()) * 0.25
        elev = elev - (self.prev_pos[1] - e.y()) * 0.25

        self.camera_params = [dist, elev, azim]

        self.prev_pos = (e.x(), e.y())

    def wheelEvent(self, e):
        if not self.is_loaded:
            return

        dist, elev, azim = self.camera_params

        # Adjust rotation speed
        dist = dist - e.angleDelta().y() * 0.01

        self.camera_params = [dist, elev, azim]

    def sizeHint(self) -> QSize:
        width = self._sidebar.width + self._graphics.width
        height = max(self._sidebar.height, self._graphics.height)
        return QSize(width, height)

    @property
    def render_pane(self) -> QWidget:
        return self._render_pane

    @property
    def camera_params(self) -> list:
        return self._graphics.camera_params

    @camera_params.setter
    def camera_params(self, value: list):
        self._graphics.camera_params = value

    @property
    def is_loaded(self) -> bool:
        return self._graphics.is_loaded
