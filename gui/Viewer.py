import os

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import *

from .Sidebar import Sidebar
from .Graphics import Graphics, Pane

from .ImageRenderer import ImageRendererStatic, ImageRendererDynamic
from .DiffRenderer import DiffRenderer

obj_filepath = '/graphics/scratch/schuelej/sar/pytorch3d-gui/data/cow.obj'
obj_filename = os.path.basename(obj_filepath)
device = 'cuda:0'


class Viewer(QMainWindow):
    def __init__(self,
                 allow_resize: bool = True):
        super(Viewer, self).__init__()

        self.image_size = 256

        # default mesh on startup
        self.filepath = obj_filepath
        self.filename = obj_filename

        self.static_renderer = ImageRendererStatic(self.image_size, device)
        self.dynamic_renderer = ImageRendererDynamic(obj_filepath, self.image_size, device)
        self.renderers = {Pane.RENDER: self.static_renderer,
                          Pane.TARGET: self.dynamic_renderer
                          }
        self.diff_renderer = DiffRenderer(self, self.renderers, device)

        self._init_ui(allow_resize)

        # camera params
        self.prev_pos = (0, 0)
        self.display_target_mesh()

        self.resize(self.sizeHint())
        self.show()

    def _init_ui(self, allow_resize):
        # self._init_menu_bar()
        # self._init_tool_bar()
        self._init_sidebar()
        self._init_status_bar()

        self._init_graphics_panes(allow_resize)
        self.setCentralWidget(self._graphics)

        self.setWindowTitle(f"Viewer - {self.filepath}")

    def _init_menu_bar(self):
        menu_bar = self.menuBar()
        label = 'TEST'
        menu_bar.addMenu(label)

    def _init_tool_bar(self):
        file_tool_bar = QToolBar("File")
        file_tool_bar.addWidget(QLabel("File: "))
        file_tool_bar.addWidget(QLabel(f'{self.filename}'))

        self._file_tool_bar = self.addToolBar(file_tool_bar)

    def _init_sidebar(self):
        self._sidebar = Sidebar(self, self.camera_params, '')
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar)

    def _init_status_bar(self):
        self._status_bar = self.statusBar()
        self._status_bar.showMessage(self.camera_params_string)

    def _init_graphics_panes(self, allow_resize: bool):
        self._graphics = Graphics(self.image_size,
                                  self.renderers,
                                  allow_resize, parent=self)

    def display_target_mesh(self):
        image = self.dynamic_renderer.render()
        self._graphics.display(image, pane_type=Pane.TARGET)

    def display_rendered_mesh(self, image):
        self._graphics.display(image, pane_type=Pane.RENDER)

    def display_texture_map(self):
        image = self.dynamic_renderer.get_texture_map()
        self._graphics.display(image)

    def differential_render(self):
        if not self.is_loaded:
            return

        self.diff_renderer.render()

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

        self.display_target_mesh()

        self.prev_pos = (e.x(), e.y())

    def wheelEvent(self, e):
        if not self.is_loaded:
            return

        dist, elev, azim = self.camera_params

        # Adjust rotation speed
        dist = dist - e.angleDelta().y() * 0.01

        self.camera_params = [dist, elev, azim]

        self.display_target_mesh()

    def sizeHint(self):
        width = self._sidebar.width + self._graphics.width
        height = max(self._sidebar.height, self._graphics.height)
        return QSize(width, height)

    @property
    def camera_params(self):
        return self.dynamic_renderer.camera_params

    @camera_params.setter
    def camera_params(self, value: list):
        self.dynamic_renderer.camera_params = value

        self._status_bar.clearMessage()
        self._status_bar.showMessage(self.camera_params_string)

    @property
    def camera_params_string(self):
        return f'Distance: {self.camera_params[0]:000.1f} | ' \
               + f'Azimuth: {self.camera_params[1]:000.0f} | ' \
               + f'Elevation: {self.camera_params[2]:000.0f}'

    @property
    def is_loaded(self):
        return self.dynamic_renderer.is_loaded
