import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

from MeshLoader import MeshLoader
from gui import LightWidget

obj_filepath = '/graphics/scratch/schuelej/sar/pytorch3d-gui/data/cow.obj'
obj_filename = os.path.basename(obj_filepath)


def _img_to_pixmap(im: np.ndarray, copy: bool = False):
    assert len(im.shape) == 3
    height, width, n_channels = im.shape

    qim = QImage()

    if n_channels == 3:
        _qim = QImage(im.data, width, height, im.strides[0], QImage.Format_RGB888)
        qim = _qim.copy() if copy else _qim

    elif n_channels == 4:
        _qim = QImage(im.data, width, height, im.strides[0], QImage.Format_ARGB32)
        qim = _qim.copy() if copy else _qim

    return QPixmap.fromImage(qim)


class Viewer(QMainWindow):
    def __init__(self,
                 allow_resize: bool = True):
        super(Viewer, self).__init__()

        self.width = 100
        self.height = 100

        # image box
        self._image_box = QLabel()
        self._init_image_box(allow_resize)

        # default mesh on startup
        self.filepath = obj_filepath
        self.filename = obj_filename
        self.mesh_loader = MeshLoader()
        self.mesh_loader.load_file(filepath=obj_filepath)

        # camera params
        self.camera_params = [2.7, 10, -150]
        self.display_rendered_image()

        # position
        self.prev_pos = (0, 0)

        self._init_ui()

    def _init_ui(self):
        # self._init_menu_bar()
        # self._init_tool_bar()
        self._init_light_widget()
        self._init_status_bar()

        self.setWindowTitle(f"Viewer - {self.filepath}")
        self.resize(self.width, self.height)
        self.show()

    def _init_menu_bar(self):
        menu_bar = self.menuBar()
        label = 'TEST'
        menu_bar.addMenu(label)

    def _init_tool_bar(self):
        file_tool_bar = QToolBar("File")
        file_tool_bar.addWidget(QLabel("File: "))
        file_tool_bar.addWidget(QLabel(f'{self.filename}'))

        self._file_tool_bar = self.addToolBar(file_tool_bar)

    def _init_light_widget(self):
        self._light_widget = LightWidget(self, self.light_location, 'Lights')
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._light_widget)

    def _init_status_bar(self):
        self._status_bar = self.statusBar()
        self._status_bar.showMessage(self.camera_params_string)

    def _init_image_box(self, allow_resize: bool):
        if allow_resize:
            self._image_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self._image_box.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(self._image_box)

    def display_rendered_image(self):
        image = self.mesh_loader.render()
        self._display_image(image)

    def display_texture_map(self):
        image = self.mesh_loader.get_texture_map()
        self._display_image(image)

    def _display_image(self, image):
        pixmap = _img_to_pixmap(image)

        self._image_box.setPixmap(pixmap)

        self.width = pixmap.width()
        self.height = pixmap.height()

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

        self._status_bar.clearMessage()
        self._status_bar.showMessage(self.camera_params_string)

        self.display_rendered_image()

        self.prev_pos = (e.x(), e.y())

    def wheelEvent(self, e):
        if not self.is_loaded:
            return

        dist, elev, azim = self.camera_params

        # Adjust rotation speed
        dist = dist - e.angleDelta().y() * 0.01

        self.camera_params = [dist, elev, azim]

        self._status_bar.clearMessage()
        self._status_bar.showMessage(self.camera_params_string)

        self.display_rendered_image()

    @property
    def light_location(self):
        return self.mesh_loader.light_location.cpu().numpy()[0]

    @property
    def camera_params(self):
        return self.mesh_loader.camera_params

    @camera_params.setter
    def camera_params(self, value: list):
        self.mesh_loader.camera_params = value

    @property
    def camera_params_string(self):
        return f'Distance: {self.camera_params[0]:000.1f} | ' \
               + f'Azimuth: {self.camera_params[1]:000.0f} | ' \
               + f'Elevation: {self.camera_params[2]:000.0f}'

    @property
    def is_loaded(self):
        return self.mesh_loader.is_loaded
