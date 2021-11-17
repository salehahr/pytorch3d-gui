import numpy as np

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtWidgets import QSizePolicy

from MeshLoader import MeshLoader


cow_obj_filepath = '/graphics/scratch/schuelej/sar/pytorch3d-gui/data/cow.obj'


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

        self.loaded = False
        self.mesh_loader = MeshLoader()

        self.imageLabel = QLabel()
        if allow_resize:
            self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCentralWidget(self.imageLabel)
        self.display_rendered_image(filepath=cow_obj_filepath)

        self.setWindowTitle("Viewer")
        self.resize(self.width, self.height)
        self.show()

    def display_rendered_image(self, filepath):
        image = self.mesh_loader.get_rendered_image(filepath)
        self._display_image(image)

    def display_texture_map(self, filepath):
        image = self.mesh_loader.get_texture_map(filepath)
        self._display_image(image)

    def _display_image(self, image):
        pixmap = _img_to_pixmap(image)

        self.imageLabel.setPixmap(pixmap)

        self.width = pixmap.width()
        self.height = pixmap.height()