import numpy as np

from enum import Enum

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QGridLayout


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


class Pane(Enum):
    RENDER = 0
    TARGET = 1


class Graphics(QLabel):

    def __init__(self,
                 image_size: int,
                 allow_resize: bool = True,
                 *args, **kwargs):
        super(Graphics, self).__init__(*args, **kwargs)

        self._image_size = image_size

        self._num_panes = len(Pane)
        self._panes = [GraphicsPane(_id.value, allow_resize) for _id in Pane]
        self._init_layout()

        if allow_resize:
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.setAlignment(Qt.AlignCenter)

    def _init_layout(self):
        layout = QGridLayout()
        layout.setRowMinimumHeight(0, self._image_size)

        for i, p in enumerate(self._panes):
            layout.addWidget(p, 0, i, Qt.AlignCenter)
            layout.setColumnMinimumWidth(i, self._image_size)

        self.setLayout(layout)

    def display(self, image, pane_type):
        pane = self._panes[pane_type.value]
        pane.display(image)

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()


class GraphicsPane(QLabel):
    def __init__(self,
                 _id: Pane,
                 allow_resize: bool = True,
                 *args, **kwargs):
        super(GraphicsPane, self).__init__(*args, **kwargs)
        self.type = Pane(_id)
        self.id = _id

        if allow_resize:
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def display(self, image):
        pixmap = _img_to_pixmap(image)
        self.setPixmap(pixmap)
