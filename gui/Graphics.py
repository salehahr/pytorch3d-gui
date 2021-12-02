from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QGridLayout

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .ImageRenderer import ImageRendererDynamic
from .DiffRenderer import DiffRenderer
from .Panes import Panes


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


class Graphics(QLabel):

    def __init__(self,
                 image_size: int,
                 background_colour: tuple,
                 allow_resize: bool = True,
                 *args, **kwargs):
        super(Graphics, self).__init__(*args, **kwargs)

        self._image_size = image_size
        self._renderer = ImageRendererDynamic(background_colour, image_size)

        self._num_panes = len(Panes)
        self._panes = [GraphicsPane(_id, self._renderer, image_size, allow_resize)
                       for _id in range(self._num_panes)]
        self._init_layout()

        if allow_resize:
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.diff_renderer = DiffRenderer(self._panes[Panes.RENDER.value], self._renderer)

        self.setStyleSheet(f'background-color: rgb{str(background_colour)};')
        self.setAlignment(Qt.AlignCenter)

    def _init_layout(self) -> None:
        layout = QGridLayout()
        layout.setRowMinimumHeight(0, self._image_size)

        for i, p in enumerate(self._panes):
            layout.addWidget(p, 0, i, Qt.AlignCenter)
            layout.setColumnMinimumWidth(i, self._image_size)

        self.setLayout(layout)

    def update(self) -> None:
        for p in self._panes:
            p.display()

    def get_pane(self, pane: Panes) -> GraphicsPane:
        return self._panes[pane.value]

    @property
    def renderer(self) -> ImageRendererDynamic:
        return self._renderer

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()

    @property
    def camera_params(self) -> list:
        return self._renderer.camera_params

    @camera_params.setter
    def camera_params(self, value: list) -> None:
        self._renderer.camera_params = value
        self.update()

        self.parent().statusBar().clearMessage()
        self.parent().statusBar().showMessage(self.camera_params_string)

    @property
    def camera_params_string(self) -> str:
        return f'Distance: {self.camera_params[0]:000.1f} | ' \
               + f'Elevation: {self.camera_params[1]:000.0f} | ' \
               + f'Azimuth: {self.camera_params[2]:000.0f}'

    @property
    def is_loaded(self) -> bool:
        return True in [p.mesh_is_loaded for p in self._panes]


class GraphicsPane(QLabel):
    def __init__(self,
                 _id: int,
                 renderer,
                 image_size: int,
                 allow_resize: bool = True,
                 *args, **kwargs):
        super(GraphicsPane, self).__init__(*args, **kwargs)
        self.type = Panes(_id)
        self.id = _id

        self._renderer: ImageRendererDynamic = renderer
        self._device = renderer.device

        self._image_size: int = image_size
        self._mesh: Optional[Meshes] = None
        self._mesh_is_loaded: bool = False

        if allow_resize:
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        if self.type == Panes.RENDER:
            self.load_mesh_sphere()

    def display(self, image: Optional[np.ndarray] = None) -> None:
        if not self._mesh_is_loaded and image is None:
            return

        if image is None:  # render mesh instead
            with torch.no_grad():
                image = self._renderer.render(self._mesh)

        pixmap = _img_to_pixmap(image)
        self.setPixmap(pixmap)

    def load_mesh(self, filepath: str) -> None:
        self.mesh = load_objs_as_meshes([filepath], device=self._device)

        if self.type == Panes.TARGET:
            self.diff_renderer.set_target_mesh(self.mesh)

    def load_mesh_sphere(self) -> None:
        mesh = ico_sphere(4, self._device)

        verts_rgb = torch.ones_like(mesh.verts_packed())[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self._device))

        self.mesh = Meshes(verts=mesh.verts_list(),
                           faces=mesh.faces_list(),
                           textures=textures
                           )

    @property
    def mesh(self) -> Meshes:
        return self._mesh

    @mesh.setter
    def mesh(self, value: Meshes) -> None:
        self._mesh = value

        if value is not None:
            self._mesh_is_loaded = True
            self.display()

    @property
    def mesh_is_loaded(self) -> bool:
        return self._mesh_is_loaded

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def diff_renderer(self):
        return self.parent().diff_renderer
