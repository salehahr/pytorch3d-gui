from typing import Optional

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QWidget, QGroupBox, QFormLayout, QLineEdit, QPushButton


class CameraParamsWidget(QGroupBox):
    def __init__(self, graphics: QWidget,
                 title: str = 'View parameters',
                 *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._graphics = graphics

        self._distanc: Optional[QLineEdit] = None
        self._elevation: Optional[QLineEdit] = None
        self._azimuth: Optional[QLineEdit] = None
        self._init_dea(graphics.camera_params)

        self._init_layout()

    def _init_dea(self, camera_params: list) -> None:
        camera_params_str = [f'{x:000.1f}' for x in camera_params]

        self._distance = QLineEdit(camera_params_str[0])
        self._elevation = QLineEdit(camera_params_str[1])
        self._azimuth = QLineEdit(camera_params_str[2])

        validator_distance = QDoubleValidator(0, 10, 2)
        validator_elevation = QDoubleValidator(0, 90, 2)
        validator_azimuth = QDoubleValidator(0, 360, 2)

        self._distance.setValidator(validator_distance)
        self._elevation.setValidator(validator_elevation)
        self._azimuth.setValidator(validator_azimuth)

    def _init_layout(self) -> None:
        layout = QFormLayout()
        layout.addRow('D', self._distance)
        layout.addRow('E', self._elevation)
        layout.addRow('A', self._azimuth)

        apply_button = QPushButton('Apply')
        apply_button.clicked.connect(self.set_dea)

        layout.addRow(apply_button)

        self.setLayout(layout)

    @property
    def dea(self) -> list:
        distance = float(self._distance.text())
        elevation = float(self._elevation.text())
        azimuth = float(self._azimuth.text())

        return [distance, elevation, azimuth]

    def set_dea(self) -> None:
        self._graphics.camera_params = self.dea
