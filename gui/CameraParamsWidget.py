from PyQt5.QtWidgets import QDockWidget, QFormLayout, \
    QLineEdit, QPushButton, QGroupBox


class CameraParamsWidget(QDockWidget):
    def __init__(self, main_window, camera_params: list,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._main_window = main_window

        camera_params_str = [f'{x:000.1f}' for x in camera_params]
        self._distance = QLineEdit(camera_params_str[0])
        self._elevation = QLineEdit(camera_params_str[1])
        self._azimuth = QLineEdit(camera_params_str[2])

        self._dea_box = self._init_dea_box()

        self.setFloating(False)
        self.setWidget(self._dea_box)

    @property
    def dea(self):
        distance = float(self._distance.text())
        elevation = float(self._elevation.text())
        azimuth = float(self._azimuth.text())

        return [distance, elevation, azimuth]

    def _init_dea_box(self) -> QGroupBox:
        box = QGroupBox('View parameters')

        layout = QFormLayout()
        layout.addRow('D', self._distance)
        layout.addRow('E', self._elevation)
        layout.addRow('A', self._azimuth)

        apply_button = QPushButton('Apply')
        layout.addRow(apply_button)
        apply_button.clicked.connect(self.set_dea)

        box.setLayout(layout)

        return box

    def set_dea(self):
        self._main_window.camera_params = self.dea
        self._main_window.display_rendered_image()
