from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLineEdit, QPushButton


class CameraParamsWidget(QGroupBox):
    def __init__(self, main_window,
                 camera_params: list,
                 title: str = 'View parameters',
                 *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._main_window = main_window

        self._distance = 0
        self._elevation = 0
        self._azimuth = 0
        self._init_dea(camera_params)

        self._init_layout()

    def _init_dea(self, camera_params):
        camera_params_str = [f'{x:000.1f}' for x in camera_params]
        self._distance = QLineEdit(camera_params_str[0])
        self._elevation = QLineEdit(camera_params_str[1])
        self._azimuth = QLineEdit(camera_params_str[2])

    def _init_layout(self):
        layout = QFormLayout()
        layout.addRow('D', self._distance)
        layout.addRow('E', self._elevation)
        layout.addRow('A', self._azimuth)

        apply_button = QPushButton('Apply')
        layout.addRow(apply_button)
        apply_button.clicked.connect(self.set_dea)

        self.setLayout(layout)

    @property
    def dea(self):
        distance = float(self._distance.text())
        elevation = float(self._elevation.text())
        azimuth = float(self._azimuth.text())

        return [distance, elevation, azimuth]

    def set_dea(self):
        self._main_window.camera_params = self.dea
        self._main_window.display_rendered_image()
