from PyQt5.QtWidgets import QDockWidget, QFormLayout, \
    QLineEdit, QPushButton, QGroupBox


class LightWidget(QDockWidget):
    def __init__(self, main_window, light_location: list,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._main_window = main_window

        self._user_light_x = QLineEdit(str(light_location[0]))
        self._user_light_y = QLineEdit(str(light_location[1]))
        self._user_light_z = QLineEdit(str(light_location[2]))

        self._light_location_box = self._init_light_location_box()

        self.setFloating(False)
        self.setWidget(self._light_location_box)

    def _init_light_location_box(self) -> QGroupBox:
        light_location_box = QGroupBox('Light location')

        light_location_layout = QFormLayout()
        light_location_layout.addRow('x', self._user_light_x)
        light_location_layout.addRow('y', self._user_light_y)
        light_location_layout.addRow('z', self._user_light_z)

        apply_button = QPushButton('Apply')
        light_location_layout.addRow(apply_button)
        apply_button.clicked.connect(self.user_set_light_location)

        light_location_box.setLayout(light_location_layout)

        return light_location_box

    def user_set_light_location(self):
        self._main_window.light_location = self.light_location
        self._main_window.display_rendered_image()

    @property
    def light_location(self):
        x = float(self._user_light_x.text())
        y = float(self._user_light_y.text())
        z = float(self._user_light_z.text())

        return [x, y, z]
