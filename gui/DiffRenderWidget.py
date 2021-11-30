from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QGroupBox


class DiffRenderWidget(QGroupBox):
    def __init__(self, main_window,
                 title: str = 'Differential rendering',
                 *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._main_window = main_window

        self._init_layout()

    def _init_layout(self):
        button = QPushButton('Render')
        button.clicked.connect(self._render)

        layout = QVBoxLayout()
        layout.addWidget(button)

        self.setLayout(layout)

    def _render(self):
        self._main_window.differential_render()
