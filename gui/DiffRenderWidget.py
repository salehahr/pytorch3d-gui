from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox
from PyQt5.QtCore import QThread


class DiffRenderWidget(QGroupBox):
    def __init__(self, render_pane,
                 title: str = 'Differential rendering',
                 *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._render_pane: QWidget = render_pane

        self._button_text = 'Render'
        self._button = QPushButton(self._button_text)

        self._thread = QThread()
        self._worker = self._render_pane.diff_renderer
        self._init_threading()

        self._init_layout()

    def _init_layout(self) -> None:
        # TODO: add renderer settings
        self._button.clicked.connect(self._worker.render)
        self._button.clicked.connect(self._disable_button)

        layout = QVBoxLayout()
        layout.addWidget(self._button)

        self.setLayout(layout)

    def _init_threading(self) -> None:
        self._worker.progress.connect(self._update_mesh)
        self._worker.finished.connect(self._enable_button)

        self._worker.moveToThread(self._thread)
        self._thread.start()

    def _update_mesh(self, mesh) -> None:
        self._render_pane.mesh = mesh

    def _disable_button(self) -> None:
        self._button.setText('Rendering...')
        self._button.setDisabled(True)

    def _enable_button(self) -> None:
        self._button.setEnabled(True)
        self._button.setText(self._button_text)
