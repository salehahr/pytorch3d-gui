from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox


class DiffRenderWidget(QGroupBox):
    def __init__(self, render_pane,
                 title: str = 'Differential rendering',
                 *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._render_pane: QWidget = render_pane

        self._init_layout()

    def _init_layout(self) -> None:
        button = QPushButton('Render')
        button.clicked.connect(self._render)

        layout = QVBoxLayout()
        layout.addWidget(button)

        self.setLayout(layout)

    def _render(self) -> None:
        if not self._render_pane.mesh_is_loaded:
            return
        self._render_pane.diff_renderer.render()
