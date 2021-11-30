import sys

from PyQt5.QtWidgets import QApplication

from gui import Viewer


if __name__ == '__main__':
    app = QApplication([])
    viewer = Viewer()
    sys.exit(app.exec())
