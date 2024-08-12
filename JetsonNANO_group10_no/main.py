import sys
import os
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

if  __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = '1'
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    # screen_rect = QApplication.desktop().screenGeometry()
    # mainWindow.window.setFixedSize(1080, 600)

    mainWindow.window.show()
    ret = app.exec_()
    sys.exit(ret)
