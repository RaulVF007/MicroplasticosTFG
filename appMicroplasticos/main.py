from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        # Carga el archivo .ui
        uic.loadUi("MainWindow.ui", self)
        self.show()
        self.fileName = ""
        pixmap = QtGui.QPixmap(self.fileName)
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.setMinimumSize(1,1)

        self.actionOpen_image.triggered.connect(self.openImage)

    def openImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self,"Open image", "","Image (*.png *.jpg *.jpeg)", options=options)
        if self.fileName:
            pixmap = QtGui.QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.width(), self.height())
            self.label.setPixmap(pixmap)

def main():
    app = QApplication([])
    window = GUI()
    app.exec_()

if __name__ == "__main__":
    main()