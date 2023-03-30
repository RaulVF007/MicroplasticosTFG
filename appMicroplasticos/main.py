from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("MainWindow.ui", self)
        self.show()
        self.fileName = ""
        pixmap = QtGui.QPixmap(self.fileName)
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.setMinimumSize(1, 1)

        self.actionOpen_image.triggered.connect(self.openImage)

        self.actionExit.triggered.connect(QApplication.quit)

        self.actionAbout_the_app.triggered.connect(self.aboutTheApp)

    def openImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Image (*.png *.jpg *.jpeg)",
                                                       options=options)
        if self.fileName:
            pixmap = QtGui.QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.width(), self.height())
            self.label.setPixmap(pixmap)

    def resizeEvent(self, event):
        try:
            pixmap = QtGui.QPixmap(self.fileName)
        except:
            pixmap = QtGui.QPixmap("")

        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.resize(self.width(), self.height())

    def aboutTheApp(self):
        QMessageBox.about(self, "About the app", "App made by Raúl Vega"
         "\n1.- Choose an image you can identify microplastics in 'File' -> 'Open image'"
         "\n2.- Start the identifying microplastics process in 'Edit' -> 'Identify microplastics'"
         "\n3.- After executing the previous process, you can save the results by pressing 'File' -> 'Save results'")


def main():
    app = QApplication([])
    window = GUI()
    app.exec_()


if __name__ == "__main__":
    main()
