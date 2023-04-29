from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
import cv2
import numpy as np
import torch
from PyQt5 import QtGui
from yolov5.models.common import DetectMultiBackend


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("MainWindow.ui", self)
        self.show()
        self.actionIdentify_microplastics.setEnabled(False)
        self.actionSave_results.setEnabled(False)

        self.fileName = ""
        pixmap = QtGui.QPixmap(self.fileName)
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.setMinimumSize(1, 1)

        self.actionOpen_image.triggered.connect(self.openImage)

        self.actionIdentify_microplastics.triggered.connect(self.identifyMicroplastics)

        self.actionExit.triggered.connect(QApplication.quit)

        self.actionAbout_the_app.triggered.connect(self.aboutTheApp)

    def openImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Image (*.png *.jpg *.jpeg)", options=options)
        if self.fileName:
            pixmap = QtGui.QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.width(), self.height())
            self.label.setPixmap(pixmap)
            self.actionIdentify_microplastics.setEnabled(True)

    def identifyMicroplastics(self):
        # Carga del modelo
        device = torch.device('cpu')
        weights = 'best.pt'
        model = DetectMultiBackend(weights, device=device)

        # Carga de la imagen
        image = cv2.imread(self.fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Ejecución del modelo
        im = image.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        pred, proto = model(im)[:2]
        print(pred)

        self.actionSave_results.setEnabled(True)

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
         "\n2.- Start the identifying microplastics process in 'Process' -> 'Microplastics'"
         "\n3.- After executing the previous process, you can save the results by pressing 'File' -> 'Save results'")

def main():
    app = QApplication([])
    window = GUI()
    app.exec_()


if __name__ == "__main__":
    main()
