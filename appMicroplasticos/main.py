import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import torch
from PyQt5 import QtGui, uic
from PyQt5.uic.properties import QtWidgets

from yolov5.utils.general import non_max_suppression
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
        if len(pred[0]) > 0:
            # Aplicar Non-Maximum Suppression
            conf_thresh = 0.55  # Umbral de confianza
            iou_thresh = 0.55  # Umbral de IoU
            pred = non_max_suppression(pred, conf_thresh, iou_thresh)

            for obj in pred[0]:
                    # Obtener las coordenadas en formato (x1, y1, x2, y2)
                    x1, y1, x2, y2 = obj[:4]
                    # Dibujar un rectángulo en la imagen
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Mostrar la imagen en el QLabel
            h, w, ch = image.shape
            bytesPerLine = ch * w
            convertToQtFormat = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 640, Qt.KeepAspectRatio)
            self.label.setPixmap(QtGui.QPixmap.fromImage(p))
            # Habilitar la opción de guardar resultados
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
