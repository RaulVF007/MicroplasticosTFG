import os
from PyQt5 import QtWidgets, QtGui, uic
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from yolov5.utils.segment.general import process_mask
from yolov5.utils.general import non_max_suppression, scale_boxes, Profile
from yolov5.utils.plots import Annotator, colors
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("MainWindow.ui", self)
        self.show()
        self.actionIdentify_microplastics.setEnabled(False)
        self.actionSave_results.setEnabled(False)
        self.actionExport_parameters.setEnabled(False)

        self.fileName = ""
        self.image = ""
        self.dataframe = []

        pixmap = QtGui.QPixmap(self.fileName)
        pixmap = pixmap.scaled(self.width(), self.height())

        self.label.setPixmap(pixmap)
        self.label.setMinimumSize(1, 1)

        self.actionOpen_image.triggered.connect(self.openImage)

        self.actionIdentify_microplastics.triggered.connect(self.identifyMicroplastics)

        self.actionSave_results.triggered.connect(self.saveResults)

        self.actionExport_parameters.triggered.connect(self.exportResults)

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
            self.image = pixmap
            self.actionIdentify_microplastics.setEnabled(True)

    def resizeEvent(self, event):
        try:
            pixmap = QtGui.QPixmap(self.image)
        except:
            pixmap = QtGui.QPixmap("")
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.resize(self.width(), self.height())

    def identifyMicroplastics(self):
        device = select_device("cpu")
        model = DetectMultiBackend("best.pt", device=device, dnn=False, data="data.yaml", fp16=False)
        names = model.names
        dt = (Profile(), Profile(), Profile())

        image = cv2.imread(self.fileName)

        with dt[0]:
            im = torch.from_numpy(image).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
                im = np.transpose(im, (0, 3, 1, 2))

        with dt[1]:
            pred, proto = model(im, augment=False, visualize=False)[:2]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5, max_det=1000, nm=32)

        for i, det in enumerate(pred):
            annotator = Annotator(image, line_width=3, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()

                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(image, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                        0).contiguous() / 255)

                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    xyxy = [coord.tolist() for coord in xyxy]
                    enabledPixels = np.where(masks[j] > 0)
                    pixelsPerMicroplastics = len(enabledPixels[0])

                    microplastic = {
                        "tipo": names[c],
                        "coordenadas": xyxy,
                        "confianza": "{:.2f}".format(conf),
                        "pixeles": pixelsPerMicroplastics
                    }
                    self.dataframe.append(microplastic)

            image = annotator.result()

        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)

        pixmap = QtGui.QPixmap.fromImage(q_image)

        self.image = q_image
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.actionSave_results.setEnabled(True)
        self.actionExport_parameters.setEnabled(True)

    def saveResults(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self, "Select location for saving result")

        file_name = os.path.basename(self.fileName)
        file_root, file_ext = os.path.splitext(file_name)
        new_file_name = f"{file_root}_identified{file_ext}"
        new_file_path = os.path.join(dir_name, new_file_name)

        qImage= self.label.pixmap().toImage()
        resized_image = qImage.scaled(1280, 1024)
        image = np.array(resized_image.bits().asarray(resized_image.width() * resized_image.height() * 4)).reshape(
            (resized_image.height(), resized_image.width(), 4))[:, :, :3]
        cv2.imwrite(new_file_path, image)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("The image has been saved correctly.")
        msg.setWindowTitle("Save result")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def exportResults(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select location for exporting file")

        file_name = os.path.basename(self.fileName)
        file_root, _ = os.path.splitext(file_name)
        new_file_name = file_root + ".txt"
        new_file_path = os.path.join(dir_name, new_file_name)

        with open(new_file_path, 'w') as file:
            file.write("Tipo de microplástico,Coordenadas,Confianza,Número de píxeles\n")

            for m in self.dataframe:
                tipo = m['tipo']
                coordenadas = ','.join([str(c) for c in m['coordenadas']])
                confianza = str(m['confianza'])
                pixeles = str(m['pixeles'])

                row = f"{tipo},{coordenadas},{confianza},{pixeles}\n"
                file.write(row)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("The file has been exported successfully.")
        msg.setWindowTitle("Export file")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def aboutTheApp(self):
        QMessageBox.about(self, "About the app", "App made by Raúl Vega\n"
         "\n1.- Choose an image you can identify microplastics in 'File' -> 'Open image'"
         "\n2.- Start the identifying microplastics process in 'Process' -> 'Microplastics'"
         "\n3.- After executing the previous process, you can save the results by pressing 'File' -> 'Save results'"
         "\n4.- After executing the process 2, you can also export a file with parameters related to the segmentation" \
         " proccess in 'Process' -> 'Export parameters'")

def main():
    app = QApplication([])
    window = GUI()
    app.exec_()

if __name__ == "__main__":
    main()
