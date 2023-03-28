import os
import cv2


def data_augmentation():
    directory = "C:/Users/raulv/Desktop/prueba"

    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            name, ext = os.path.splitext(filename)
            img = cv2.imread(os.path.join(directory, filename))

            img_flipped_vertical = cv2.flip(img, 0)

            cv2.imwrite(os.path.join(directory, name +"-ver-flip" + ext), img_flipped_vertical)

data_augmentation()