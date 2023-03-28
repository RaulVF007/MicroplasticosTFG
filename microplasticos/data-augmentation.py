import os
import cv2
import albumentations as A

def data_augmentation():
    directory = "C:/Users/raulv/Desktop/prueba"
    transform = A.Compose([
        A.RandomBrightnessContrast(p=1),
    ])

    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            name, ext = os.path.splitext(filename)
            img = cv2.imread(os.path.join(directory, filename))

            img_flipped_vertical = cv2.flip(img, 0)
            img_flipped_horizontal = cv2.flip(img, 1)

            cv2.imwrite(os.path.join(directory, name +"-ver-flip" + ext), img_flipped_vertical)
            cv2.imwrite(os.path.join(directory, name + "-hor-flip" + ext), img_flipped_horizontal)

            transformed = transform(image=img)
            img_bright_contrast = transformed["image"]

            cv2.imwrite(os.path.join(directory, name + "-bright-contrast" + ext), img_bright_contrast)

data_augmentation()