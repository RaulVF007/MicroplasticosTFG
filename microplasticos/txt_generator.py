import os
import shutil


def txt_generator():
    path_to_images = "C:/Users/raulv/Desktop/prueba/images"
    path_to_txts = "C:/Users/raulv/Desktop/prueba/txts"

    imgFiles = os.listdir(path_to_images)

    for img in imgFiles:
        if img.endswith("-ver-flip.JPG") or img.endswith("-ver-flip-bright-contrast.JPG"):
            title = img.replace(".JPG", ".txt")
            txts_path = os.path.join(path_to_txts, title)

            originalTxtTitle = title.replace("-ver-flip", "")
            originalTxt = os.path.join(path_to_txts, originalTxtTitle)
            with open(originalTxt, "r") as f:
                lines = f.readlines()
                result = ""
                for line in lines:
                    content = line.split()
                    index = 0
                    for i in content:
                        if index % 2 == 0 and "." in i:
                            result += (str(float(1) - float(i))) + " "
                            index += 1

                        elif index % 2 == 1 and "." in i:
                            result += (str(i)) + " "
                            index += 1

                        else:
                            result += (str(i)) + " "
                            index += 1

                    result += "\n"
                with open(txts_path, "w") as f:
                    f.write(result)

        elif img.endswith("-hor-flip.JPG") or img.endswith("-hor-flip-bright-contrast.JPG"):
            title = img.replace(".JPG", ".txt")
            txts_path = os.path.join(path_to_txts, title)

            if img.endswith("-hor-flip.JPG"):
                originalTxtTitle = title.replace("-hor-flip", "")

            if img.endswith("-hor-flip-bright-contrast.JPG"):
                originalTxtTitle = title.replace("-hor-flip-bright-contrast", "")

            originalTxt = os.path.join(path_to_txts, originalTxtTitle)
            with open(originalTxt, "r") as f:
                lines = f.readlines()
                result = ""
                for line in lines:
                    content = line.split()
                    index = 0
                    for i in content:
                        if index % 2 == 0 and "." in i:
                            result += (str(i)) + " "
                            index += 1

                        elif index % 2 == 1 and "." in i:
                            result += (str(float(1) - float(i))) + " "
                            index += 1

                        else:
                            result += (str(i)) + " "
                            index += 1

                    result += "\n"
            with open(txts_path, "w") as f:
                f.write(result)

        elif img.endswith("-bright-contrast.JPG"):
            title = img.replace(".JPG", ".txt")
            txts_path = os.path.join(path_to_txts, title)
            src = txts_path.replace("-bright-contrast","")
            shutil.copy(src, txts_path)

txt_generator()
