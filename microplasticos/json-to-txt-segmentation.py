import os
import json


def json_to_txt_segmentation():
    path_to_json = 'C:/Users/raulv/Documents/MicroplasticosTFG/json'
    for file in os.listdir(path_to_json):
        with open(os.path.join(path_to_json, file), "r") as jsonFile:
            json_data = json.load(jsonFile)
        path_to_txt = os.path.join('C:/Users/raulv/Documents/MicroplasticosTFG/imagenes-microplasticos-etiquetar/labels', file)
        with open(os.path.splitext(path_to_txt)[0] + ".txt", "w") as txt:
            element = json_data['shapes'][0]['label']
            objectsNumber = len(json_data['shapes'])

            if element == 'Fragment':
                numberClass = '0'
            if element == 'Line':
                numberClass = '1'
            if element == 'Pellet':
                numberClass = '2'

            for object in range(objectsNumber):
                if (object > 0):
                    txt.write('\n')
                txt.write(numberClass)
                pointsNumber = len(json_data['shapes'][object]['points'])
                content = ""
                for point in range(pointsNumber):
                    pointX = json_data['shapes'][object]['points'][point][0]
                    pointY = json_data['shapes'][object]['points'][point][1]
                    width = json_data['imageWidth']
                    height = json_data['imageHeight']

                    x = str(pointX/width)
                    y = str(pointY/height)
                    content = " " + x + " " + y

                    txt.write(content)

json_to_txt_segmentation()
