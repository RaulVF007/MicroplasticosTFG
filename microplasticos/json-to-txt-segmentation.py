import os
import json


def json_to_txt_segmentation():
    path_to_json = 'C:/Users/raulv/MicroplasticosTFG/json'
    path_to_labels = 'C:/Users/raulv//MicroplasticosTFG/dataset/labels'

    for file in os.listdir(path_to_json):
        json_path = os.path.join(path_to_json, file)
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        base_name = os.path.splitext(file)[0]
        txt_path = os.path.join(path_to_labels, base_name + ".txt")

        with open(txt_path, "w") as txt_file:
            shapes = json_data['shapes']
            for i, shape in enumerate(shapes):
                element = shape['label']
                if element == 'Fragment':
                    number_class = '0'
                elif element == 'Line':
                    number_class = '1'
                elif element == 'Pellet':
                    number_class = '2'
                else:
                    number_class = '-1'

                if i > 0:
                    txt_file.write('\n')
                txt_file.write(number_class)
                points = shape['points']

                for j, point in enumerate(points):
                    point_x = point[0]
                    point_y = point[1]
                    width = json_data['imageWidth']
                    height = json_data['imageHeight']
                    x = str(point_x / width)
                    y = str(point_y / height)
                    content = " " + x + " " + y
                    txt_file.write(content)
            json_file.close()
            txt_file.close()

json_to_txt_segmentation()