import json
import pandas as pd
import numpy as np

from scipy.misc import imread, imsave
from skimage.transform import resize

GTSDB_PATH = '../dataset/gtsdb/'
NEW_WIDTH = 640
NEW_HEIGHT = 480

metadata = pd.read_csv(GTSDB_PATH + 'metadata.csv', header=None, sep=';')


def draw_rectangle(image, rect):
    color = np.array([0, 1.0, 0], dtype=np.float32)

    image[rect[1], rect[0]:rect[2]] = color
    image[rect[1]:rect[3], rect[0]] = color
    image[rect[3], rect[0]:rect[2]] = color
    image[rect[1]:rect[3], rect[2]] = color


def resize_bounding_box(bbox, w_ratio, h_ratio):
    x_left = int(bbox[0] * w_ratio)
    y_left = int(bbox[1] * h_ratio)
    x_right = int(bbox[2] * w_ratio)
    y_right = int(bbox[3] * h_ratio)
    return x_left, y_left, x_right, y_right


current_image = {}
current_image_shape = None
annotations = []

for i, filename, x_left, y_left, x_right, y_right, _ in metadata.itertuples():
    image_path = GTSDB_PATH + 'jpg/' + filename.replace('ppm', 'jpg')

    if image_path == current_image.get('image_path'):
        h_ratio = float(NEW_HEIGHT) / current_image_shape[0]
        w_ratio = float(NEW_WIDTH) / current_image_shape[1]

        bbox = resize_bounding_box(
            (x_left, y_left, x_right, y_right), w_ratio, h_ratio
        )

        current_image['rects'].append({
            'x1': bbox[0],
            'y1': bbox[1],
            'x2': bbox[2],
            'y2': bbox[3]
        })
    else:
        if current_image != {}:
            annotations.append(current_image)
            current_image = {}

        image = imread(GTSDB_PATH + filename)
        current_image_shape = image.shape

        resized_image = resize(image, (NEW_HEIGHT, NEW_WIDTH), mode='constant')
        imsave(image_path, resized_image)

        h_ratio = float(NEW_HEIGHT) / image.shape[0]
        w_ratio = float(NEW_WIDTH) / image.shape[1]

        bbox = resize_bounding_box(
            (x_left, y_left, x_right, y_right), w_ratio, h_ratio
        )

        current_image['image_path'] = image_path
        current_image['rects'] = [{
            'x1': bbox[0],
            'y1': bbox[1],
            'x2': bbox[2],
            'y2': bbox[3]
        }]

    print("{0} / {1}".format(i, metadata.shape[0]))

encoder = json.JSONEncoder(indent=4, sort_keys=True)

with open('train_boxes.json', 'w') as f:
    json_file = encoder.encode(annotations)
    f.write(json_file)
