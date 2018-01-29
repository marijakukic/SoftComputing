import json
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from scipy.misc import imread
from src.model import Model
from src.dataset import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "tensorbox"))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorbox.predict import initialize, hot_predict


def initialize_detector():
    hypes_path = 'tensorbox/output/overfeat_rezoom_2018_01_29_16.37/hypes.json'
    config = json.load(open(hypes_path, 'r'))

    weights_path = os.path.join(os.path.dirname(
        hypes_path), config['solver']['weights']
    )

    return initialize(weights_path, hypes_path)


def detect_traffic_signs(image_path):
    detector = initialize_detector()
    return hot_predict(image_path, detector)


def draw_rectangle(image, rect, label=None):
    color = np.array([0, 255, 0], dtype=np.uint8)

    image[rect[1], rect[0]:rect[2]] = color
    image[rect[1]:rect[3], rect[0]] = color
    image[rect[3], rect[0]:rect[2]] = color
    image[rect[1]:rect[3], rect[2]] = color

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 255, 0)
        x = (rect[0] + rect[2]) / 2 - (rect[2] - rect[0]) / 2
        y = rect[3] + 20
        label_point = (x, y)
        cv2.putText(image,
                    label, label_point, font,
                    0.8, font_color, 2, cv2.LINE_AA)


if __name__ == '__main__':
    parser = ArgumentParser(usage='usage: %prog --image <image>')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    predictions = detect_traffic_signs(args.image)
    image = imread(args.image)

    input_shape = (48, 48, 3)
    num_classes = 4
    cnn_weights_path = 'model/weights.h5'
    DELTA = 15

    dataset = Dataset()
    cnn = Model(input_shape, num_classes, cnn_weights_path)

    for bounding_box in predictions:
        x1 = int(bounding_box['x1']) - DELTA
        y1 = int(bounding_box['y1']) - DELTA
        x2 = int(bounding_box['x2']) + DELTA
        y2 = int(bounding_box['y2']) + DELTA

        traffic_sign = image[y1:y2, x1:x2]
        processed_image = dataset._preprocess_image(
            traffic_sign, centered=True
        )

        cnn_input = np.expand_dims(processed_image, axis=0)

        label = cnn.predict(cnn_input)
        draw_rectangle(image, (x1, y1, x2, y2), label)

    plt.imshow(image)
    plt.show()
