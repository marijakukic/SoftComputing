import os
import sys
import cv2
import json
import numpy as np

from argparse import ArgumentParser
from skimage.transform import resize
from src.dataset import Dataset
from src.model import Model

sys.path.append(os.path.join(os.path.dirname(__file__), "tensorbox"))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tensorbox.predict import initialize, hot_predict

NEW_WIDTH = 640
NEW_HEIGHT = 480


def initialize_detector():
    hypes_path = 'tensorbox/output/overfeat_rezoom_2018_01_29_16.37/hypes.json'
    config = json.load(open(hypes_path, 'r'))

    weights_path = os.path.join(os.path.dirname(
        hypes_path), config['solver']['weights']
    )

    return initialize(weights_path, hypes_path)


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


def process_video(video_path):
    detector = initialize_detector()

    input_shape = (48, 48, 3)
    num_classes = 4
    cnn_weights_path = 'model/weights.h5'
    DELTA = 15

    dataset = Dataset()
    cnn = Model(input_shape, num_classes, cnn_weights_path)

    cap = cv2.VideoCapture(video_path)

    while (cap.isOpened()):
        _, frame = cap.read()

        frame = resize(frame, (NEW_HEIGHT, NEW_WIDTH), mode='constant')
        print(frame.shape)

        predictions = hot_predict('dummy', detector, image=frame)

        for bounding_box in predictions:
            x1 = int(bounding_box['x1']) - DELTA
            y1 = int(bounding_box['y1']) - DELTA
            x2 = int(bounding_box['x2']) + DELTA
            y2 = int(bounding_box['y2']) + DELTA

            traffic_sign = frame[y1:y2, x1:x2]
            processed_image = dataset._preprocess_image(
                traffic_sign, centered=True
            )

            cnn_input = np.expand_dims(processed_image, axis=0)

            label = cnn.predict(cnn_input)
            draw_rectangle(frame, (x1, y1, x2, y2), label)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser(usage='usage: python3 video.py --video <video>')
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()

    process_video(args.video)
