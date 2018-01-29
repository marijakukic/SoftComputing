import glob
import json
import keras
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread
from skimage import color, exposure, transform
from sklearn.model_selection import train_test_split

CATEGORIES = [
    'danger',
    'info',
    'mandatory',
    'priority'
]

IMG_SIZE = 48


class Dataset(object):

    def __init__(self,
                 train_path='train/',
                 test_path='test/',
                 labels_map_path='labels_map.json',
                 augmented=True):

        self._train_path = './dataset/' + train_path
        self._test_path = './dataset/' + test_path
        self._labels_map = json.load(open('./dataset/' + labels_map_path))

        self._augmented = augmented

        self._train_samples = []
        self._train_labels = []
        self._test_samples = []
        self._test_labels = []

        self._datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            shear_range=0.1
        )

    def load_data(self, split_size=0.2):
        print("\n=> Loading dataset ...")

        self._load_train_data()
        self._load_test_data()

        if self._augmented:
            print("=> Augmenting dataset ...")
            self._augment_dataset()

        print("\n----------------")
        for i in range(4):
            print(
                "{0} - {1}".format(CATEGORIES[i], self._train_labels.count(i))
            )
        print("----------------\n")

        x_train, x_val, y_train, y_val = train_test_split(
            np.array(self._train_samples, dtype='float32'),
            np.array(self._train_labels),
            test_size=split_size,
            random_state=42
        )

        x_test = np.array(self._test_samples, dtype='float32')
        y_test = keras.utils.to_categorical(self._test_labels, len(CATEGORIES))
        y_train = keras.utils.to_categorical(y_train, len(CATEGORIES))
        y_val = keras.utils.to_categorical(y_val, len(CATEGORIES))

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _load_train_data(self):
        for index, category in enumerate(CATEGORIES):
            for folder in self._labels_map[category]:
                images = self._load_images(self._train_path + folder)
                self._train_samples.extend(images)
                self._train_labels.extend([int(index)] * len(images))

    def _load_test_data(self):
        test_metadata = pd.read_csv(
            self._test_path + 'metadata.csv', usecols=[0, 7])

        for _, image_path, class_id in test_metadata.itertuples():
            image = self._load_image(self._test_path + image_path)
            self._test_samples.append(image)
            self._test_labels.append(self._get_label_from_class_id(class_id))

    def _get_label_from_class_id(self, class_id):
        for index, key in enumerate(self._labels_map):
            for value in self._labels_map[key]:
                if class_id == int(value):
                    return index

    def _load_images(self, path):
        images = []
        image_paths = glob.glob(path + '/*.ppm')

        for image_path in image_paths:
            images.append(self._load_image(image_path))

        return images

    def _load_image(self, image_path):
        image = imread(image_path)
        return self._preprocess_image(image)

    def _preprocess_image(self, image, centered=True):
        hsv = color.rgb2hsv(image)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        if centered:
            min_side = min(img.shape[:-1])
            centre = img.shape[0] // 2, img.shape[1] // 2
            img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
                      centre[1] - min_side // 2: centre[1] + min_side // 2,
                      :]

        img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

        return img

    def _augment_dataset(self):
        augmentation_sizes = [(5200, 2), (3000, 6), (5000, 3)]
        aug_images = []
        aug_labels = []

        for label, (n_samples, multiplier) in enumerate(augmentation_sizes):
            indices = np.nonzero(np.array(self._train_labels) == label)[0]
            indices = indices[: n_samples]

            for base in np.array(self._train_samples)[indices]:
                images = self._generate_images(base, multiplier)
                aug_images.extend(images)
                aug_labels.extend([label] * len(images))

        self._train_samples.extend(aug_images)
        self._train_labels.extend(aug_labels)

    def _generate_images(self, base, count):
        base = base.reshape((1,) + base.shape)

        images = []
        for i in range(count):
            gen_image = self._datagen.flow(base, batch_size=1).next()
            image = gen_image.squeeze(axis=0)
            images.append(image)

        return images
