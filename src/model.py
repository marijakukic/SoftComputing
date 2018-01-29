import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from dataset import CATEGORIES


class Model(object):

    def __init__(self, input_shape, num_classes, weights_path=None):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=input_shape,
                         activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        if weights_path:
            model.load_weights(weights_path)

        self._model = model

    def train(self, x_train, y_train,
              x_val, y_val,
              batch_size=64,
              epochs=10,
              validation_split=0.2):

        checkpoint = ModelCheckpoint(
            'model/weights.h5',
            save_weights_only=True,
            save_best_only=True
        )

        self._model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=validation_split,
            callbacks=[checkpoint],
            validation_data=(x_val, y_val)
        )

    def predict(self, x):
        predictions = self._model.predict(x)
        index = np.argmax(predictions)
        return CATEGORIES[index]

    def evaluate(self, x_test, y_test):
        loss, accuracy = self._model.evaluate(x_test, y_test, batch_size=128)
        print('\nTest loss:', loss)
        print('Test accuracy:\n', accuracy)
