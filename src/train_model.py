from dataset import Dataset, IMG_SIZE, CATEGORIES
from model import Model

input_shape = (IMG_SIZE, IMG_SIZE, 3)
num_classes = len(CATEGORIES)

dataset = Dataset(augmented=True)
x_train, y_train, x_val, y_val, x_test, y_test = dataset.load_data()

model = Model(input_shape, num_classes)
model.train(x_train, y_train, x_val, y_val)
model.evaluate(x_test, y_test)
