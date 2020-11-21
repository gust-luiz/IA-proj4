from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from numpy import reshape
from numpy.random import permutation

import variabels
from variabels import NUM_CLASSES, NUM_CHANNELS

# Based on: https://github.com/mjbhobe/dl-tensorflow-keras/blob/master/MNIST%20-%20Multiclass%20Classification%20-%20CNN%20-%20Keras.ipynb


def get_data(to_normalize=False):
    (train_digits, train_labels), (test_digits, test_labels) = load_data()

    # converting 60,000 training digits and 10,000 test digits, on shape[0],1
    # from `heigh`x`width` to `heigh`x`width`x1 data
    variabels.height, variabels.width = train_digits.shape[1:]

    train_data = reshape(train_digits, (train_digits.shape[0], variabels.height, variabels.width, NUM_CHANNELS))
    test_data = reshape(test_digits, (test_digits.shape[0], variabels.height, variabels.width, NUM_CHANNELS))

    # converting pixel value range from 0-255 to 0-1
    if to_normalize:
        train_data = train_data.astype('float32') / 255.
        test_data = test_data.astype('float32') / 255.

    # converting labels to one-hot code, e.g: 4 = [0 0 0 0 1 0 0 ...]
    train_labels = to_categorical(train_labels, NUM_CLASSES)
    test_labels = to_categorical(test_labels, NUM_CLASSES)

    return (train_data, train_labels), (test_data, test_labels)


def shuffle_data(data, labels, cnt=1):
    if cnt < 1:
        return data, labels

    for _ in range(cnt):
        indexes = permutation(len(data))

    data = data[indexes]
    labels = labels[indexes]

    return data, labels


def split_data(data, labels, percent=0.5):
    return (data, labels), (None, None)
