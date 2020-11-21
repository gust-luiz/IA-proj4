from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from numpy import reshape
from numpy.random import permutation


def get_data(to_normalize=False):
    (train_digits, train_labels), (test_digits, test_labels) = load_data()

    # converting 60,000 training digits and 10,000 test digits, on shape[0],1
    # from `heigh`x`width` to `heigh`x`width`x1 data
    height, width = train_digits.shape[1:]
    num_channels = 1

    train_data = reshape(train_digits, (train_digits.shape[0], height, width, num_channels))
    test_data = reshape(test_digits, (test_digits.shape[0], height, width, num_channels))

    # converting pixel value range from 0-255 to 0-1
    if to_normalize:
        train_data = train_data.astype('float32') / 255.
        test_data = test_data.astype('float32') / 255.

    # converting labels to one-hot code, e.g: 4 = [0 0 0 0 1 0 0 ...]
    num_classes = 10

    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

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
