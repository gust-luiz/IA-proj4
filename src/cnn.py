from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

import .variabels
from .variabels import NUM_CHANNELS, NUM_CLASSES


def get_1_layer_model():
    model = Sequential()

    # ...

    return model


def get_2_layers_model():
    filters_cnt = 32
    kernel_size = 3
    # 'valid' | 'same'
    padding = 'valid'
    # 'relu' | 'sigmoid' | 'softmax' | 'softplus' |
    # 'softsign' | 'tanh' | 'selu' | 'elu' | 'exponential'
    # https://keras.io/api/layers/activations/
    activation = 'relu'
    final_activation = 'softmax'

    pool_size = 2
    dense_size = 128

    model = Sequential()

    model.add(Conv2D(
        filters=filters_cnt,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation,
        input_shape=(variabels.height, variabels.width, NUM_CHANNELS)
    ))
    model.add(MaxPooling2D(
        pool_size=(pool_size, pool_size)
    ))

    model.add(Conv2D(
        filters=filters_cnt * 2,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(pool_size, pool_size)
    ))

    model.add(Flatten())

    model.add(Dense(
        dense_size,
        activation=activation
    ))

    model.add(Dense(
        NUM_CLASSES,
        activation=final_activation
    ))

    model.compile(
        # 'sgd' | 'rmsprop' | 'adam' | 'adadelta' | 'adagrad' | 'adamax' | 'nadam' | 'ftrl'
        # https://keras.io/api/optimizers/
        optimizer='adam',
        # 'categorical_crossentropy' | 'sparse_categorical_crossentropy'
        # https://keras.io/api/losses/probabilistic_losses/
        loss='categorical_crossentropy',
        # https://keras.io/api/metrics/
        metrics=['accuracy']
    )

    return model



def get_3_layers_model():
    model = Sequential()

    # ...

    return model



def get_4_layers_model():
    model = Sequential()

    # ...

    return model
