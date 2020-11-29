from os import mkdir
from os.path import exists, join

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json

import variables
from variables import (ACTIVATION, DENSE_SIZE, FILTER_CNT, FINAL_ACTIVATION,
                       KERNEL_SIZE, NEXT_LAYER_FILTER_PROP, NUM_CHANNELS,
                       NUM_CLASSES, OPTIMIZER, PADDING, POOL_SIZE)


def get_1_layer_model(
        filters_cnt=FILTER_CNT,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION,
        dense_size=DENSE_SIZE,
        optimizer=OPTIMIZER
    ):
    model = Sequential()

    # First and only Convolution layer
    model.add(Conv2D(
        filters=filters_cnt,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation,
        input_shape=(variables.height, variables.width, NUM_CHANNELS)
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Flatten())

    # Dense layer
    model.add(Dense(
        dense_size,
        activation=activation
    ))

    # Output layer
    model.add(Dense(
        NUM_CLASSES,
        activation=FINAL_ACTIVATION
    ))

    # Compiling model
    model.compile(
        optimizer=optimizer,
        # 'categorical_crossentropy' | 'sparse_categorical_crossentropy'
        # https://keras.io/api/losses/probabilistic_losses/
        loss='categorical_crossentropy',
        # https://keras.io/api/metrics/
        metrics=['accuracy']
    )

    return model


def get_2_layers_model(
        filters_cnt=FILTER_CNT,
        next_layer_filter_prop=NEXT_LAYER_FILTER_PROP,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION,
        dense_size=DENSE_SIZE,
        optimizer=OPTIMIZER
    ):
    model = Sequential()

    model.add(Conv2D(
        filters=filters_cnt,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation,
        input_shape=(variables.height, variables.width, NUM_CHANNELS)
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Flatten())

    model.add(Dense(
        dense_size,
        activation=activation
    ))

    model.add(Dense(
        NUM_CLASSES,
        activation=FINAL_ACTIVATION
    ))

    model.compile(
        optimizer=optimizer,
        # 'categorical_crossentropy' | 'sparse_categorical_crossentropy'
        # https://keras.io/api/losses/probabilistic_losses/
        loss='categorical_crossentropy',
        # https://keras.io/api/metrics/
        metrics=['accuracy']
    )

    return model


def get_3_layers_model(
        filters_cnt=FILTER_CNT,
        next_layer_filter_prop=NEXT_LAYER_FILTER_PROP,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION,
        dense_size=DENSE_SIZE,
        optimizer=OPTIMIZER
    ):
    model = Sequential()

    # First Convolution layer
    model.add(Conv2D(
        filters=filters_cnt,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation,
        input_shape=(variables.height, variables.width, NUM_CHANNELS)
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    # Second Convolution layer
    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    # Third Convolution layer
    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Flatten())

    # Dense layer
    model.add(Dense(
        dense_size,
        activation=activation
    ))

    # Output layer
    model.add(Dense(
        NUM_CLASSES,
        activation=FINAL_ACTIVATION
    ))

    # Compiling model
    model.compile(
        optimizer=optimizer,
        # 'categorical_crossentropy' | 'sparse_categorical_crossentropy'
        # https://keras.io/api/losses/probabilistic_losses/
        loss='categorical_crossentropy',
        # https://keras.io/api/metrics/
        metrics=['accuracy']
    )

    return model


def get_4_layers_model(
        filters_cnt=FILTER_CNT,
        next_layer_filter_prop=NEXT_LAYER_FILTER_PROP,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION,
        dense_size=DENSE_SIZE,
        optimizer=OPTIMIZER
    ):
    model = Sequential()

    model.add(Conv2D(
        filters=filters_cnt,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation,
        input_shape=(variables.height, variables.width, NUM_CHANNELS)
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Conv2D(
        filters=filters_cnt * next_layer_filter_prop,
        kernel_size=(kernel_size, kernel_size), padding=padding,
        activation=activation
    ))
    model.add(MaxPooling2D(
        pool_size=(POOL_SIZE, POOL_SIZE)
    ))

    model.add(Flatten())

    model.add(Dense(
        dense_size,
        activation=activation
    ))

    model.add(Dense(
        NUM_CLASSES,
        activation=FINAL_ACTIVATION
    ))

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Based on: https://github.com/mjbhobe/dl-tensorflow-keras/blob/master/kr_helper_funcs.py
def save_model(model, file_name, save_dir):
    """ save the model structure to JSON & weights to HD5 """
    # check if save_dir exists, else create it
    if not exists(save_dir):
        try:
            mkdir(save_dir)
        except OSError as err:
            print(f'Não foi possível criar o repositório "{save_dir}", para salvar o modelo. Terminando a execução!')
            raise err

    # model structure is saved to $(save_dir)/base_file_name.json
    # weights are saved to $(save_dir)/base_file_name.h5
    model_json = model.to_json()
    json_file_path = join(save_dir, (file_name + ".json"))
    h5_file_path = join(save_dir, (file_name + ".h5"))

    with open(json_file_path, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5\n",
    model.save_weights(h5_file_path)

    print(f'Modelo salvo nos arquivos: "{json_file_path}" e "{h5_file_path}" ')


def load_model(file_name, load_dir):
    """ loads model structure & weights from previously saved state """
    # model structure is loaded $(load_dir)/base_file_name.json
    # weights are loaded from $(load_dir)/base_file_name.h5

    # load model from save_path
    loaded_model = None
    json_file_path = join(load_dir, (file_name + ".json"))
    h5_file_path = join(load_dir, (file_name + ".h5"))

    if exists(json_file_path) and exists(h5_file_path):
        with open(json_file_path, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(h5_file_path)

        print(f'Modelo construído a partir dos arquivos: "{json_file_path}" e "{h5_file_path}"')

    else:
        print(
            f'Arquivos não encontrados: "{(file_name + ".json")}" e "{(file_name + ".h5")}", na pasta "{load_dir}"'
        )

    return loaded_model
