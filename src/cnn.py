from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json
from os.path import exists, join
from os import mkdir

import variabels
from variabels import NUM_CHANNELS, NUM_CLASSES


def get_1_layer_model():
    model = Sequential()

    # ...

    return model


def get_2_layers_model():
    filters_cnt = 32
    kernel_size = 5
    # 'valid' | 'same'
    padding = 'same'
    # 'relu' | 'sigmoid' | 'softmax' | 'softplus' |
    # 'softsign' | 'tanh' | 'selu' | 'elu' | 'exponential'
    # https://keras.io/api/layers/activations/
    activation = 'relu'
    final_activation = 'softmax'

    pool_size = 2
    dense_size = 200

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
