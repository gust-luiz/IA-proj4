NUM_CHANNELS = 1

NUM_CLASSES = 10

EPOCHS = 15

BATCH_SIZE = 90

TRAIN_PERC = .9

FILTER_CNT = 27,

NEXT_LAYER_FILTER_PROP = 2.4

KERNEL_SIZE = 5

PADDING = 'same'

# 'relu' | 'sigmoid' | 'softmax' | 'softplus' |
# 'softsign' | 'tanh' | 'selu' | 'elu' | 'exponential'
# https://keras.io/api/layers/activations/
ACTIVATION = 'softplus'

DENSE_SIZE = 210
# 'sgd' | 'rmsprop' | 'adam' | 'adadelta' | 'adagrad' | 'adamax' | 'nadam' | 'ftrl'
# https://keras.io/api/optimizers/
OPTIMIZER = 'nadam'

POOL_SIZE = 2
FINAL_ACTIVATION  = 'softmax'


height = 0

width = 0

def models_dir():
    from utils import path_relative_to

    return path_relative_to(__file__, '../models')
