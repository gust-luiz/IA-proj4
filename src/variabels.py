NUM_CHANNELS = 1

NUM_CLASSES = 10

EPOCHS = 3

BATCH_SIZE = 64

TRAIN_PERC = .9

height = 0

width = 0

def models_dir():
    from utils import path_relative_to

    return path_relative_to(__file__, '../models')
