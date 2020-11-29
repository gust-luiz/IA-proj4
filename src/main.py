from cnn import (get_1_layer_model, get_2_layers_model, get_3_layers_model,
                 get_4_layers_model, load_model, save_model)
from data_handler import get_data, shuffle_data, split_data
from performance import evaluate, predictions, show_plots
from utils import path_relative_to
from variables import BATCH_SIZE, EPOCHS, TRAIN_PERC, models_dir

MODEL_FILE_NAME = 'cnn_4_layers'

(train_data, train_labels), (test_data, test_labels) = get_data(to_normalize=True)
train_data, train_labels = shuffle_data(train_data, train_labels)

(train_data, train_labels), (xvalidate_data, xvalidate_labels) = split_data(train_data, train_labels, TRAIN_PERC)

# model = load_model(MODEL_FILE_NAME, models_dir())
# model = model or get_2_layers_model()
model = get_4_layers_model()
print('model', model.summary())

results = model.fit(
    train_data, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(xvalidate_data, xvalidate_labels)
)

save_model(model, MODEL_FILE_NAME, models_dir())

print('history', results.history)

show_plots(results.history)

evaluate(model, test_data, test_labels)
predictions(model, test_data, test_labels)


# def test_model():
#     with open('batch_size.csv', 'w') as csv_file:
#         print('batch_size;tx_erro;tx_accu;predic', file=csv_file)

#         for batch_size in range(10, 120, 10):
#             print('testando:', batch_size)
#             print(batch_size, end=';', file=csv_file)

#             model = get_2_layers_model()

#             model.fit(
#                 train_data, train_labels,
#                 epochs=EPOCHS,
#                 batch_size=batch_size,
#                 validation_data=(xvalidate_data, xvalidate_labels)
#             )

#             evaluate(model, test_data, test_labels, csv_file)

#             predictions(model, test_data, test_labels, csv_file)

#             print()
#             print('', file=csv_file)

# test_model()
