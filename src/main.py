from cnn import get_1_layer_model, get_2_layers_model, load_model, save_model
from data_handler import get_data, shuffle_data, split_data
from performance import evaluate, show_plots, predictions
from utils import path_relative_to
from variabels import BATCH_SIZE, EPOCHS, TRAIN_PERC, models_dir

MODEL_FILE_NAME = 'modelo_2_camadas'

(train_data, train_labels), (test_data, test_labels) = get_data()
train_data, train_labels = shuffle_data(train_data, train_labels)

(train_data, train_labels), (xvalidate_data, xvalidate_labels) = split_data(train_data, train_labels, TRAIN_PERC)

model = None # load_model(MODEL_FILE_NAME, models_dir())

model = model or get_2_layers_model()
print('model', model.summary())

results = model.fit(
    train_data, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(xvalidate_data, xvalidate_labels)
)

# save_model(model, MODEL_FILE_NAME, models_dir())

print('history', results.history)

show_plots(results.history)

evaluate(model, test_data, test_labels)

predictions(model, test_data, test_labels)
