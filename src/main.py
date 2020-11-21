from cnn import get_1_layer_model
from data_handler import get_data, shuffle_data, split_data
from variabels import BATCH_SIZE, EPOCHS
from performance import show_plots

(train_data, train_labels), (test_data, test_labels) = get_data()

train_data, train_labels = shuffle_data(train_data, train_labels)

(train_data, train_labels), (xvalidate_data, xvalidate_labels) = split_data(train_data, train_labels)

model = get_1_layer_model()

results = model.fit(
    train_data, train_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(test_data, test_labels)
)

show_plots(results.history)
