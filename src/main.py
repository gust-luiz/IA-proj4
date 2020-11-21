from cnn import get_1layer_model
from data_handler import get_data, shuffle_data, split_data

(train_data, train_labels), (test_data, test_labels) = get_data()

train_data, train_labels = shuffle_data(train_data, train_labels)

(train_data, train_labels), (xvalidate_data, xvalidate_labels) = split_data(train_data, train_labels)

model = get_1layer_model()
