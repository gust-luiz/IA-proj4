import matplotlib.pyplot as pyplot
from numpy import argmax

from variabels import BATCH_SIZE


def show_plots(history, plot_title=None, fig_size=None):
    """ Useful function to view plot of loss values & accuracies across the various epochs
        Works with the history object returned by the train_model(...) call

        Based on: https://github.com/mjbhobe/dl-tensorflow-keras/blob/master/kr_helper_funcs.py
    """
    assert type(history) is dict

    # NOTE: the history object should always have loss & acc (for training data), but MAY have
    # val_loss & val_acc for validation data
    loss_vals = history['loss']
    val_loss_vals = history.get('val_loss')
    epochs = range(1, len(history['accuracy']) + 1)

    _, ax = pyplot.subplots(nrows=1, ncols=2, figsize=fig_size or (16, 4))

    # plot losses on ax[0]
    ax[0].plot(epochs, loss_vals, color='navy', marker='o', linestyle=' ', label='Treinamento')

    if val_loss_vals:
        ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validação')
        ax[0].set_title('Taxa de Erros em Treinamento e Validação')
        ax[0].legend(loc='best')
    else:
        ax[0].set_title('Taxa de Erros em Treinamento')

    ax[0].set_xlabel('Gerações')
    ax[0].set_ylabel('Taxa de Erros')
    ax[0].grid(True)

    # plot accuracies
    acc_vals = history['accuracy']
    val_acc_vals = history.get('val_accuracy')

    ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Treinamento')

    if val_acc_vals:
        ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validação')
        ax[1].set_title('Taxa de Acurácia em Treinamento e Validação')
        ax[1].legend(loc='best')
    else:
        ax[1].set_title('Taxa de Acurácia em Treinamento')

    ax[1].set_xlabel('Gerações')
    ax[1].set_ylabel('Taxa de Acurácia')
    ax[1].grid(True)

    if plot_title is not None:
        pyplot.suptitle(plot_title)

    pyplot.show()
    pyplot.close()

    # delete locals from heap before exiting (to save some memory!)
    del loss_vals, epochs, acc_vals

    if val_loss_vals:
        del val_loss_vals

    if val_acc_vals:
        del val_acc_vals


def evaluate(model, data, labels, file=None):
    loss, accuracy = model.evaluate(data, labels, batch_size=BATCH_SIZE)

    if not file:
        print(f'Taxa de Erros: {loss:.4f}')
        print(f'Taxa de Acurácia: {accuracy:.4f}')
    else:
        print(f'{loss:.4f};{accuracy:.4f}', end=';', file=file)


def predictions(model, data, labels, file=None):
    pred = argmax(model.predict(data), axis=1)
    true = argmax(labels, axis=1)

    if not file:
        print(f'Previsões erradas: {(pred != true).sum()}/{len(labels)}')
    else:
        print(f'{(pred != true).sum()}', end=';', file=file)
