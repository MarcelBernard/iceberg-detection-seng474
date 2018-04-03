import json
import os
from data_preprocessing import get_input_data, get_rotated_images
import csv
import numpy as np
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(420)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation, BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import np_utils


def get_model(learning_rate, dropout):
    """ Generates and compiles a sequential model with 4 convolutional 2D layers,
    2 dense layer and a sigmoid function. Initial implementation inspired by
    https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516"""
    # Sequential model
    model = Sequential()
    # Conv2D input layer
    model.add(Convolution2D(64, (3, 3), strides = (1,1), activation = 'relu', input_shape = (75, 75, 2)))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    model.add(Dropout(dropout))
    # Conv2D layer 2
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # Conv2D layer 3
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # Conv2D layer 4
    model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # Flatten data fro dense layers
    model.add(Flatten())
    # Dense layer 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout))
    # Dense layer 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    # Dense layer 3/sigmoid boi
    model.add(Dense(1, activation='sigmoid'))
    # Compile model

    optimizer = optimizers.Adam(lr=learning_rate, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    X, y = get_input_data(train_file_path='train.json')

    # Incorporate rotated images into training data (note that this significantly increases training time)
    X_rotated, y_rotated = get_rotated_images(X, y)
    X = np.concatenate([X, X_rotated])
    y = np.concatenate([y, y_rotated])

    model = get_model(learning_rate=0.001, dropout=0.2)

    accuracies = []
    losses = []
    precision_scores = []
    recall_scores = []

    validation_accuracies = []
    validation_losses = []
    training_accuracies = []
    training_losses = []

    # Perform 10-fold cross validation
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    kfold_count = 0
    for train_index, test_index in kfolds.split(X, y):
        print('STARTING KFOLD {}'.format(kfold_count))
        print()
        kfold_count += 1

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Train and test model
        train_history = model.fit(X_train, y_train, epochs=15, verbose=1, batch_size=32, validation_data=(X_test, y_test))
        y_predictions = model.predict(X_test)

        # Convert predictions to binary
        threshold = 0.5
        threshold_predictions = []
        for prediction in y_predictions:
            if prediction > threshold:
                threshold_predictions.append(1)
            else:
                threshold_predictions.append(0)

        # Evaluate metrics
        accuracies.append(accuracy_score(y_true=y_test, y_pred=threshold_predictions))
        losses.append(model.evaluate(X_test, y_test)[0])
        precision_scores.append(precision_score(y_true=y_test, y_pred=threshold_predictions))
        recall_scores.append(recall_score(y_true=y_test, y_pred=threshold_predictions))

        print('Accuracy: {}'.format(accuracies[-1]))
        print('Loss: {}'.format(losses[-1]))
        print('Precision: {}'.format(precision_scores[-1]))
        print('Recall: {}'.format(recall_scores[-1]))
        print()

        validation_losses.append(train_history.history['val_loss'])
        validation_accuracies.append(train_history.history['val_acc'])
        print('Validation accuracy: {}'.format(train_history.history['val_acc']))
        print('Validation loss: {}'.format(train_history.history['val_loss']))

        training_losses.append(train_history.history['loss'])
        training_accuracies.append(train_history.history['acc'])
        print('Training accuracy: {}'.format(train_history.history['acc']))
        print('Training loss: {}'.format(train_history.history['loss']))

    avg_validation_loss = []
    for ix in range(15):
        loss_sum = 0
        avg_validation_loss.append(0)
        for losses in validation_losses:
            loss_sum += losses[ix]

        avg_validation_loss[ix] = loss_sum / 10

    avg_validation_accuracy = []
    for ix in range(15):
        acc_sum = 0
        avg_validation_accuracy.append(0)
        for acc in validation_accuracies:
            acc_sum += acc[ix]

        avg_validation_accuracy[ix] = acc_sum / 10

    avg_train_loss = []
    for ix in range(15):
        loss_sum = 0
        avg_train_loss.append(0)
        for losses in training_losses:
            loss_sum += losses[ix]

        avg_train_loss[ix] += loss_sum / 10

    avg_train_accuracy = []
    for ix in range(15):
        acc_sum = 0
        avg_train_accuracy.append(0)
        for acc in training_accuracies:
            acc_sum += acc[ix]

        avg_train_accuracy[ix] = acc_sum / 10

    print()
    print('Average validation accuracy: {}'.format(avg_validation_accuracy))
    print('Average training accuracy: {}'.format(avg_train_accuracy))
    print('Average validation loss: {}'.format(avg_validation_loss))
    print('Average training loss: {}'.format(avg_train_loss))
    print()
    print('Average accuracy: {}'.format(np.mean(accuracies)))
    print('Average log loss: {}'.format(np.mean(losses)))
    print('Average precision: {}'.format(np.mean(precision_scores)))
    print('Average recall: {}'.format(np.mean(recall_scores)))

    with open('overfitting_results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['validation accuracy'] + avg_validation_accuracy)
        writer.writerow(['training accuracy'] + avg_train_accuracy)
        writer.writerow(['validation loss'] + avg_validation_loss)
        writer.writerow(['training loss'] + avg_train_loss)
