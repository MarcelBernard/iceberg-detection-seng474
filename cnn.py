import json
import os
import numpy as np
from data_preprocessing import get_input_data, get_rotated_images

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

    # Perform 10-fold cross validation
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    for train_index, test_index in kfolds.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Train and test model
        model.fit(X_train, y_train, epochs=10, verbose=1)
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

    print('Average accuracy: {}'.format(np.mean(accuracies)))
    print('Average log loss: {}'.format(np.mean(losses)))
    print('Average precision: {}'.format(np.mean(precision_scores)))
    print('Average recall: {}'.format(np.mean(recall_scores)))
