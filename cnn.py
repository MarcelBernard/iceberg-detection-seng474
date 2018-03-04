'''
shape input layers properly (75 x 75 x 3)
try different numbers of layers and parameters to find out what is good
dense layers, then sigmoid function
'''
import json
import os
import numpy as np
from data_preprocessing import *

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import np_utils

'''
    Class containing methods for generating and fitting a model for the iceberg dataset.
'''
class Model:
    '''
        Generates and compiles a sequential model with 3 convolutional 2D layers, 
        1 dense layer and a sigmoid function.
    '''
    def get_model():
        # Sequential model
        model = Sequential()
        # Conv2D input layer
        model.add(Convolution2D(16, (3, 3), activation = 'relu', input_shape = (75, 75, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        # Conv2D layer 2
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        # Conv2D layer 3
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        # Flatten data fro dense layers
        model.add(Flatten())
        # Dense layer
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        # Dense layer 2/sigmoid boi
        model.add(Dense(10, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    '''
        Fits the model generated by get_model() to the processed data.
        Takes the model, processed x and processed y training data as arguments.
    '''
    def fit_model(model, X_train, y_train):
        # Fit model
        model.fit(X_train, y_train,
                batch_size = 24,
                epochs = 5,
                verbose = 1
                # verbose = 1,
                # validation_data = (X_valid, y_valid),
                # callbacks = callbacks
                )

    '''
        Processes data to use in the model.
        Takes the preprocessed X and y training data as arguments and returns X_train, X_valid, 
        y_train, y_valid to be used with the model.
    '''
    def process_data(X, y):
        # Split the training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)
        # preprocess labels for y_train and y_valid
        y_train = np_utils.to_categorical(y_train, 10)
        y_valid = np_utils.to_categorical(y_valid, 10)
        return X_train, X_valid, y_train, y_valid

    if __name__ == '__main__':
        # X, y, train = get_input_data(train_file_path='train.json')
        X, y = get_input_data(train_file_path='train.json')
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        # train = optimizer.minimize(loss)
        # target_train=train['is_iceberg']
        # X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)
        X_train, X_valid, y_train, y_valid = process_data(X, y)
        '''
        X_train_cv = X_train.astype('float32')
        X_train_cv /= 255
        '''
        # y_train = np_utils.to_categorical(y_train, 10)
        # print("Shape of training data: {}".format(X_train.shape))
        # print("Shape of training data: {}".format(Y_train.shape))
        model = get_model()
        fit_model(model, X_train, y_train)
