"""Module for preprocessing testing and training data into a format usable by the classifier."""

import json

import numpy as np


def get_input_data(train_file_path='train.json'):
    """Retrieves training (X) and label (y) matrices. Note that this can take a few seconds to run.

    Args:
        train_file_path is the path of the file containing training data.

    Returns:
        A tuple containing the X training matrix in the first position, and the y label matrix in the second position.
        X is of shape (N, 75, 75, 3), where N is the number of training images, 75 x 75 is the dimension of the images,
        and 3 represents the number of channels for each image.
    """

    with open(train_file_path, 'r') as train_file:
        json_data = train_file.read()
        train_data = json.loads(json_data)

    band_1 = [instance['band_1'] for instance in train_data]
    band_2 = [instance['band_2'] for instance in train_data]

    band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_1])
    band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_2])

    # Combine all three channels into an array of 1604 tensors (number of training images) with dimension 75 x 75 x 3
    X_train = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis]], axis=-1)

    # True labels of data, either iceberg or not iceberg
    y_train = np.array([instance['is_iceberg'] for instance in train_data])

    return X_train, y_train


def get_rotated_images(X, y):
    """Rotates input data in order to generate more training instances.

    Returns:
        Similar to get_input_data, returns an X tensor of training images and y matrix of training labels. These
        matrices should be appended to the existing X and y training matrices prior to training.
    """

    X_rotated = []
    y_rotated = []

    # Generate three 90 degree rotations for each input image
    for instance, label in zip(X, y):
        rotate_90 = np.rot90(instance, axes=(0, 1))
        rotate_180 = np.rot90(rotate_90, axes=(0, 1))
        rotate_270 = np.rot90(rotate_180, axes=(0, 1))

        X_rotated.extend([rotate_90, rotate_180, rotate_270])
        y_rotated.extend(3 * [label])

    X_rotated = np.array(X_rotated)
    y_rotated = np.array(y_rotated)

    return X_rotated, y_rotated


if __name__ == '__main__':
    # Example usage
    X_train, y_train = get_input_data(train_file_path='train.json')
    print('Shape of training data is {}'.format(X_train.shape))
    print('Shape of training labels is {}'.format(y_train.shape))

    X_rotated, y_rotated = get_rotated_images(X_train, y_train)
    X_train = np.concatenate([X_train, X_rotated])
    y_train = np.concatenate([y_train, y_rotated])

    print(X_train.shape)
    print(y_train.shape)
