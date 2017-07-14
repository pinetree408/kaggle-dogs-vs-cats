import os
import random
import numpy as np
from glob import glob
from skimage import io
from scipy.misc import imresize

def get_dataset(files_path, image_size):

    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')

    cat_files = sorted(glob(cat_files_path))[:10000]
    dog_files = sorted(glob(dog_files_path))[:10000]

    n_files = len(cat_files) + len(dog_files)

    image_flatten = image_size * image_size * 3

    all_x = np.zeros((n_files, image_flatten), dtype='float64')
    all_y = np.zeros((n_files, 2), dtype='float64')

    count = 0
    for f in cat_files:
        img = io.imread(f)
        new_img = np.zeros((image_size, image_size, 3))
        for i in range(3):
            new_img[:, :, i] = imresize(img[:, :, i], (image_size, image_size))
        all_x[count] = new_img.reshape(image_flatten)
        all_y[count] = np.array([0.0, 1.0])
        count += 1

    for f in dog_files:
        img = io.imread(f)
        new_img = np.zeros((image_size, image_size, 3))
        for i in range(3):
            new_img[:, :, i] = imresize(img[:, :, i], (image_size, image_size))
        all_x[count] = new_img.reshape(image_flatten)
        all_y[count] = np.array([1.0, 0.0])
        count += 1

    index_shuf = range(n_files)
    random.shuffle(index_shuf)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for index, i in enumerate(index_shuf):
        if index < int(n_files * 0.9):
            x_train.append(all_x[i])
            y_train.append(all_y[i])
        else:
            x_test.append(all_x[i])
            y_test.append(all_y[i])
    return x_train, x_test, y_train, y_test, n_files
