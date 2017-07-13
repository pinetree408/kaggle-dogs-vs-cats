import os
import random
import numpy as np
from glob import glob
from skimage import io
from scipy.misc import imresize

def get_dataset(files_path, image_size):

    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')

    cat_files = sorted(glob(cat_files_path))[:4000]
    dog_files = sorted(glob(dog_files_path))[:4000]

    n_files = len(cat_files) + len(dog_files)

    all_x = np.zeros((n_files, image_size * image_size), dtype='float64')
    all_y = np.zeros((n_files, 2), dtype='float64')

    count = 0
    for f in cat_files:
        img = io.imread(f, as_grey=True)
        new_img = imresize(img, (image_size, image_size))
        all_x[count] = np.array(new_img).reshape(image_size * image_size)
        all_y[count] = np.array([0.0, 1.0])
        count += 1

    for f in dog_files:
        img = io.imread(f, as_grey=True)
        new_img = imresize(img, (image_size, image_size))
        all_x[count] = np.array(new_img).reshape(image_size * image_size)
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
