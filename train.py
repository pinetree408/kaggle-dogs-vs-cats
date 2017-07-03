import tensorflow as tf
import time
import model
import os
import random
from glob import glob
from skimage import io
from scipy.misc import imresize
import numpy as np

if __name__ == "__main__":

    tf.reset_default_graph()

    seed = int(time.time())
    tf.set_random_seed(seed)

    files_path = 'train/'

    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')

    cat_files = sorted(glob(cat_files_path))[:500]
    dog_files = sorted(glob(dog_files_path))[:500]

    n_files = len(cat_files) + len(dog_files)

    size_image = 64

    allX = np.zeros((n_files, size_image * size_image), dtype='float64')
    ally = np.zeros((n_files, 2), dtype='float64')

    count = 0
    for f in cat_files:
        img = io.imread(f, as_grey=True)
        new_img = imresize(img, (size_image, size_image))
        allX[count] = np.array(new_img).reshape(size_image * size_image)
        ally[count] = np.array([0.0, 1.0])
        count += 1

    for f in dog_files:
        img = io.imread(f, as_grey=True)
        new_img = imresize(img, (size_image, size_image))
        allX[count] = np.array(new_img).reshape(size_image * size_image)
        ally[count] = np.array([1.0, 0.0])
        count += 1

    index_shuf = range(n_files)
    random.shuffle(index_shuf)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for index, i in enumerate(index_shuf):
        if index < (n_files * 0.9):
            X_train.append(allX[i])
            Y_train.append(ally[i])
        else:
            X_test.append(allX[i])
            Y_test.append(ally[i])

    image_size = 64

    x = tf.placeholder(tf.float32, [None, image_size * image_size])
    x_image = tf.reshape(x, [-1, image_size, image_size, 1])

    keep_prob = tf.placeholder(tf.float32)

    k = model.inference(x_image, keep_prob)
    p = tf.nn.softmax(k)

    t = tf.placeholder(tf.float32, [None, 2])

    with tf.name_scope('train') as scope: 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k,labels=t)) 
        loss_summary = tf.summary.scalar('cost', loss)
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    with tf.name_scope('test') as scope:
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    log_dir = "log/"
    test_log_dir = log_dir + "test"
    train_log_dir = log_dir + "train"

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        # start training
        for i in range(100):
            print i
            batch_xs = X_train[i*9:(i+1)*9] 
            batch_ts = Y_train[i*9:(i+1)*9]
            feed_train = {x: batch_xs, t: batch_ts, keep_prob: 0.5}
            train_result = sess.run([merged, train_step], feed_dict=feed_train)
            train_writer.add_summary(train_result[0], i)
            if i > 0 and i % 10 == 0: 
                feed_test = {x: X_test, t: Y_test, keep_prob: 1.0}
                test_result = sess.run([merged], feed_dict=feed_test)
                test_writer.add_summary(test_result[0], i)