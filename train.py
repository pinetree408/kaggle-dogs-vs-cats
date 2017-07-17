import tensorflow as tf
import time
import model
from dataset import get_dataset

if __name__ == "__main__":

    tf.reset_default_graph()

    seed = int(time.time())
    tf.set_random_seed(seed)

    image_size = 128

    log_dir = "log/"
    test_log_dir = log_dir + "test"
    train_log_dir = log_dir + "train"

    learning_rate = 0.0001
    batch_size = 4
    epoches = 100

    x_train, x_test, y_train, y_test, n_files = get_dataset('train/', image_size)

    x = tf.placeholder(tf.float32, [None, image_size * image_size * 3])
    x_image = tf.reshape(x, [-1, image_size, image_size, 3])

    keep_prob = tf.placeholder(tf.float32)

    k = model.inference(x_image, keep_prob, image_size)
    p = tf.nn.softmax(k)

    t = tf.placeholder(tf.float32, [None, 2])

    with tf.name_scope('train') as scope: 
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k,labels=t)) 
        loss_summary = tf.summary.scalar('loss', cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope('test') as scope:
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        for j in range(epoches):
            print j
            for i in range(int((n_files * 0.9)/(batch_size * 9))):
                batch_x = x_train[i*9:(i+batch_size)*9] 
                batch_t = y_train[i*9:(i+batch_size)*9]
                feed_train = {
                        x: batch_x,
                        t: batch_t,
                        keep_prob: 0.5
                        }
                train_result = sess.run([merged, train_step], feed_dict=feed_train)

            feed_test = {
                    x: x_test,
                    t: y_test,
                    keep_prob: 1.0
                    }
            test_result = sess.run([merged], feed_dict=feed_test)
            
            train_writer.add_summary(train_result[0], j)
            test_writer.add_summary(test_result[0], j)
