import tensorflow as tf

def inference(image, keep_prob):

    # conv1
    num_filters1 = 32
    w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filters1], stddev=0.1))
    h_conv1 = tf.nn.conv2d(image, w_conv1, strides=[1,1,1,1], padding='SAME')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
    h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

    # pool1
    h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # conv2
    num_filters2 = 64
    w_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filters1,num_filters2], stddev=0.1))
    h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding='SAME')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
    h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

    # pool2
    h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # fully connected
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*num_filters2])
    
    num_units1 = 16*16*num_filters2
    num_units2 = 1024

    w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
    b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
    hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

    w0 = tf.Variable(tf.zeros([num_units2, 2]))
    b0 = tf.Variable(tf.zeros([2]))
    k = tf.matmul(hidden2_drop, w0) + b0

    return k

