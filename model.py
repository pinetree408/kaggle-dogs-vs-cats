import tensorflow as tf
from helper import conv_layer, fc_layer

def inference(image, keep_prob, image_size):

    # conv1
    num_filters1 = 32
    layer_conv1, weights_conv1 = conv_layer(image, 3, 3, num_filters1)
    # conv2
    num_filters2 = 32
    layer_conv2, weights_conv2 = conv_layer(layer_conv1, num_filters1, 3, num_filters2)

    # conv3
    num_filters3 = 64
    layer_conv3, weights_conv3 = conv_layer(layer_conv2, num_filters2, 3, num_filters3)

    # flatten
    num_units1 = (image_size/8) * (image_size/8) * num_filters3
    layer_flat = tf.reshape(layer_conv3, [-1, num_units1])
    
    # fc1
    num_units2 = 128
    layer_fc1 = fc_layer(layer_flat, num_units1, num_units2, True)
    
    layer_drop = tf.nn.dropout(layer_fc1, keep_prob)

    # fc2
    layer_fc2 = fc_layer(layer_drop, num_units2, 2, False)

    return layer_fc2
