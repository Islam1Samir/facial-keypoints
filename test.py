import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def predictKeypoints(img,weights = 'facial_point.ckpt'):

    inputs = tf.placeholder(tf.float32,(None,224,224,1),name = 'inputs')

    conv1 = tf.layers.conv2d(inputs, 32, 5, padding='same')
    relu1 = tf.nn.relu(conv1)
    maxpool1 = tf.layers.max_pooling2d(relu1, (2,2), (2,2), padding='same')
    drop1 = tf.layers.dropout(maxpool1,0.1)

    conv2 = tf.layers.conv2d(drop1, 64, 3, padding='same')
    relu2 = tf.nn.relu(conv2)
    maxpool2 = tf.layers.max_pooling2d(relu2, (2,2), (2,2), padding='same')
    drop2 = tf.layers.dropout(maxpool2,0.2)


    conv3 = tf.layers.conv2d(drop2, 128, 3, padding='same')
    relu3 = tf.nn.relu(conv3)
    maxpool3 = tf.layers.max_pooling2d(relu3, (2,2), (2,2), padding='same')
    drop3 = tf.layers.dropout(maxpool3,0.3)

    conv4 = tf.layers.conv2d(drop3, 128, 3, padding='same')
    relu4 = tf.nn.relu(conv4)
    maxpool4 = tf.layers.max_pooling2d(relu4, (2,2), (2,2), padding='same')
    drop4 = tf.layers.dropout(maxpool4,0.3)


    flat = tf.layers.flatten(drop4)
    dense1 = tf.layers.dense(flat,1000)
    relu5 = tf.nn.relu(dense1)
    logits = tf.layers.dense(relu5,136)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)

      loader = tf.train.import_meta_graph(weights+'.meta')
      loader.restore(sess, weights)

      img = img/255.0

      predict = sess.run(logits, feed_dict={inputs:img })

      predicted_key_pts = predict[0]
      predicted_key_pts = np.array(predicted_key_pts)

      predicted_key_pts = predicted_key_pts * 50.0 + 100

      predicted_key_pts = predicted_key_pts.reshape(-1, 2)
      return predicted_key_pts

