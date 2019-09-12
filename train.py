import pandas as pd
import cv2
import numpy as np
import os
import tensorflow as tf
from utils import rescale,random_crop




def get_batches(x, y, batch_size):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')
x_train = key_pts_frame.iloc[:, 0]
y_train = key_pts_frame.iloc[:, 1:].values
y_train = y_train.astype('float').reshape(y_train.shape[0], -1, 2)

arr = []
for i in range(len(x_train)):
    img = cv2.imread(os.path.join('data/training/', x_train[i]))
    img = img[:, :, 0:3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray, y_train[i] = rescale(gray, 250, y_train[i])
    gray, y_train[i] = random_crop(gray, 224, y_train[i])
    gray = gray / 255.0
    y_train[i] = (y_train[i] - 100) / 50
    gray = gray.reshape(224, 224, 1)
    arr.append(np.array(gray))

y_train = y_train.reshape(3462, -1)

gen = get_batches(arr, y_train, 64)

inputs = tf.placeholder(tf.float32, (None, 224, 224, 1), name='inputs')
target = tf.placeholder(tf.float32, (None, 136), name='targets')

conv1 = tf.layers.conv2d(inputs, 32, 5, padding='same')
relu1 = tf.nn.relu(conv1)
maxpool1 = tf.layers.max_pooling2d(relu1, (2, 2), (2, 2), padding='same')
drop1 = tf.layers.dropout(maxpool1, 0.1)

conv2 = tf.layers.conv2d(drop1, 64, 3, padding='same')
relu2 = tf.nn.relu(conv2)
maxpool2 = tf.layers.max_pooling2d(relu2, (2, 2), (2, 2), padding='same')
drop2 = tf.layers.dropout(maxpool2, 0.2)

conv3 = tf.layers.conv2d(drop2, 128, 3, padding='same')
relu3 = tf.nn.relu(conv3)
maxpool3 = tf.layers.max_pooling2d(relu3, (2, 2), (2, 2), padding='same')
drop3 = tf.layers.dropout(maxpool3, 0.3)

conv4 = tf.layers.conv2d(drop3, 128, 3, padding='same')
relu4 = tf.nn.relu(conv4)
maxpool4 = tf.layers.max_pooling2d(relu4, (2, 2), (2, 2), padding='same')
drop4 = tf.layers.dropout(maxpool4, 0.3)

flat = tf.layers.flatten(drop4)
dense1 = tf.layers.dense(flat, 1000)
relu5 = tf.nn.relu(dense1)
logits = tf.layers.dense(relu5, 136)

loss = tf.reduce_mean(tf.square(logits - target))

optimizer = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
cnt = 0

with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init)

    # Iterating through all the epochs
    for epoch in range(100):
        for (_x, _y) in get_batches(arr, y_train, 64):
            l, _ = sess.run([loss, optimizer], feed_dict={inputs: _x, target: _y})

    saver.save(sess, "facial_point.ckpt")
    print('saved')