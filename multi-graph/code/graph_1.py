import tensorflow as tf
import numpy as np
import os
import time
from util import PROJECT_ROOT, MODEL_ROOT, CODE_ROOT

MY_MODEL_ROOT = os.path.join(MODEL_ROOT, 'graph_1')

checkpoint_prefix = os.path.join(MY_MODEL_ROOT, 'ckpt')

X = np.random.random(size=(1000000,1))
Y = 3*X+4

graph_1 = tf.Graph()
with graph_1.as_default():
  with tf.name_scope("graph_1"):
    x = tf.placeholder(tf.float32, name="inputs")
    y = tf.placeholder(tf.float32, name="targets")
    w = tf.Variable(tf.constant(1.0), dtype=tf.float32, name="w_1")
    b = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="b_1")
    predict = tf.add(tf.multiply(x, w), b, name="predict")
    loss = tf.sqrt(tf.square(y-predict))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session(graph=graph_1) as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(3000):
    loss_, _ = sess.run([loss, optimizer], feed_dict={x: X[i][0], y: Y[i][0]})
    print w.eval(), b.eval()
  saver.save(sess, checkpoint_prefix)