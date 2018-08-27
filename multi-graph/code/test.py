import tensorflow as tf
import numpy as np
import os
import time
from util import PROJECT_ROOT, MODEL_ROOT, CODE_ROOT

MY_MODEL_ROOT = os.path.join(MODEL_ROOT, 'model')
g1_model_path = os.path.join(MODEL_ROOT, 'graph_1/ckpt')
g2_model_path = os.path.join(MODEL_ROOT, 'graph_2/ckpt')
g1_meta_graph = os.path.join(MODEL_ROOT, 'graph_1/ckpt.meta')
g2_meta_graph = os.path.join(MODEL_ROOT, 'graph_2/ckpt.meta')

checkpoint_prefix = os.path.join(MY_MODEL_ROOT, 'ckpt')
X = np.random.random(size=(1000000,1))
Y = 3*X+4
Z = 8*Y + 7

graph = tf.Graph()
with graph.as_default():

  # import the subgraph separately
  saver_1 = tf.train.import_meta_graph(g1_meta_graph)
  saver_2 = tf.train.import_meta_graph(g2_meta_graph)
  initialized_vars = set(tf.global_variables())

  # fetch the placeholder and useful outputs of each subgraph
  # fetch from subgraph 1
  inputs_1 = graph.get_tensor_by_name("graph_1/inputs:0")
  predict_1 = graph.get_tensor_by_name("graph_1/predict:0")

  # fetch from subgraph 2
  inputs_2 = graph.get_tensor_by_name("graph_2/inputs:0")
  predict_2 = graph.get_tensor_by_name("graph_2/predict:0")

  # compose those sub graph to intigrate a bigger graph,
  # and you can add some extra operation to fine tuning you net work
  with tf.name_scope("graph"):
    z = tf.placeholder(tf.float32)
    w_1 = tf.Variable(tf.constant(3.1), dtype=tf.float32, name="w_1")
    b = tf.Variable(tf.constant(4.1), dtype=tf.float32, name="b")
    predict = w_1 * (predict_1 + predict_2) + b
    loss = tf.sqrt(tf.square(predict-z))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


with tf.Session(graph=graph) as sess:

  # recover the stored Variables values, those variables are initialized after restore(), can not use
  # sess.run(tf.global_variables_initializer())
  saver_1.restore(sess, g1_model_path)
  saver_2.restore(sess, g2_model_path)

  # initialize the new Variables added to the big graph
  # (initialize the uninitialized Variables sor far)

  # tf.global_variables() function return a list of Variable objects in scope,
  # if scope is none, all variable objects will be returned

  # tf.report_uninitialized_variables(var_list) recieve a list of variable objects and return an op,
  # when run this op, check var in var_list and return a list of names who are uninitialized so far

  # tf.get_variable(name, **kargs), return a existing variable if name belong to a existing variable object
  # or create a new variable if name does not exist, something about shared variable, go into details: https://www.tensorflow.org/guide/variables#sharing_variables

  # tf.variables_initializer(var_list, name='init'), this function returns an op that initializes a list of variables.
  # this returned op runs all the initializers of the variables in var_list in parallel,
  # if some variable's initialization is related to another variable, some error will occur when sess.run() this op
  
  # sess.run(tf.variables_initializer(list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables(tf.global_variables(scope=None))))))
  sess.run(tf.variables_initializer(set(tf.global_variables())-initialized_vars))
  


  for i in range(10000):
    sess.run(optimizer, feed_dict={inputs_1: X[i], inputs_2: X[i], z: Z[i]})

  # Note: if you want fetch the value of Variable of one subset after restoring them to sess,
  # you need fetch them as an Operation by its name and using the values() method.
  # (an operation has values() method)
  print sess.run(graph.get_operation_by_name("graph_1/w_1").values())
  print sess.run(graph.get_operation_by_name("graph_1/b_1").values())
  print sess.run(graph.get_operation_by_name("graph_2/w_1").values())
  print sess.run(graph.get_operation_by_name("graph_2/b_1").values())
  print w_1.eval()
  print b.eval()
