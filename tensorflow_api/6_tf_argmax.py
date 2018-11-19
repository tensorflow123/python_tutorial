from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


sess = tf.Session()
a = tf.constant([1.,2.,3.,0.,9.,])
b = tf.constant([[1,2,3],
                 [3,2,1],
                 [4,5,6],
                 [6,5,4]])

col_max0 = sess.run(tf.argmax(a, 0))
print (col_max0)
#  4

col_max = sess.run(tf.argmax(b, 0) )  #当axis=0时返回每一列的最大值的位置索引
print (col_max)
#  [3 2 2]

row_max = sess.run(tf.argmax(b, 1) )  #当axis=1时返回每一行中的最大值的位置索引
print (row_max)
#  [2 0 2 0]
