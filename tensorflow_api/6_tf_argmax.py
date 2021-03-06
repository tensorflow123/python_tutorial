#  /* Copyright 2018 kunming.xie
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *    http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#   */
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
