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
import tensorflow as tf

#  tf.reduce_sum(
#      input_tensor,
#      axis=None,
#      keepdims=None,
#      name=None,
#      reduction_indices=None,
#      keep_dims=None
#      )
#
#  Computes the sum of elements across dimensions of a tensor. (deprecated arguments)


#  1 2  4
#  8 16 32
#
#  x has a shape of (2, 3) (two rows and three columns):
#
#  The 0 axis in tensorflow is the rows, 1 axis is the columns.
#  By doing tf.reduce_sum(x, 0) the tensor is reduced along the first dimension
#  (rows), so the result is [1, 2, 4] + [8, 16, 32] = [9, 18, 32].
#
#  By doing tf.reduce_sum(x, 1) the tensor is reduced along the second dimension
#  (columns), so the result is [1, 9] + [2, 16] + [4, 32] = [7, 56].

x = tf.constant([[1, 2, 4], [8, 16, 32]])
a = tf.reduce_sum(x, 0)  # [ 9 18 36]
b = tf.reduce_sum(x, 1)  # [ 7 56]
c = tf.reduce_sum(x, [0, 1])  # 63
d = tf.reduce_sum(x, 1, keepdims=True) # [[ 7]
                                       #  [56]]


with tf.Session() as sess:
    print(sess.run(tf.rank(x)))

    output_a = sess.run(a)
    print(output_a)
    output_b = sess.run(b)
    print(output_b)
    output_c = sess.run(c)
    print(output_c)
    output_d = sess.run(d)
    print(output_d)
