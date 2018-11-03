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

#  tf.rank(
#      input,
#      name=None
#      )
#
#  Returns the rank of a tensor.

#  tf.rank returns the dimension of a tensor, not the number of elements. For
#  instance, the output from tf.rank called for the 2x2 matrix would be 2.

x = tf.constant([[1, 2, 4]]) # 2x2 matrix
x2 = tf.constant([[1, 2, 4], [8, 16, 32]]) # 2x2 matrix
x3 = tf.constant([1, 2, 4]) # 1x1 matrix

with tf.Session() as sess:
    print(sess.run(tf.rank(x))) # 2
    print(sess.run(tf.rank(x2))) # 2
    print(sess.run(tf.rank(x3))) # 1
