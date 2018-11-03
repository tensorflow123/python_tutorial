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
import numpy as np

#  Computes the mean of elements across dimensions of a tensor. (deprecated arguments)

#  tf.reduce_mean(
#      input_tensor,
#      axis=None,
#      keepdims=None,
#      name=None,
#      reduction_indices=None,
#      keep_dims=None
#      )

c = np.array([[3.,4], [5.,6], [6.,7]])

step = tf.reduce_mean(c, 1)
with tf.Session() as sess:
    print(sess.run(step))

