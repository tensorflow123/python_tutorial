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

with tf.Session() as sess:
    x = 1.

    tensor_x = tf.convert_to_tensor(x, tf.float32)

    #  Tensor("Const:0", shape=(), dtype=float32)
    print(tensor_x)

    # 1.0
    print(tensor_x.eval())

