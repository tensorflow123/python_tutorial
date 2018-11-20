# Copyright 2018 kunming.xie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf 
import numpy as np
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# dropout
with tf.Session() as sess:
    x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
    x = tf.convert_to_tensor(x, name="x")


    keep_prob = 0.5

    # keep_prob转为张量
    keep_prob = tf.convert_to_tensor(
        keep_prob, dtype=x.dtype, name="keep_prob")

    noise_shape = array_ops.shape(x)

    #--------------------------------------------------

    # 生成随机张量，取值范围[keep_prob, 1.0 + keep_prob)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob

    #  0.5
    print(random_tensor.eval())

    random = random_ops.random_uniform(
                    noise_shape, seed=None, dtype=x.dtype)
    #  [0.23802626 0.12538171 0.36279774 0.9606353  0.28212202 0.52161884
    #   0.15747523 0.42480624 0.8671547  0.77771187]
    print(random.eval())

    random_tensor += random

    #  Tensor("add:0", shape=(10,), dtype=float32)
    print(random_tensor)

    #  [1.3773876  1.0458589  1.1407845  1.0497456  0.8394364  0.7275758
    #   0.64525044 0.7586849  1.0062541  1.431607  ]
    print(random_tensor.eval())

    #--------------------------------------------------

    # 在期间[keep_prob, 1.0)之间的，取值为0.
    # 在期间[1.0, 1.0+keep_prob)之间的，取值为1.
    binary_tensor = math_ops.floor(random_tensor)

    #  [0. 0. 0. 1. 0. 1. 1. 1. 1. 0.]
    print(binary_tensor.eval())

    #--------------------------------------------------

    #  ret = (x / keep_prob) * binary_tensor
    ret = math_ops.div(x, keep_prob) * binary_tensor
    #  [ 2.  4.  6.  8. 10. 12. 14. 16. 18. 20.]
    print(math_ops.div(x, keep_prob).eval())

    #  [ 0.  4.  0.  0.  0.  0. 14. 16. 18. 20.]
    print(ret.eval())

