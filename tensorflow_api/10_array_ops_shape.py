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
from tensorflow.python.ops import array_ops

#  https://stackoverflow.com/questions/34079787/tensor-with-unspecified-dimension-in-tensorflow/34082273#34082273
#  As Ishamael says, all tensors have a static shape, which is known at graph construction time 
#  and accessible using Tensor.get_shape(); and a dynamic shape, which is only known at runtime
#  and is accessible by fetching the value of the tensor, or passing it to an operator like tf.shape.
#  In many cases, the static and dynamic shapes are the same, but they can be different -
#  the static shape can be partially defined - in order allow the dynamic shape to vary from one step to the next.

with tf.Session() as sess:
    x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)

    tensor_x = tf.convert_to_tensor(x, tf.float32)

    # Tensor("Shape:0", shape=(1,), dtype=int32)
    print(array_ops.shape(tensor_x))

    # 10
    print(array_ops.shape(tensor_x).eval())
