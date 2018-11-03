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
import numpy as np

# numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None) : Returns number spaces evenly w.r.t interval. Similiar to arange but instead of step it uses sample number.

x = np.linspace(0, 1, 5)
x2 = np.linspace(0, 1, 5)[:, np.newaxis]

print(x) # [0.   0.25 0.5  0.75 1.  ]
print(x.shape) # (5,)

print(x2)
#  [[0.  ]
#   [0.25]
#   [0.5 ]
#   [0.75]
#   [1.  ]]
print(x2.shape) # (5,1)


#  newaxis
#  Simply put, the newaxis is used to increase the dimension of the existing array
#  by one more dimension, when used once. Thus,
#
#  1D array will become 2D array
#  2D array will become 3D array
#  3D array will become 4D array

# 1x1
a = np.arange(3)
print(a) # [0 1 2]
print(a.shape) #(3,)

# 2x2
row_vec = a[np.newaxis, :]
print(row_vec) # [[0 1 2]]
print(row_vec.shape) # (1, 3)

# 2x2
col_vec = a[:, np.newaxis]
print(col_vec)
#  [[0]
#   [1]
#   [2]]

print(col_vec.shape) # (3, 1)
