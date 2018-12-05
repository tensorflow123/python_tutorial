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
[ 代码 ] (https://github.com/tensorflow123/python_tutorial/tree/master/sklearn/1_train_test_split.py)
from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split

# X生成[0...9], 再转成5行2列的二维矩阵
X, y = np.arange(10).reshape((5, 2)), range(5)

print(X)
#  [[ 0 1]
#  [ 2 3]
#  [ 4 5]
#  [ 6 7]
#  [ 8 9]]


print(y)
#  range(0, 5), [0, 1, 2, 3, 4]

# random_state是随机种子，如果随机种子一样，则split出来的数据是不变的
X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = 0.4, random_state = 22)
print(X_train)
#  [[6 7]
#   [0 1]
#  [8 9]]

print(y_train)
#  [3, 0, 4]

print(X_test)
#  [[2 3]
#  [4 5]]

print(y_test)
#  [1, 2]
