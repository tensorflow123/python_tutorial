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

# 定义一个计算图，实现两个向量的减法操作

# 定义两个输入，a为常量，b为变量

a=tf.constant([10.0, 20.0, 40.0], name='a')

b=tf.Variable(tf.random_uniform([3]), name='b')

output=tf.add_n([a,b], name='add')

# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中

writer=tf.summary.FileWriter('/tmp/logs', tf.get_default_graph())

writer.close()

