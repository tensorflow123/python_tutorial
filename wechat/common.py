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
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import glob
import h5py 
from keras.models import model_from_json  
import os
import keras
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
from scipy import misc
#misc.imread把图片转化成矩阵，
#misc.imresize重塑图片尺寸misc.imresize(misc.imread(img), img_size)  img_size是自己设定的尺寸
#ord()函数主要用来返回对应字符的ascii码，
#chr()主要用来表示ascii码对应的字符他的输入时数字，可以用十进制，也可以用十六进制。

def data_generator(data, batch_size): #样本生成器，节省内存
    while True:
        #np.random.choice(x,y)生成一个从x中抽取的随机数,维度为y的向量，y为抽取次数
        batch = np.random.choice(data, batch_size)
        x,y = [],[]
        for img in batch:
            #读取resize图片,再存进x列表
            x.append(misc.imresize(misc.imread(img), img_size))

            #把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
            y.append([ord(i)-ord('a') for i in img[-8:-4]]) 

        #原先是dtype=uint8转成一个纯数字的array
        x = preprocess_input(np.array(x).astype(float))
        y = np.array(y)

        #输出：图片array和四个转化成数字的字母 例如：[array([6]), array([0]), array([3]), array([24])])
        yield x,[y[:,i] for i in range(4)]
