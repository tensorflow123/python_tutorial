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
#博客地址:https://blog.csdn.net/kelsel
#[代码](https://github.com/tensorflow123/python_tutorial/tree/master/starter_bundle/10_stochasitc_gradient_descent.py)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 正则化函数
# $y = \frac{1}{ 1+e^{-x}}$
def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds

def next_batch(X, y, batchSize):
    #  np.arange(start, stop, step, dtype)
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i : i+ batchSize], y[i : y + batchSize])

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 100,
        help="# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01,
        help = "learning rate")
ap.add_argument("-b", "--batch-size", type = float, default = 32,
        help = "size of SGD mini-batches")
args = vars(ap.parse_args())

# 造数据
(X, y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2,
        cluster_std = 1.5, random_state = 1)
# y.shape (1000,) -> (1000, 1)
y = y.reshape((y.shape[0], 1))

#  按行连接
#  In [1]: from numpy import c_
#  In [2]: a = ones(4)
#  In [3]: b = zeros((4,10))
#  In [4]: c_[a,b]
#  Out[4]:
#  array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y,
        test_size = 0.5, random_state = 42)


# 随机初始化W和loss
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []


for epoch in np.arange(0, args["epochs"]):
    epochLosses = []

    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):

        preds = sigmoid_activation(batchX.dot(W))

        # 计算方差
        error = preds - batchY
        epochLosses.append(np.sum(error ** 2))

        gradient = batchX.T.dot(error)

        # 右边的计算结果是(3, 1)的矩阵
        W += -args["alpha"] * gradient

    loss = np.average(epochLosses)
    losses.append(loss)

    # 右边的计算结果是(3, 1)的矩阵
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1), loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))


