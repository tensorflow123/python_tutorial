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

train_dir=r'/tmp/wechat_test'
if not os.path.exists(train_dir):
    os.mkdir(r'/tmp/wechat_test')
os.chdir(r'/tmp/wechat_test')

#获取指定目录下的所有图片
samples = glob.glob(r'/home/v/.keras/datasets/sample/*.jpg')
np.random.shuffle(samples) 

nb_train = 90000 #共有10万+样本，9万用于训练，1万+用于验证
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]

# [a-z]
letter_list = [chr(i) for i in range(97,123)]

from keras.applications.xception import Xception,preprocess_input
from keras.layers import Input,Dense,Dropout
from keras.models import Model

img_size = (50, 120) 
#CNN适合在高宽都是偶数的情况，否则需要在边缘补齐，
#那么我们也可以把全体图片都resize成这个尺寸(高，宽，通道)
input_image = Input(shape=(img_size[0],img_size[1],3))

#具体方案是：直接将验证码输入，做几个卷积层提取特征，
#然后把这些提出来的特征连接几个分类器（26分类，因为不区分大小写），
#如果需要加入数字就是36分类，微信验证码里没有数字。

#输入图片
#用预训练的Xception提取特征,采用平均池化
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')

#用全连接层把图片特征接上softmax然后26分类，dropout为0.5，激活使用softmax因为是多分类问题。
#ReLU - 用于隐层神经元输出
#Sigmoid - 用于隐层神经元输出
#Softmax - 用于多分类神经网络输出
#Linear - 用于回归神经网络输出（或二分类问题）
#对四个字母做26分类
predicts = [Dense(26, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#optimizer：优化器的选择可以参考这篇博客https://www.jianshu.com/p/d99b83f4c1a6
#loss：损失函数，这里选去稀疏多类对数损失
#metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
#sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表。在下面fit函数的解释中有相关的参考内容。
#weighted_metrics: metrics列表，在训练和测试过程中，这些metrics将由sample_weight或clss_weight计算并赋权
#target_tensors: 默认情况下，Keras将为模型的目标创建一个占位符，该占位符在训练过程中将被目标数据代替。如果你想使用自己的目标张量（相应的，Keras将不会在训练时期望为这些目标张量载入外部的numpy数据），你可以通过该参数手动指定。目标张量可以是一个单独的张量（对应于单输出模型），也可以是一个张量列表，或者一个name->tensor的张量字典。
#model.summary()

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

from keras.utils.vis_utils import plot_model
plot_model(model, to_file="model.png", show_shapes=True)

model.fit_generator(data_generator(train_samples, 100), steps_per_epoch=1000, epochs=10, validation_data=data_generator(test_samples, 100), validation_steps=100) 
#  参数：generator生成器函数,
#  samples_per_epoch，每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
#  step_per_epoch:整数，当生成器返回step_per_epoch次数据是记一个epoch结束，执行下一个epoch
#  epochs:整数，数据迭代的轮数
#  validation_data三种形式之一，生成器，类（inputs,targets）的元组，或者（inputs,targets，sample_weights）的元祖
#  若validation_data为生成器，validation_steps参数代表验证集生成器返回次数
#  class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
#  sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
#  workers：最大进程数
#  max_q_size：生成器队列的最大容量
#  pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
#  initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。

#保存模型
model.save('CaptchaForWechat.h5')

#评价模型的全对率(批量预测,num为预测样本总量)
def predict1(num):
    from tqdm import tqdm
    total = 0.
    right = 0.
    step = 0
    for x,y in tqdm(data_generator(test_samples, num)):
        z = model.predict(x)
        print (z)
        z = np.array([i.argmax(axis=1) for i in z]).T
        #print (z)
        #i.argmax(axis = 1)返回每行中最大数的索引，i.argmax(axis = 0)返回每列中最大数的索引
        #26分类，索引为（0-25）对应（a-z），取出概率最大的索引找出对应的字母即可
        y = np.array(y).T #原先的正确结果
        total += len(x) #样本数量
        right += ((z == y).sum(axis=1) == 4).sum()#四个都对就默认对了一个
        if step < 100:
            step += 1
        else:
            break
    result = u'模型全对率：%s'%(right/total)
    return result

test_samples1 = glob.glob(r'/tmp/wechat_test/test/*.jpg')
test_list = [i for i in range(len(test_samples1))]
def data_generator_test1(data, n): #样本生成器，节省内存
    while True:
        batch = np.array([data[n]])
        x,y = [],[]
        for img in batch:
            x.append(misc.imresize(misc.imread(img), img_size)) #读取resize图片,再存进x列表
            y.append([ord(i)-ord('a') for i in img[-8:-4]]) #把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
        x = preprocess_input(np.array(x).astype(float)) #原先是dtype=uint8转成一个纯数字的array
        y = np.array(y)
        yield x,[y[:,i] for i in range(4)]        

#单张图片测试    
def predict2(n):
    x,y = next(data_generator_test1(test_samples1, n))
    z = model.predict(x)
    z = np.array([i.argmax(axis=1) for i in z]).T
    result = z.tolist() 
    v = []
    for i in range(len(result)):
        for j in result[i]:
            v.append(letter_list[j])
    image = mpimg.imread(test_samples1[n])
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    #输出测试结果
    str = ''
    for i in v:
        str += i
    return (str)
