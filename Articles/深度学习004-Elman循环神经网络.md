【火炉炼AI】深度学习004-Elman循环神经网络
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2 )

Elman神经网络是最早的循环神经网络，由Elman于1990年提出，又称为SRN（Simple Recurrent Network, 简单循环网络）。SRN考虑了时序信息，当前时刻的输出不仅和当前时刻的输入有关，还和前面所有时刻的输入有关。SRN是RNN结构中最简单的一种，相对于传统的两层全连接前馈网络，它仅仅在全连接层添加了时序反馈连接。

简单点来理解，以前的深度神经网络的计算可以简单理解为：yt=f(Xt)，而SRN却把上一个时刻的结果也当做输入放入模型中，相当于yt=f(Xt,yt-1)，由于这种递归性，每一个yt结果不仅仅和自身的特征向量Xt有关，还和前一个时刻的输出结果yt-1有关，如此递归，便认为yt和以前的所有的Xt，Xt-1,Xt-2...都有关，那么yt就相当于“记住了”前面Ｎ个时刻的所有输入变量X.

那么SRN是怎么做到这一点的了？SRN一般分为四层：输入层，隐含层，承接层，输出层，和前面讲到的简单神经网络不同的是，承接层就起到一个临时变量Var的作用，在t-1时刻得到的结果yt-1后，把yt-1输出的同时也保存一份给Var，然后再计算t时刻的结果是，把Var也当做一个变量输入，所以Var相当于一个延时算子，达到记忆的目的，使得整个网络结构具有适应时间序列的应变能力。如下图所示：

![](https://i.imgur.com/Tev7uJC.png)

图中的循环层其实就是承接层，名称不同而已，这个网络结构不太容易看出时序性，故而展开后为：

![](https://i.imgur.com/RbZwPc7.png)

图片来源于[Recurrent Neural Network(循环神经网络)](https://www.cnblogs.com/neopenx/p/4623328.html).

关于更多更复杂的循环神经网络结构，可以参考博文[循环神经网络(RNN, Recurrent Neural Networks)介绍](https://blog.csdn.net/heyongluoyao8/article/details/48636251)

那么怎么建立并训练SRN模型了？

<br/>

## 1. 构建并训练Elman循环神经网络

### 1.1 准备数据集

此次我们自动生成一系列数据，该系列数据包含四段数据，如下为数据生成函数。

```py
# 准备数据集
# 用np生成一些序列数据，这个序列数据有四段
def waveform_dataset(points_num):
    '''建立波形数据集，这个数据集含有四段，每一段的数据点数为points_num'''
    stage1=1*np.cos(np.arange(points_num))
    stage2=2*np.cos(np.arange(points_num))
    stage3=3*np.cos(np.arange(points_num))
    stage4=4*np.cos(np.arange(points_num))
    
    dataset_X=np.array([stage1,stage2,stage3,stage4])# 4行points_num列
    dataset_X=dataset_X.reshape(points_num*4,1) # 转变为：4*points_num行，一列，即为整个序列
    
    amp1 = np.ones(points_num) # 每一段数据的幅度不同 分别是1,4,2,0.5
    amp2 = 4 + np.zeros(points_num) 
    amp3 = 2 * np.ones(points_num) 
    amp4 = 0.5 + np.zeros(points_num) 
    dataset_y=np.array([amp1,amp2,amp3,amp4]).reshape(points_num*4,1)
    return dataset_X,dataset_y
```

可以看看数据集的分布情况：

![](https://i.imgur.com/QtYqVRM.png)


### 1.2 构建并训练模型

直接上代码，此处用neurolab模块中的现有函数newelm()来构建一个SRN模型，包含两层神经网络。

```py
# 构建并训练模型
import neurolab as nl
net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
# 创建两层的神经网络
net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
net.layers[1].initf= nl.init.InitRand([-0.1, 0.1], 'wb')
net.init()
# 网络的初始化
error = net.train(dataset_X, dataset_y, epochs=3000, show=300, goal=0.01)
```

**-------------------------------------输---------出--------------------------------**

Epoch: 300; Error: 0.08632353521527447;
Epoch: 600; Error: 0.07758197978278435;
Epoch: 900; Error: 0.047083147244329486;
Epoch: 1200; Error: 0.03948011155907889;
Epoch: 1500; Error: 0.03808612642771739;
Epoch: 1800; Error: 0.03600983543384789;
Epoch: 2100; Error: 0.04108011778013388;
Epoch: 2400; Error: 0.0388262030539809;
Epoch: 2700; Error: 0.033576743782171244;
Epoch: 3000; Error: 0.03329548827926802;
The maximum number of train epochs is reached

**--------------------------------------------完-------------------------------------**

### 1.3 用训练好的模型预测新样本

此处假设训练集所用的dataset_X为新样本，那么看看得到的结果和预测值之间的差异。

```py
# 用训练好的模型预测新样本
predict_y=net.sim(dataset_X)
plt.plot(dataset_y,label='dataset')
plt.plot(predict_y,label='predicted')
plt.legend()
plt.title('Comparison of Truth and Predicted')
```

![](https://i.imgur.com/GYzPhbA.png)

当然，我们也可以用函数waveform_dataset()来产生一些新数据，然后用训练好的模型预测一下试试。

```py
# 生成新的数据集
newset_X,newset_y=waveform_dataset(100)
predict_y=net.sim(newset_X)
plt.plot(newset_y,label='dataset')
plt.plot(predict_y,label='predicted')
plt.legend()
plt.title('Comparison of Truth and Predicted')
```

![](https://i.imgur.com/HhCnLrP.png)

可以发现，对于新产生的序列数据，该模型也能够大体预测出来。


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，neurolab中已经集成了一些简单的神经网络方面的函数，比如最简单的循环神经网络模型--Elman循环神经网络，对于复杂的或者需要自己定义的循环神经网络，需要用其他更复杂的深度学习框架。**

**1，Elman循环神经网络模型是一种最简单的循环神经网络结构，只能解决一些相对简单的序列数据问题。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译