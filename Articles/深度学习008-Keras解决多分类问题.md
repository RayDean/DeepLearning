【火炉炼AI】深度学习008-Keras解决多分类问题
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2, Keras 2.1.6, Tensorflow 1.9.0)

在我前面的文章[【火炉炼AI】深度学习005-简单几行Keras代码解决二分类问题](https://www.jianshu.com/p/09b5a5d82eec)中，介绍了用Keras解决二分类问题。那么多分类问题该怎么解决？有哪些不同？


<br/>

## 1. 准备数据集

为了演示，本次选用了博文[keras系列︱图像多分类训练与利用bottleneck features进行微调（三）](https://blog.csdn.net/sinat_26917383/article/details/72861152)中提到的数据集，原始的数据集将所有类别的train照片放到train文件夹中，所有的test照片放在test文件夹中，而用不同数字开头来表示不同类别，比如以3开头的照片就是bus类等。首先将这些不同类别的照片放在不同的文件夹中，最终的train文件夹有5个子文件夹，每个子文件夹中有80张图片，最终的test文件夹中有5个子文件夹，每个子文件夹中有20张图片。总共只有500张图片。

在代码上，需要用ImageDataGenerator来做数据增强，并且用flow_from_directory来从文件夹中产生数据流。

代码和二分类的文章基本相同，此处就不贴出来了，可以去[**我的github**](https://github.com/RayDean/DeepLearning)直接看全部的代码。

唯一的不同之处是要设置class_mode='categorical'，而不是原来二分类问题的class_mode='binary'

<br/>

## 2. 模型的构建和训练

基本和二分类一样，如下为模型的构建部分：

```py
# 4，建立Keras模型：模型的建立主要包括模型的搭建，模型的配置
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
def build_model(input_shape):
    # 模型的搭建：此处构建三个CNN层+2个全连接层的结构
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # Dropout防止过拟合
    model.add(Dense(class_num)) # 此处多分类问题，用Dense(class_num)
    model.add(Activation('softmax')) #多分类问题用softmax作为activation function
    
    # 模型的配置
    model.compile(loss='categorical_crossentropy', # 定义模型的loss func，optimizer，
                  optimizer=optimizers.RMSprop(), # 使用默认的lr=0.001
                  metrics=['accuracy'])# 主要优化accuracy

    return model # 返回构建好的模型
```

改变之处是：最后的Dense层需要用Dense(class_num)来代替Dense(1)，然后用多分类的标配activation function: softmax。在模型的配置方面，也需要将loss function改为'categorical_crossentropy'。

通过模型的训练后，最终结果如下所示：

![](https://i.imgur.com/khi9bZX.png)

从结果上看：没有出现过拟合现象，但是test acc不太稳定，变化比较大。在平台期后的test acc约为0.85.


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，多分类问题和二分类问题基本相同，不同之处在于：1，设置flow_flow_directory时要用设置class_mode='categorical'。2，模型的最后一层要用Dense(class_num)和softmax这个多分类专用激活函数。3，模型的loss function要使用categorical_crossentropy。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。
