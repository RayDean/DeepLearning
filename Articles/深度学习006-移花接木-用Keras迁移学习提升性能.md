【火炉炼AI】深度学习006-移花接木-用Keras迁移学习提升性能
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2, Keras 2.1.6, Tensorflow 1.9.0)

上一篇文章我们用自己定义的模型来解决了二分类问题，在20个回合的训练之后得到了大约74%的准确率，一方面是我们的epoch太小的原因，另外一方面也是由于模型太简单，结构简单，故而不能做太复杂的事情，那么怎么提升预测的准确率了？一个有效的方法就是迁移学习。

迁移学习其本质就是移花接木：将其他大数据集（比如ImageNet等）上得到的网络结构及其weights应用到一个新的项目中来，比如此处的猫狗二分类问题。当然，ImageNet中肯定有猫和狗这两个类别，可以说此处的小数据集是ImageNet的一个子集，但是，对于和ImageNet完全没有任何关系的其他数据集，迁移学习也有一定的效果，当然，对于两个数据集的相关性比较差的数据集，使用迁移学习可能效果不太好。

具体做法是：使用一个成熟的网络结构（比如此处用VGG16)和参数，把它的全连接层全部都去掉，只保留卷积层，这些卷积层可以看成是图片的特征提取器（得到的特征被称为bottleneck features），而全连接层是分类器，对这些图片的特征进行有效分类。对于新项目，我们要分类的类别数目并不是ImageNet的1000类，而是比如此处的2类。故而分类器对我们毫无用处，我们需要创建和训练自己的分类器。如下为VGG16网络的结构：

![](https://i.imgur.com/MNeFpuo.png)

其中的Conv block 1-5 都是卷积层和池化层，组成了图片的特征提取器，而后面的Flatten和Dense组成了分类器。

此处我们将Conv block 1-5的结构和参数都移接过来，在组装上自己的分类器即可。

在训练时，我们可以先我上一篇博文一样，建立图片数据流，将图片数据流导入到VGG16模型中提取特征，然后将这些特征送入到自定义的分类器中训练，优化自定义分类器的参数，但是这种方式训练速度很慢，此处我们用VGG16的卷积层统一提取所有图片的特征，将这些特征保存，然后直接加载特征来训练，加载数字比加载图片要快的多，故而训练也快得多。

我这篇博文主要参考了：[keras系列︱图像多分类训练与利用bottleneck features进行微调（三）](https://blog.csdn.net/sinat_26917383/article/details/72861152)，这篇博文也是参考的[Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，但我发现这两篇博文有很多地方的代码跑不起来，主要原因可能是Keras或Tensorflow升级造成的，所以我做了一些必要的修改。

<br/>

## 1. 准备数据集

首先使用预训练好的模型VGG16来提取train set和test set图片的特征，然后将这些特征保存，这些特征实际上就是numpy.ndarray，故而可以保存为数字，然后加载这些数字来训练。

```py
# 此处的训练集和测试集并不是原始图片的train set和test set，而是用VGG16对图片提取的特征，这些特征组成新的train set和test set
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255) # 不需图片增强

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet') 
    # 使用imagenet的weights作为VGG16的初始weights,由于只是特征提取，故而只取前面的卷积层而不需要DenseLayer，故而include_top=False

    generator = datagen.flow_from_directory( # 产生train set
        train_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode=None, 
        shuffle=False) # 必须为False，否则顺序打乱之后，和后面的label对应不上。
    bottleneck_features_train = model.predict_generator(
        generator, train_samples_num // batch_size) # 如果是32，这个除法得到的是62，抛弃了小数，故而得到1984个sample
    np.save('E:\PyProjects\DataSet\FireAI\DeepLearning\FireAI006/bottleneck_features_train.npy', bottleneck_features_train)
    print('bottleneck features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)
    np.save('E:\PyProjects\DataSet\FireAI\DeepLearning\FireAI006/bottleneck_features_val.npy',bottleneck_features_validation)
    print('bottleneck features of test set is saved.')
```

经过上面的代码，trainset图片集的特征被保存到E:\PyProjects\DataSet\FireAI\DeepLearning\FireAI006/bottleneck_features_train.npy文件中，而test set的特征也被保存到../bottleneck_features_val.npy中。

<br/>

## 2. 构建模型并训练

很显然，此处我们并不要提取图片的各种特征，前面的VGG16已经帮我们做完了这件事，所以我们只需要对这些特征进行分类即可，所以相当于我们只建立一个分类器模型就可以。

用keras建立一个简单的二分类模型，如下：

```py
def my_model():
    '''
    自定义一个模型，该模型仅仅相当于一个分类器，只包含有全连接层，对提取的特征进行分类即可
    :return:
    '''
    # 模型的结构
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:])) # 将所有data进行flatten
    model.add(Dense(256, activation='relu')) # 256个全连接单元
    model.add(Dropout(0.5)) # dropout正则
    model.add(Dense(1, activation='sigmoid')) # 此处定义的模型只有后面的全连接层，由于是本项目特殊的，故而需要自定义

    # 模型的配置
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy']) # model的optimizer等

    return model
```

模型虽然建立好了，但我们要训练里面的参数，使用刚刚VGG16提取的特征来进行训练：

```py
# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load('E:\PyProjects\DataSet\FireAI\DeepLearning\FireAI006/bottleneck_features_train.npy') # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array(
    [0] * int((train_samples_num / 2)) + [1] * int((train_samples_num / 2)))
# label是1000个cat，1000个dog，由于此处VGG16特征提取时是按照顺序，故而[0]表示cat，1表示dog

validation_data = np.load('E:\PyProjects\DataSet\FireAI\DeepLearning\FireAI006/bottleneck_features_val.npy')
validation_labels = np.array(
    [0] * int((val_samples_num / 2)) + [1] * int((val_samples_num / 2)))

# 构建分类器模型
clf_model=my_model()
history_ft = clf_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
```

**-------------------------------------输---------出--------------------------------**

Train on 2000 samples, validate on 800 samples
Epoch 1/20
2000/2000 [==============================] - 6s 3ms/step - loss: 0.8426 - acc: 0.7455 - val_loss: 0.4280 - val_acc: 0.8063
Epoch 2/20
2000/2000 [==============================] - 5s 3ms/step - loss: 0.3928 - acc: 0.8365 - val_loss: 0.3078 - val_acc: 0.8675
Epoch 3/20
2000/2000 [==============================] - 5s 3ms/step - loss: 0.3144 - acc: 0.8720 - val_loss: 0.4106 - val_acc: 0.8588

.......

Epoch 18/20
2000/2000 [==============================] - 5s 3ms/step - loss: 0.0479 - acc: 0.9830 - val_loss: 0.5380 - val_acc: 0.9025
Epoch 19/20
2000/2000 [==============================] - 5s 3ms/step - loss: 0.0600 - acc: 0.9775 - val_loss: 0.5357 - val_acc: 0.8988
Epoch 20/20
2000/2000 [==============================] - 5s 3ms/step - loss: 0.0551 - acc: 0.9810 - val_loss: 0.6057 - val_acc: 0.8825

**--------------------------------------------完-------------------------------------**

将训练过程中的loss和acc绘图如下：

![](https://i.imgur.com/Uwr9s63.png)

很显然，在第5个epoch之后，train set和test set出现了很明显的分离，表明后面出现了比较强烈的过拟合，但是在test set上的准确率仍然有90%左右。

可以看出，相对上一篇文章我们自己定义的三层卷积层+两层全连接层的网络结构，用VGG16网络结构的方法得到的准确率更高一些，而且训练所需要的时间也更少。

注意一点：此处我们并没有训练VGG16中的任何参数，而仅仅训练自己定义的分类器模型中的参数。


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，迁移学习就是使用已经存在的模型及其参数，使用该模型来提取图片的特征，然后构建自己的分类器，对这些特征进行分类即可。**

**2，此处我们并没有训练已存在模型的结构和参数，仅仅是训练自定义的分类器，如果要训练已存在模型的参数，那就是微调（Fine-tune）的范畴了**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。
