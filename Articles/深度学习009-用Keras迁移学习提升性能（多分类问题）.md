【火炉炼AI】深度学习009-用Keras迁移学习提升性能（多分类问题）
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2, Keras 2.1.6, Tensorflow 1.9.0)

本文是仿照前面的文章[【火炉炼AI】深度学习006-移花接木-用Keras迁移学习提升性能](https://www.jianshu.com/p/4e3d13a7f4bf)，原文是针对二分类问题，使用迁移学习的方式来提升准确率，此处我用迁移学习的方式来提升多分类问题的准确率。

同时，在我前面的文章中[【火炉炼AI】深度学习008-Keras解决多分类问题](https://www.jianshu.com/p/6904aa059387)，使用普通的小型3层卷积网络+2层全连接层实现了多分类的85%左右的准确率， 此处我将用迁移学习的方式来进一步提升性能。

<br/>

## 1. 准备数据集

与上一篇文章相同。

<br/>

## 2. 模型的构建和训练

使用迁移学习包括两个步骤：第一步是用VGG16网络提取bottleneck features，然后保存这些特征，第二步建立自己的分类器，然后用用保存的特征来训练分类器，优化weights。下面是第一步的代码，只注意一个地方：要修改为：class_mode='categorical'

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
        class_mode='categorical', # 这个地方要修改，要不然出错
        shuffle=False) # 必须为False，否则顺序打乱之后，和后面的label对应不上。
    bottleneck_features_train = model.predict_generator(
        generator, train_samples_num // batch_size) 
    np.save(os.path.join(save_folder,'bottleneck_features_train.npy'), bottleneck_features_train)
    print('bottleneck features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)
    np.save(os.path.join(save_folder,'bottleneck_features_val.npy'),bottleneck_features_validation)
    print('bottleneck features of test set is saved.')
```

第二步： 定义自己的分类器，专门适用于本项目的多分类问题，要将最后的Dense层修改为：model.add(Dense(class_num, activation='softmax'))，并且loss使用'categorical_crossentropy'。

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
    model.add(Dense(class_num, activation='softmax')) # 与二分类不同之处：要用Dense(class_num)和softmax

    # 模型的配置
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) # model的optimizer等

    return model
```

然后使用保存的特征对该分类器进行训练，训练时有一个地方要注意：在二分类问题时，我们直接将labels定义为np.array([0]*80+[1]*80)，但是多分类问题，需要用to_categorical进行转换，然后再放入到fit中。

```py
from keras.utils import to_categorical
# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load(os.path.join(save_folder,'bottleneck_features_train.npy')) # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array([0] * 80 + [1] * 80+ [2]*80+[3]*80+[4]*80)
# label是每个类别80张图片，共5个类别
# 设置标签，并规范成Keras默认格式
train_labels = to_categorical(train_labels, class_num)

validation_data = np.load(os.path.join(save_folder,'bottleneck_features_val.npy'))
validation_labels = np.array([0] * 20 + [1] * 20+ [2]*20+[3]*20+[4]*20)
validation_labels = to_categorical(validation_labels, class_num)

# 构建分类器模型
clf_model=my_model()
history_ft = clf_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
```

**-------------------------------------输---------出--------------------------------**

Train on 400 samples, validate on 100 samples
Epoch 1/50
400/400 [==============================] - 1s 2ms/step - loss: 4.8496 - acc: 0.4725 - val_loss: 3.4864 - val_acc: 0.6900
Epoch 2/50
400/400 [==============================] - 0s 878us/step - loss: 3.4436 - acc: 0.7400 - val_loss: 3.3577 - val_acc: 0.7400
Epoch 3/50
400/400 [==============================] - 0s 845us/step - loss: 0.6014 - acc: 0.9300 - val_loss: 0.1798 - val_acc: 0.9600

...

Epoch 48/50
400/400 [==============================] - 0s 860us/step - loss: 5.4962e-06 - acc: 1.0000 - val_loss: 0.3184 - val_acc: 0.9500
Epoch 49/50
400/400 [==============================] - 0s 853us/step - loss: 0.0438 - acc: 0.9850 - val_loss: 0.2470 - val_acc: 0.9600
Epoch 50/50
400/400 [==============================] - 0s 855us/step - loss: 5.5508e-04 - acc: 1.0000 - val_loss: 0.2798 - val_acc: 0.9700

**--------------------------------------------完-------------------------------------**

最后将acc和loss绘图，如下：

![](https://i.imgur.com/6IkNkvC.png)

可以看出最后的test acc达到了0.96以上。


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，多分类的迁移学习问题需要注意几个地方：提取bottleneck features时要设置class_mode='categorical'。构建自己的分类器时要把输出层的Dense修改，然后把激活函数和损失函数修改为多分类对应的函数。在做labels时，要先用to_categorical转变为one-hot形式再输入到模型中进行fit.**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。
