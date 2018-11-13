【火炉炼AI】深度学习005-简单几行Keras代码解决二分类问题
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2, Keras 2.1.6, Tensorflow 1.9.0)

很多文章和教材都是用MNIST数据集作为深度学习届的“Hello World”程序，但是这个数据集有一个很大的特点：它是一个典型的多分类问题（一共有10个分类），在我们刚刚开始接触深度学习时，我倒是觉得应该从最简单的二分类问题着手。

在深度学习框架方面，目前比较流行的是Tensorflow，Keras，PyTorch，Theano等，但是我建议新手入门，可以从Keras入手，然后进阶时转移到Tensorflow上，实际上，Keras的后端是可以支持Tensorflow和Theano，可以说，Keras是在Tensorflow和Theano的基础上进一步封装，更加的简单实用，更容易入门，通常几行简单的代码就可以解决一个小型的项目问题。

我这篇博文主要参考了：[keras系列︱图像多分类训练与利用bottleneck features进行微调（三）](https://blog.csdn.net/sinat_26917383/article/details/72861152)，这篇博文也是参考的[Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，但我发现这两篇博文有很多地方的代码跑不起来，主要原因可能是Keras或Tensorflow升级造成的，所以我做了一些必要的修改。

<br/>

## 1. 准备数据集

最经典的二分类数据集就是Kaggle竞赛中的“猫狗大战”数据集（train set有25K张图片，test set： 12.5K），此处按照原始博文的做法，我从train_set中选取1000张Dog的照片+1000张Cat照片作为我们新的train set，选取400张Dog+400张Cat照片作为新的test set。所以train和test两个文件夹下都有两个子文件夹（cats和dogs子文件夹）。当然，选取是随机的，也是用代码来实现的，准备小数据集的代码如下：

```py
def dataset_prepare(raw_set_folder,dst_folder,train_num_per_class=1000,test_num_per_class=400):
    '''
    准备小数据集，从原始的raw_set_folder数据集中提取train_num_per_class(每个类别)的照片放入train中，
    提取val_num_per_class（每个类别）放入到validation文件夹中
    :param raw_set_folder: 含有猫狗的照片，这些照片的名称必须为cat.101.jpg或dog.102.jpg形式
    :param dst_folder: 将选取之后的图片放置到这个文件夹中
    :param train_num_per_class:
    :param test_num_per_class:
    :return:
    '''
    all_imgs=glob(os.path.join(raw_set_folder,'*.jpg'))
    img_len = len(all_imgs)
    assert img_len > 0, '{} has no jpg image file'.format(raw_set_folder)

    cat_imgs=[]
    dog_imgs=[]
    for img_path in all_imgs:
        img_name=os.path.split(img_path)[1]
        if img_name.startswith('cat'):
            cat_imgs.append(img_path)
        elif img_name.startswith('dog'):
            dog_imgs.append(img_path)
    random.shuffle(cat_imgs)
    random.shuffle(dog_imgs)
    [ensure_folder_exists(os.path.join(dst_folder,type_folder,class_folder)) for type_folder in ['train','test']
        for class_folder in ['dogs','cats']]
    # 下面的代码可以进一步优化。。。。
    for cat_img_path in cat_imgs[:train_num_per_class]: # 最开始的N个图片作为train
        _, fname = os.path.split(cat_img_path)  # 获取文件名和路径
        shutil.copyfile(cat_img_path, os.path.join(dst_folder, 'train', 'cats',fname))
    print('imgs saved to train/cats folder')
    for dog_img_path in dog_imgs[:train_num_per_class]:
        _, fname = os.path.split(dog_img_path)  # 获取文件名和路径
        shutil.copyfile(dog_img_path, os.path.join(dst_folder, 'train', 'dogs',fname))
    print('imgs saved to train/dogs folder')
    for cat_img_path in cat_imgs[-test_num_per_class:]: # 最末的M个图片作为test
        _, fname = os.path.split(cat_img_path)  # 获取文件名和路径
        shutil.copyfile(cat_img_path, os.path.join(dst_folder, 'test', 'cats',fname))
    print('imgs saved to test/cats folder')
    for dog_img_path in dog_imgs[-test_num_per_class:]: # 最末的M个图片作为test
        _, fname = os.path.split(dog_img_path)  # 获取文件名和路径
        shutil.copyfile(dog_img_path, os.path.join(dst_folder, 'test', 'dogs',fname))
    print('imgs saved to test/dogs folder')
    print('finished...')
```

运行该函数即可完成小数据集的构建，下面为Keras创建图片数据流，为模型的构建做准备。

```py
# 2，准备训练集，keras有很多Generator可以直接处理图片的加载，增强等操作，封装的非常好
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( # 单张图片的处理方式，train时一般都会进行图片增强
        rescale=1. / 255, # 图片像素值为0-255，此处都乘以1/255，调整到0-1之间
        shear_range=0.2, # 斜切
        zoom_range=0.2, # 放大缩小范围
        horizontal_flip=True) # 水平翻转

train_generator = train_datagen.flow_from_directory(# 从文件夹中产生数据流
    train_data_dir, # 训练集图片的文件夹
    target_size=(IMG_W, IMG_H), # 调整后每张图片的大小
    batch_size=batch_size,
    class_mode='binary') # 此处是二分类问题，故而mode是binary

# 3，同样的方式准备测试集
val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='binary')
```

上面构建的generator就是keras需要的数据流，该数据流使用flow_from_directory首先从图片文件夹（比如train_data_dir）中加载图片到内存中，然后使用train_datagen来对图片进行预处理和增强，最终得到处理完成之后的batch size大小的数据流，这个数据流会无限循环的产生，直到达到一定的训练epoch数量为止。

上面用到了ImageDataGenerator来进行图片增强，里面的参数说明为：（可以参考Keras的[官方文档](https://keras.io/zh/backend/)）

rotation_range是一个0~180的度数，用来指定随机选择图片的角度。

width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。

rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。

shear_range是用来进行剪切变换的程度

zoom_range用来进行随机的放大

horizontal_flip随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候

fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素


<br/>

## 2. 构建并训练Keras模型

由于Keras已经封装了很多Tensorflow的函数，所以在使用上更加简单容易，当然，如果想调整里面的结构和参数等，也比较麻烦一些，所以对于高手，想要调整模型的结构和自定义一些函数，可以直接用Tensorflow.

### 2.1 Keras模型的构建

不管是Keras模型还是Tensorflow模型，我个人认为其构建都包括两个部分：模型的搭建和模型的配置，所以可以从这两个方面来建立一个小型的模型。代码如下：

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
    model.add(Dense(1)) # 此处虽然是二分类，但是不能用Dense(2)，因为后面的activation是sigmoid，这个函数只能输出一个值，即class_0的概率
    model.add(Activation('sigmoid')) #二分类问题用sigmoid作为activation function
    
    # 模型的配置
    model.compile(loss='binary_crossentropy', # 定义模型的loss func，optimizer，
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=['accuracy'])# 主要优化accuracy
    # 二分类问题的loss function使用binary_crossentropy，此处使用准确率作为优化目标
    return model # 返回构建好的模型
```

这个函数就搭建了模型的结构，对模型进行了配置，主要配置了loss function, optimzer, 优化目标等，当然可以做更多其他配置。

此处，为了简单说明，只是建立了三层卷积层+两层全连接层的小型网络结构，当然，对于一些比较简单的图像问题，这个小型模型也能解决。如果需要构建更为复杂的模型，只需要自定义这个函数，修改里面的模型构建和配置方法即可。

### 2.2 模型的训练

由于此处我们使用generator来产生数据流，故而训练时要使用fit_generator函数。代码如下：

```py
model=build_model(input_shape=(IMG_W,IMG_H,IMG_CH)) # 输入的图片维度
# 模型的训练
model.fit_generator(train_generator, # 数据流
                    steps_per_epoch=train_samples_num // batch_size, 
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_samples_num // batch_size)
```

由于我在自己的笔记本上训练，没有独立显卡，更没有英伟达那么NB的显卡，故而速度很慢，但是的确能运行下去。运行的具体结果可以去[**我的github**](https://github.com/RayDean/DeepLearning)上看。

**-------------------------------------输---------出--------------------------------**

Epoch 1/20
62/62 [==============================] - 136s 2s/step - loss: 0.6976 - acc: 0.5015 - val_loss: 0.6937 - val_acc: 0.5000
Epoch 2/20
62/62 [==============================] - 137s 2s/step - loss: 0.6926 - acc: 0.5131 - val_loss: 0.6846 - val_acc: 0.5813
Epoch 3/20
62/62 [==============================] - 152s 2s/step - loss: 0.6821 - acc: 0.5544 - val_loss: 0.6735 - val_acc: 0.6100

。。。

Epoch 18/20
62/62 [==============================] - 140s 2s/step - loss: 0.5776 - acc: 0.6880 - val_loss: 0.5615 - val_acc: 0.7262
Epoch 19/20
62/62 [==============================] - 143s 2s/step - loss: 0.5766 - acc: 0.6971 - val_loss: 0.5852 - val_acc: 0.6800
Epoch 20/20
62/62 [==============================] - 140s 2s/step - loss: 0.5654 - acc: 0.7117 - val_loss: 0.5374 - val_acc: 0.7450

**--------------------------------------------完-------------------------------------**

从训练后的loss和acc上可以大致看出，loss在不断减小，acc也不断增大，趋势比较平稳。

此处我们可以将训练过程中的loss和acc绘图，看看他们的变化趋势。

```py
# 画图，将训练时的acc和loss都绘制到图上
import matplotlib.pyplot as plt
%matplotlib inline
def plot_training(history):
    plt.figure(12)
    
    plt.subplot(121)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b',label='train_acc')
    plt.plot(epochs, val_acc, 'r',label='test_acc')
    plt.title('Train and Test accuracy')
    plt.legend()
    
    plt.subplot(122)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b',label='train_loss')
    plt.plot(epochs, val_loss, 'r',label='test_loss')
    plt.title('Train and Test loss')
    plt.legend()
 
    plt.show()
```

![](https://i.imgur.com/LOGjNcB.png)

很明显，由于epoch次数太少，acc和loss都没有达到平台期，后续可以增大epoch次数来达到一个比较好的结果。在原始博文中，作者在50个epoch之后达到了约80%左右的准确率，此处我20个epoch后的准确率为74%。

### 2.3 预测新样本

**单张图片的预测**

模型训练好之后，就需要用来预测新的图片，看看它能不能准确的给出结果。预测函数为：

```py
# 用训练好的模型来预测新样本
from PIL import Image
from keras.preprocessing import image
def predict(model, img_path, target_size):
    img=Image.open(img_path) # 加载图片
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img) 
    x *=1./255 # 相当于ImageDataGenerator(rescale=1. / 255)
    x = np.expand_dims(x, axis=0) # 调整图片维度
    preds = model.predict(x) # 预测
    return preds[0]
```

用这个函数可以预测单张图片：

```py
predict(model,'E:\PyProjects\DataSet\FireAI\DeepLearning/FireAI005/cat11.jpg',(IMG_W,IMG_H))

predict(model,'E:\PyProjects\DataSet\FireAI\DeepLearning//FireAI005/dog4.jpg',(IMG_W,IMG_H))
```

**-------------------------------------输---------出--------------------------------**

array([0.14361556], dtype=float32)

array([0.9942463], dtype=float32)

**--------------------------------------------完-------------------------------------**

可以看出，对于单张图片cat11.jpg得到的概率为0.14，而dog4.jpg的概率为0.99，可以看出第0个类别是dog，第1个类别是cat，模型能够很好的区分开来。

**多张图片的预测**

如果想用这个模型来预测一个文件夹中的所有图片，那么该怎么办了？

```py
# 预测一个文件夹中的所有图片
new_sample_gen=ImageDataGenerator(rescale=1. / 255)
newsample_generator=new_sample_gen.flow_from_directory(
        'E:\PyProjects\DataSet\FireAI\DeepLearning',
        target_size=(IMG_W, IMG_H),
        batch_size=16,
        class_mode=None,
        shuffle=False)
predicted=model.predict_generator(newsample_generator)
print(predicted)
```

**-------------------------------------输---------出--------------------------------**

Found 4 images belonging to 2 classes.
[[0.14361556]
 [0.5149474 ]
 [0.71455824]
 [0.9942463 ]]

**--------------------------------------------完-------------------------------------**

上面的结果中第二个0.5149对应的应该是cat，应该小于0.5，这个预测是错误的，不过粗略估计正确率有3/4=75%。


### 2.4 模型的保存和加载

模型一般要及时保存到硬盘上，防止数据丢失，下面是保存的代码：

```py
# 模型保存
# model.save_weights('E:\PyProjects\DataSet\FireAI\DeepLearning//FireAI005/FireAI005_Model.h5') # 这个只保存weights，不保存模型的结构
model.save('E:\PyProjects\DataSet\FireAI\DeepLearning//FireAI005/FireAI005_Model2.h5') # 对于一个完整的模型，应该要保存这个
```

```py
# 模型的加载，预测
from keras.models import load_model
saved_model=load_model('E:\PyProjects\DataSet\FireAI\DeepLearning//FireAI005/FireAI005_Model2.h5')

predicted=saved_model.predict_generator(newsample_generator)
print(predicted) # saved_model的结果和前面的model结果一致，表面模型正确保存和加载
```

此处得到的结果和上面model预测的结果一模一样，表明模型被正确保存和加载。

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，本篇文章讲解了：准备一个简单的小数据集，从数据集中建立数据流，将该数据流引入到Keras的模型中进行训练，并使用训练后的模型进行新图片的预测，然后将模型进行保存，加载保存好的模型到内存中。 **

**2，此处使用的模型是我们自己搭建的，结构比较简单，只有三层卷积层和两层全连接层，故而模型的准确率不太高，而且此处由于时间关系，我只训练了20个epoch，训练并没有达到平台期。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。
