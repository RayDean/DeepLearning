【火炉炼AI】深度学习010-Keras微调提升性能（多分类问题）
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2, Keras 2.1.6, Tensorflow 1.9.0)

前面的文章([【火炉炼AI】深度学习007-Keras微调进一步提升性能](https://www.jianshu.com/p/5c766be9a9d7))我们对二分类问题用Keras进行了Fine-tune，使得模型的准确率进一步提升，此处我们看看对于多分类问题，怎么使用Fine-tune来提升性能。

<br/>

## 1. 准备数据集

和文章[【火炉炼AI】深度学习008-Keras解决多分类问题](https://www.jianshu.com/p/6904aa059387)一模一样，在使用flow_from_directory时需要将class_mode修改为'categorical'。

<br/>

## 2. 对VGG16的后半部分进行Fine-tune

此处我们建立的模型是使用VGG16的身子（inclue_top=False）作为特征提取器，和我们自己定义的头（前面文章[【火炉炼AI】深度学习009-用Keras迁移学习提升性能（多分类问题）](https://www.jianshu.com/p/d0723a267771)中训练的模型和weights），这篇文章中已经达到了0.96的准确率。但是在前面这篇文章的迁移学习中，我们并没有修改VGG16这个网络的weights参数，而是直接拿来提取特征，此处的Fine-tune就是要调整VGG16网络的较高层的卷积层的weights，使其更加适用于我们自己的项目。

如下是模型构建的函数：

```py
# 4，构建模型
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras.models import Model
def build_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(IMG_W, IMG_H,IMG_CH))
    # 此处我们只需要卷积层不需要全连接层，故而inclue_top=False,一定要设置input_shape，否则后面会报错
    # 这一步使用applications模块自带的VGG16函数直接加载了模型和参数，作为我们自己模型的“身子”
    
    # 下面定义我们自己的分类器，作为我们自己模型的“头”
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:])) 
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(class_num, activation='softmax')) # 多分类问题
    
    top_model.load_weights(os.path.join(save_folder,'top_FC_model')) 
    # 上面定义了模型结构，此处要把训练好的参数加载进来，
    
    my_model = Model(inputs=base_model.input, outputs=top_model(base_model.output)) # 将“身子”和“头”组装到一起
    # my_model就是我们组装好的完整的模型，也已经加载了各自的weights
    
    # 普通的模型需要对所有层的weights进行训练调整，但是此处我们只调整VGG16的后面几个卷积层，所以前面的卷积层要冻结起来
    for layer in my_model.layers[:15]: # 15层之前都是不需训练的
        layer.trainable = False
        
    # 模型的配置
    my_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), # 使用一个非常小的lr来微调
                  metrics=['accuracy'])
    return my_model
```

**-------------------------------------输---------出--------------------------------**

start to fine-tune my model
Epoch 1/50
8/8 [==============================] - 124s 16s/step - loss: 0.0170 - acc: 0.9950 - val_loss: 0.2767 - val_acc: 0.9700
Epoch 2/50
8/8 [==============================] - 131s 16s/step - loss: 3.2684e-04 - acc: 1.0000 - val_loss: 0.2694 - val_acc: 0.9700
Epoch 3/50
8/8 [==============================] - 131s 16s/step - loss: 0.0175 - acc: 0.9950 - val_loss: 0.2593 - val_acc: 0.9700

。。。

Epoch 48/50
8/8 [==============================] - 132s 16s/step - loss: 0.0025 - acc: 1.0000 - val_loss: 0.2758 - val_acc: 0.9700
Epoch 49/50
8/8 [==============================] - 130s 16s/step - loss: 0.0080 - acc: 0.9950 - val_loss: 0.2922 - val_acc: 0.9700
Epoch 50/50
8/8 [==============================] - 131s 16s/step - loss: 4.7076e-04 - acc: 1.0000 - val_loss: 0.2875 - val_acc: 0.9700

**--------------------------------------------完-------------------------------------**

acc 和loss图为：

![](https://i.imgur.com/1zv6xIL.png)

可以看出，test acc的准确率一直在0.97附近，在训练开始和结束时的acc和loss变化不大，说明对于这个项目，微调的性能提升并不明显，这是因为本项目所用的数据集太小，容易发生过拟合，通过增大数据集可以解决这些问题。


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，使用Fine-Tune来解决多分类问题时，需要将loss改成categorical_crossentropy，并且使用SGD优化器，使用非常小的learning rate，防止lr太大使得前面的卷积层网络结构发生较大改变。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/DeepLearning)）上，欢迎下载。
