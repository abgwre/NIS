
import numpy as np
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

def model():



    # 加载mnist数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 加载fashionmnist数据集作为ood类
    (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
    train_label=np.zeros(60000)

    # 调整fashionmnist类标签为10作为ood
    for i in range(60000):
        train_label[i] = 10

    # 将mnist数据集和fashionmnist数据集的特征合并
    x_train = np.concatenate((x_train, train_images), axis=0)

    # 将mnist数据集和fashionmnist数据集的标签合并
    y_train = np.concatenate((y_train, train_label), axis=0)


    # 数据预处理
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = tf.keras.utils.to_categorical(y_train, 11)
    y_test = tf.keras.utils.to_categorical(y_test, 11)


    # 构建CNN模型
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))

    # 编译模型
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose = 1)  # 每个batch包含128个样本，训练10轮。输出进度条

    # 模型保存为(mnist_ood)
    model.save("mnist")




if __name__ == '__main__':
    model()

