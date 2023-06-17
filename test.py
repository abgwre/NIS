import random
import numpy as np
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from oodcls import OodCls
if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    mnist_load = tf.keras.models.load_model("./mnist_ood")

    i = random.randint(0, 10000)
    plt.subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(y_test[i])
    plt.show()

    pre = x_test[i].reshape(1, 28, 28, 1)

    res = mnist_load.predict(pre)

    print("预测的数字为：", np.argmax(res))
    print("实际数字为：", y_test[i])

    ood = OodCls()
    ood.__init__()

    for i in range(16):
        print(y_train[i])
    print(x_train.shape[0])
    out = ood.classify(x_train)     #由于mnist数据集过大，调用时手动设置classif中len为16，即选取前16个进行识别
    print(out)