import numpy as np
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OodCls:
    def __init__(self):
        self.mnist_load = tf.keras.models.load_model("./mnist_ood")


    def classify(self, imgs : torch.Tensor) -> torch.Tensor:

        len = imgs.shape[0]
        preds = torch.empty(len)

        for i in range(len):

            pre = imgs[i].reshape(1, 28, 28, 1)
            res = self.mnist_load.predict(pre)
            preds[i] = np.argmax(res)

        return preds

