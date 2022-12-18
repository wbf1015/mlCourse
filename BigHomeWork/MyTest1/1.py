#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File: lenet.py
# datetime: 2020/8/7 21:24
# software: PyCharm

import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import struct
import matplotlib.pyplot as plt


def data_fetch_preprocessing():
    train_image = open('train-images.idx3-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    y_test = np.fromfile(test_label,
                         dtype=np.uint8).reshape(10000, 1)
    # print(y_train[0])
    # 训练数据共有60000个
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784)

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784)
    # print(x_train.shape)
    # 可以通过这个函数观察图像
    # data=x_train[:,0].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train_label, x_test, y_test


class convolution_neural_network(nn.Module):
    def __init__(self):
        super(convolution_neural_network, self).__init__()

        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # 28x28x1-->24x24x6
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12x12x6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 8x8x16
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4x16
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    # 获取数据
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    x_train = x_train.reshape(60000, 1, 28, 28)
    # 建立模型实例
    LeNet = convolution_neural_network()
    # plt.imshow(x_train[2][0], cmap='Greys', interpolation=None)
    # plt.show()
    # 交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    optimizer = optim.Adam(params=LeNet.parameters(), lr=0.001)
    # epoch = 5
    for e in range(5):
        precision = 0
        for i in range(60000):
            prediction = LeNet(torch.tensor(x_train[i]).float().reshape(-1, 1, 28, 28))
            # print(prediction)
            # print(torch.from_numpy(y_train[i]).reshape(1,-1))
            # exit(-1)
            if torch.argmax(prediction) == y_train[i]:
                precision += 1
            loss = loss_function(prediction, torch.tensor([y_train[i]]).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        print('第%d轮迭代，loss=%.3f，准确率：%.3f' % (e, loss_list[-1],precision/60000))