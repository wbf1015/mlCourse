# coding=utf-8
from concurrent.futures import thread
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from data_process import load_mnist, load_data

from ex1.ex1.evaluate import predict, cal_accuracy


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and
    # the objective function value of every iteration and update the theta
    m = len(x)
    f = list()
    ff = list()
    lam = 0.00  # 正则项
    data = x.T  # 矩阵转置
    print(theta.shape)
    mnist_dir = "mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    train_images, train_labels, test_images, test_labels = \
        load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)

    for i in range(iters):
        out = np.dot(theta, data)  # 做一次矩阵乘法

        output_exp = np.exp(out)
        exp_sum = output_exp.sum(axis=0)  # 对矩阵做e的运算
        y_hat = (output_exp / exp_sum)

        batch_size = y.shape[1]
        y_hatlog = np.log(y_hat)  # 把e取下来
        l_sum = 0.0
        for i in range(batch_size):
            l_sum += np.dot(y_hatlog[:, i].T, y[:, i])
        train_loss = -(1.0 / batch_size) * l_sum  # 计算损失函数

        print(train_loss)  # 输出损失函数
        f.append(train_loss)

        batch_size = y.shape[1]
        g = -(1.0 / batch_size) * np.dot((y - y_hat), data.T) + lam * theta  # 计算梯度
        theta = theta - alpha * g  # 梯度下降

        y_predict = predict(test_images, theta)
        accuracy = cal_accuracy(y_predict, test_labels)
        ff.append(accuracy)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(iters), f)
    plt.show()

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(iters), ff)
    plt.show()

    return theta
