# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds


def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    correct = 0.0 #有多少预测正确
    for i in range(len(y)):
        if y_pred[i] == int(y[i]): #如果预测正确
            correct += 1
    acc = correct / len(y)
    return acc
