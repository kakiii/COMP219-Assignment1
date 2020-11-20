from time import time
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split


# Student ID: 201521233


def main():
    digits = load_digits()
    num = len(digits.images)
    labels = list(digits.target_names)
    X = digits.images.reshape((num, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1313)

    print('the dataset I use is the digits dataset')
    print('the number of classes is: ', len(digits.target_names))
    print('target names are', labels)
    print('the number of data in each class is: ', end="")
    for i in range(len(np.bincount(y))):
        print(np.bincount(y)[i], end="")
        if len(np.bincount(y)) - i != 1:
            print(', ', end='')
        else:
            print()
    print('the number of instance is: ', num)
    print('the number of features of instance is', digits.data.shape[1])

    # use the built-in clf
    clf0 = KNeighborsClassifier(n_neighbors=3)
    clf0.fit(X_train, y_train)
    predicted0_train = clf0.predict(X_train)
    predicted0_test = clf0.predict(X_test)

    # use my own
    clf1 = myKNN(n_neighbors=3)
    clf1.fit(X_train, y_train)
    predicted1_train = clf1.predict(X_train)
    predicted1_test = clf1.predict(X_test)
    # print(predicted1)

    confusion_matrix_0_train = confusion_matrix(y_train, predicted0_train)
    confusion_matrix_0_test = confusion_matrix(y_test, predicted0_test)
    confusion_matrix_1_train = confusion_matrix(y_train, predicted1_train)
    confusion_matrix_1_test = confusion_matrix(y_test, predicted1_test)

    plot_confusion_matrix(confusion_matrix_0_test, labels, 'Built-in Classifier for test data')
    plot_confusion_matrix(confusion_matrix_0_train, labels, 'Built-in Classifier for train data')
    plot_confusion_matrix(confusion_matrix_1_test, labels, 'My Classifier for test data')
    plot_confusion_matrix(confusion_matrix_1_train, labels, 'My Classifier for train data')

    print('scores of built-in classifier are :', clf0.score(X_train, y_train), clf0.score(X_test, y_test))
    time_start = time()
    print('score of classifier written by myself is: ', clf1.score(X_train, y_train), clf1.score(X_test, y_test))
    time_end = time()
    print('time cost of my own classifier : %.3f' % (time_end - time_start))


class myKNN(object):
    train_data = []
    train_label = []
    k = 0
    score = 0

    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.train_data = X_train
        self.train_label = y_train

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            result.append(self.KNN(X_test[i]))
        return result

    def score(self, X_test, y_test):
        error_num = 0
        for i in range(X_test.shape[0]):
            result = self.KNN(X_test[i])
            # print('the classifier result is {},and the real num is {}.'.format(result, y_test[i]))
            if result != y_test[i]:
                error_num += 1
        return 1 - (error_num / X_test.shape[0])

    def KNN(self, index):
        data_set_size = self.train_data.shape[0]
        diff_mat = np.tile(index, (data_set_size, 1)) - self.train_data
        distances = ((diff_mat ** 2).sum(axis=1)) ** 0.5
        sorted_dist_indices = np.argsort(distances, axis=0)
        class_count = {}
        for i in range(self.k):
            neigh_label = self.train_label[sorted_dist_indices[i]]
            class_count[neigh_label] = class_count.get(neigh_label, 0) + 1
        sorted_class_count = sorted(class_count.items())
        return sorted_class_count[0][0]


def plot_confusion_matrix(data, labels, title):
    df_cm = pd.DataFrame(data, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    plt.title(title)
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__ == "__main__":
    main()
