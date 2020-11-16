import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from time import time


def handwriting_class_test():
    digits = load_digits()

    n_samples = len(digits.images)

    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123123)
    clf = myKNN(n_neighbors=3)
    clf.fit(X_train, y_train)
    time_start = time()
    print(clf.score(X_test, y_test))
    time_end = time()
    print('totally cost %.3f' % (time_end - time_start))
    # time_start = time()

    # for i in range(X_test.shape[0]):
    #     classifier_result = KNN(X_test[i], X_train, y_train, 3)
    #     print('the classifier result is {},and the real num is {}.'.format(classifier_result, y_test[i]))
    #     # if not match ,error_count += 1
    #     if classifier_result != y_test[i]:
    #         error_count += 1
    #
    # time_end = time()
    # print('total number of test data is ', X_test.shape[0])
    # print('score is %.3f' % (1 - (error_count / float(X_test.shape[0]))))
    # print('totally cost %.3f' % (time_end - time_start))


# def KNN(index, train_data, train_label, k):
#     data_set_size = train_data.shape[0]
#     diff_mat = np.tile(index, (data_set_size, 1)) - train_data
#     distances = ((diff_mat ** 2).sum(axis=1)) ** 0.5
#     sorted_dist_indices = np.argsort(distances, axis=0)
#     class_count = {}
#     for i in range(k):
#         neigh_label = train_label[sorted_dist_indices[i]]
#         class_count[neigh_label] = class_count.get(neigh_label, 0) + 1
#     sorted_class_count = sorted(class_count.items())
#     return sorted_class_count[0][0]


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
        print('length of test data is ', len(result))
        return result

    def score(self, X_test, y_test):
        error_num = 0
        print('length of test label is ', len(y_test))
        for i in range(X_test.shape[0]):
            result = self.KNN(X_test[i])
            print('the classifier result is {},and the real num is {}.'.format(result, y_test[i]))
            if result != y_test[i]:
                error_num += 1
        return format(1 - (error_num / X_test.shape[0]), '.3f')

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


if __name__ == "__main__":
    handwriting_class_test()
