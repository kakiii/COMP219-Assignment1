import operator
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def KNN(index, train_data, train_label, k):
    data_set_size = train_data.shape[0]
    diff_mat = np.tile(index, (data_set_size, 1)) - train_data
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_ilabel = train_label[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda x:x[1])
    return sorted_class_count[0][0]


def handwriting_class_test():

    digits = load_digits()

    n_samples = len(digits.images)

    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    error_count = 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    for i in range(X_test.shape[0]):

        classifier_result = KNN(X_test[i], X_train,y_train, 1)
        print('the classifier result is {},and the real num is {}.'.format(classifier_result, y_train[i]))
        # if not match ,error_count += 1
        if (classifier_result != y_train[i]): error_count += 1
    print('error rate is {}.'.format(error_count / float(X_test.shape[0])))


if __name__ == '__main__':
    handwriting_class_test()
