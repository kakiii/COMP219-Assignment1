import operator
import numpy as np


def KNN(test_data,train_data,train_label,k):
    #已知分类的数据集（训练集）的行数
    dataSetSize = train_data.shape[0]
    #求所有距离：tile函数将输入点拓展成与训练集相同维数的矩阵，并计算测试样本与每一个训练样本的距离
    all_distances = np.sqrt(np.sum(np.square(np.tile(test_data,(dataSetSize,1))-train_data),axis=1))
    #按all_distances中元素进行升序排序后得到其对应索引的列表
    sort_distance_index = all_distances.argsort()
    #选择距离最小的k个点
    all_predictive_value = {}
    for i in range(k):
        #返回最小距离的训练集的索引(预测值)
        predictive_value = train_label[sort_distance_index[i]]
        print('第',i+1,'次预测值',predictive_value)
        all_predictive_value[predictive_value] = all_predictive_value.get(predictive_value,0)+1
    #求众数：按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sorted_class_count = sorted(all_predictive_value.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_class_count[0][0]


def train_myKNN(X_train, Y_train, X_test, Y_test):
    error_Sum = 0
    test_Num = Y_train.shape[0]
    for i in range(test_Num):
        clf_result = KNN(X_test, X_train, Y_train, 3)
        print('the classifier result is {},and the real num is {}.'.format(clf_result, Y_test[i]))
        if clf_result != Y_test: error_Sum += 1
    print('error rate is ' + error_Sum / test_Num)
