# COMP219 Assignment1

> Name: Jiaqi Wang
>
> Student ID: 201521233

## Part 1

To run my code, please set the script path as `main.py`  and then click run.

This program depends on `sklearn` , including `datasets, neighbors, metrics, model_selection`. It also imports time to deploy a timer for my classifier.

For the file `my_clf.py`, it simply depends on `numpy`.

## Part 2

### Functionalities

1. Since it's a dataset returning a two-dimensional matrix, it is very easy to know that the number of instances is the number of rows and the number of features for each one is the number of columns. I print these two numbers out with some explanation.

2. I choose *K Nearest Neighbor* (KNN) as my algorithm, so it does not have a model as it is really **"lazy"** :  it won't train the data input. Instead, it just stores it in and do nothing. As a result, there are only 2 files(main.py and this report) with 2 model files *missing*. 
3. I learnt it from lecture 10 and 11 and used **Euclidean distance** to computer the difference between training data and testing data. Similarly, I also wrote some methods like `predict()` , `score()` and `fit()`.
4. Although KNN actually does nothing to the input data for training, it still can distinguish each of samples, comparing it with the whole bunch of data. I supposed it should be 100% for the accuracy rate, however it is not. I used `pandas` for the data structure of `frames` and use the `heatmap` from `seaborn` to draw the confusion matrices.
5. I don't have saved models, but users can change their input in the `main.py` file.

### Additional Requirements

1.  I don't have saved models, so it is possible that I cannot meet the requirement. Instead, you can simply start `main.py`
2. KNN does not train data, but you can use `fit()` to store data.

## Part 3

- I wrote this model into a class, called `myKNN()`. To initialize it, you need to enter the parameter K, which represents the number of the nearest neighbors. 

- Since KNN does not train data, the `fit(self,X_train,y_train)` method just takes in the data and label to be ''trained". 
- The `predict(self, X_test`)` method will take in a list of test data and returns the corresponding result. 
- The `score(self,X_test,y_test)` will evaluate the result with the answer, if not number of error will increase by one. It will return the accuracy ratio.
- The core algorithm, called `KNN(self,index)`, will take in one test data and calculate the Euclidean distance of the train data. Next, it will sort the indices of the distance and extract the first three numbers. It will then query the labels of these three and return the most possible answer.

