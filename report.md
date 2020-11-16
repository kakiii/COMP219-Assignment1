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



2. I choose *K Nearest Neighbor* (KNN) as my algorithm, so it does not have a model as it is really **"lazy"** as it won't train the data input. Instead, it just stores it in and do nothing. As a result, there are only 2 files with 2 model files *missing*. I checked out some docs about built-in KNN classifier and then deployed it in my code.
3. I learnt it from lecture 10 and 11 and used **Euclidean distance** to computer the difference between training data and testing data. Similarly, I also wrote some methods like `predict()` , `score()` and `fit()`( actually it really does nothing apart from storing data).
4. 

