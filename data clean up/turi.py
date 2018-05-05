
import turicreate as tc
from turicreate import nearest_neighbor_classifier
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import time
import numpy as np

#Iris Classification Cross Validation
def iris_cross():
    iris = datasets.load_iris()
    #features = iris.feature_names
    features = ['0','1','2','3']
    target = iris.target_names
    x = iris.data
    y = iris.target.astype('str')

    undata = np.column_stack((x,y))


    #data = tc.SFrame(map(SArray,undata))
    data = tc.SFrame(pd.DataFrame(undata))


    train_data, test_data = data.random_split(.7)

    # clf = tc.nearest_neighbor_classifier.create(train_data, target='4', features=features)

    clf = tc.decision_tree_classifier.create(train_data, target='4', max_depth = 5)
    clf.export_coreml('turi_decisionTree.mlmodel')
    # clf = tc.random_forest_classifier.create(train_data, target='4', max_depth=5, max_iterations=3)
    # clf.export_coreml('turi_iris_randomForestClass.mlmodel')



    # sum = 0
    # for i in range(0, 5):
    #     start = time.time()
    #     clf = tc.random_forest_classifier.create(train_data, target='4', max_depth=5, max_iterations=3)
    #     end = time.time()
    #     finish = end - start
    #     sum += finish
    # print(sum)
    # average = sum / 5.0
    #
    # print(average)
    #prediction = clf.predict(test_data)
    #results = clf.evaluate(test_data)
    print('done')



#Digit Classification Cross Validation
def digit_cross():
    digit = datasets.load_digits()
    #features = iris.feature_names
    features = ['0','1','2','3','4','5','6','7','8','9']
    target = digit.target_names
    x = digit.data
    y = digit.target.astype('str')

    undata = np.column_stack((x,y))


    #data = tc.SFrame(map(SArray,undata))
    data = tc.SFrame(pd.DataFrame(undata))


    train_data, test_data = data.random_split(.7)

    #clf = tc.nearest_neighbor_classifier.create(train_data, target='4', features=features)
    # clf = tc.decision_tree_classifier.create(train_data, target='4', max_depth = 5)
    # clf.export_coreml('turi_digit_decisionTreeClass.mlmodel')
    clf = tc.random_forest_classifier.create(train_data, target='4', max_depth=5, max_iterations=3)
    clf.export_coreml('turi_digit_randForestClass.mlmodel')


    #
    # sum = 0
    # for i in range(0, 5):
    #     start = time.time()
    #     clf = tc.decision_tree_classifier.create(train_data, target='4', max_depth = 5)
    #     end = time.time()
    #     finish = end - start
    #     sum += finish
    # print(sum)
    # average = sum / 5.0
    #
    # print(average)
    #prediction = clf.predict(test_data)
    #results = clf.evaluate(test_data)
    print('done')


def wine_cross():
    wine = datasets.load_wine()
    x = wine.data
    y = wine.target.astype('str')

    data = tc.SFrame(pd.DataFrame(x))


    train_data, test_data = data.random_split(.7)

    #clf = tc.kmeans.create(train_data, num_clusters=4)

    sum = 0
    for i in range(0, 5):
        start = time.time()
        clf = tc.kmeans.create(train_data, num_clusters=4)
        end = time.time()
        finish = end - start
        sum += finish
    print(sum)
    average = sum / 5.0

    print(average)

    print('done')


def house_cross():
    house = datasets.load_boston()
    #features = iris.feature_names
    features = ['0','1','2','3','4','5','6','7','8','9','10','11','12']

    x = house.data
    y = house.target.astype('float')


    undata = np.column_stack((x,y))
    data = tc.SFrame(pd.DataFrame(undata))


    train_data, test_data = data.random_split(.7)

    # Create a model.
    clf = tc.linear_regression.create(train_data, target='13',features=features)
    clf.export_coreml('turi_house_linearRegression.mlmodel')



    # sum = 0
    # for i in range(0, 5):
    #     start = time.time()
    #     clf = tc.decision_tree_regression.create(train_data, target='13',max_depth=5)
    #     end = time.time()
    #     finish = end - start
    #     sum += finish
    # print(sum)
    # average = sum / 5.0
    #
    # print(average)
    #prediction = clf.predict(test_data)
    #results = clf.evaluate(test_data)
    print('done')


house_cross()