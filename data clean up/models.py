import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
import time

#Iris Classification
def iris_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

    #clf = KNeighborsClassifier(n_neighbors=5)
    #clf = DecisionTreeClassifier(max_depth = 5)
    #clf = GaussianNB()
    #clf = SVC(gamma=2, C=1)
    #clf = MLPClassifier(alpha =1)

    classifiers = [KNeighborsClassifier(n_neighbors=5),
                   DecisionTreeClassifier(max_depth = 5),
                   GaussianNB(),
                   SVC(gamma=2, C=1),
                   MLPClassifier(alpha=1)]
    averageLearnTime = []
    for clf in classifiers:
        sum = 0
        average = 0.0
        for i in range(0,5):
            print(i)
            start = time.time()
            #clf.fit(x_train,y_train)
            clf.fit(x, y)
            end = time.time()
            finish = end - start
            sum += finish
            print(sum)
        average = sum / 5.0
        averageLearnTime.append(average)



    #y_pred = clf.predict(x_test)

    print(averageLearnTime)
    #print('accuracy: ',  accuracy_score(y_test, y_pred))


iris_data()
