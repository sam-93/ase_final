
import coremltools


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
import time

#Iris Classification Cross Validation
def iris_cross():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    features = ['0', '1', '2', '3']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)


    clf = KNeighborsClassifier(n_neighbors=5)
    #clf = DecisionTreeClassifier(max_depth = 5)
    #clf = GaussianNB()
    #clf = SVC(gamma=2, C=1)
    #clf = MLPClassifier(alpha =1)

    clf.fit(x_train, y_train)
    core_mlmodel = coremltools.converters.sklearn.convert(clf, features, '4')
    core_mlmodel.save('iris_DecisionTreeClassifier.mlmodel')


    # y_pred = clf.predict(x_test)
    # print('accuracy: ',  accuracy_score(y_test, y_pred))


#Digit Classification Cross Validation
def digit_cross():
    digit = datasets.load_digits()
    x = digit.data
    y = digit.target
    ran = range(64)
    features = [str(x) for x in ran]
    print(features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

    #clf = KNeighborsClassifier(n_neighbors=5)
    #clf = DecisionTreeClassifier(max_depth = 16)
    #clf = GaussianNB()
    clf = SVC(4)
    #clf = MLPClassifier(alpha =1)

    clf.fit(x_train,y_train)
    core_mlmodel = coremltools.converters.sklearn.convert(clf, features, '64')
    core_mlmodel.save('digit_svc.mlmodel')

    # y_pred = clf.predict(x_test)
    # print('accuracy: ',  accuracy_score(y_test, y_pred))



#Digit Classification
def digit_data():
    digit = datasets.load_digits()
    x = digit.data
    y = digit.target

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
            start = time.time()
            clf.fit(x, y)
            end = time.time()
            finish = end - start
            sum += finish
            print(sum)
        average = sum / 5.0
        averageLearnTime.append(average)

    print(averageLearnTime)

#Wine Clustering Cross Validation
def wine_cross():
    wine = datasets.load_wine()
    x = wine.data
    y = wine.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

    clf = NearestNeighbors(n_neighbors=5)
    clf = KMeans(n_clusters=4, random_state=0)
    clf.fit(x_train)
    clf.labels_

    y_pred = clf.predict(x_test)
    print('accuracy: ', accuracy_score(y_test, y_pred))




#Wine Clustering
def wine_data():
    wine = datasets.load_wine()
    x = wine.data
    y = wine.target

    classifiers = [NearestNeighbors(n_neighbors=5),
                   KMeans(n_clusters=4, random_state=0)]
    averageLearnTime = []
    for clf in classifiers:
        sum = 0
        average = 0.0
        for i in range(0,5):
            start = time.time()
            clf.fit(x)
            end = time.time()
            finish = end - start
            sum += finish
            print(sum)
        average = sum / 5.0
        averageLearnTime.append(average)

    print(averageLearnTime)


#Boston Housing Regrssion Cross Validation
def house_cross():
    house = datasets.load_boston()
    x = house.data
    y = house.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)
    features = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    regr = linear_model.LinearRegression()
    #regr  = Ridge(.001)


    regr.fit(x_train,y_train)

    core_mlmodel = coremltools.converters.sklearn.convert(regr,features,'13')
    core_mlmodel.save('house_linearRegression.mlmodel')



    y_pred = regr .predict(x_test)
    print(' mean_squared_error: ',   mean_squared_error(y_test, y_pred))
house_cross()
#Housing Price Regression
def housing_data():
    house = datasets.load_boston()
    x = house.data
    y = house.target

    classifiers = [linear_model.LinearRegression(),
                   Ridge(.001)]
    averageLearnTime = []
    for clf in classifiers:
        sum = 0
        average = 0.0
        for i in range(0,5):
            start = time.time()
            clf.fit(x,y)
            end = time.time()
            finish = end - start
            sum += finish
            print(sum)
        average = sum / 5.0
        averageLearnTime.append(average)

    print(averageLearnTime)