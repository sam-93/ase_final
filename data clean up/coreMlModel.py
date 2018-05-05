import coremltools
from sklearn import datasets
from sklearn.model_selection import train_test_split

digit = datasets.load_digits()
x = digit.data
y = digit.target
ran = range(64)
features = [str(x) for x in ran]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)


model = coremltools.models.MLModel('digit_DecisionTreeClassifier.mlmodel')

prediction = model.predict(x_test)