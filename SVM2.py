from sklearn import svm, datasets

iris = datasets.load_iris()

# Taking the first two lengths for sepal length and width
X = iris.data[:, :2]
y = iris.target #0: Setosa, 1: Versicolour, 2: Virginica

print (y)

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower given the pedal length
p = clf.predict([[5.4, 3.2]])

print(p)