from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib, time

X, y = load_iris(return_X_y=True)

print("\n___X_VALUE___\n", X)

clf = GaussianNB()
clf.fit(X, y)

p = clf.predict([[5.0, 3.4, 1.5, 0.4]])

print("\n___P_VALUE___\n", p)

joblib.dump(clf, 'model.pkl')

# Read from file and make a prediction from the tested data - TMU
print("\nSaving to file: Success.")
time.sleep(5)
print("Now Making Prediction...\n")

clf2 = joblib.load('model.pkl')
p = clf2.predict([[5.4, 3.2, 1,5, 0.4]])

print("\n___Prediction___\n", p)