import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X = cancer.data # All of the features
y = cancer.target # All of the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

clf = SVC()
clf.fit(X_train, y_train)

# Prediction
y_predict = clf.predict(X_test)

# Print Confusion Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer', 'predicted_healthy'])
print("___Display_Confusion___\n", confusion, "\n")
print("___Display_Classification_Report___\n", classification_report(y_test, y_predict), "\n")