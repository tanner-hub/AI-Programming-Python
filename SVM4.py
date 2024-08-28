from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

print("___Shape___\n", df.shape, "\n")
print("___Head___\n", df.head(10), "\n")
print("___Tail___\n", df.tail(10), "\n")
print("___DecribeDF___\n", df.describe(), "\n")

# count NAN
print("___isnaSumSum___\n", df.isna().sum().sum(), "\n")
# drop NAN values
df = df.dropna()

print("___SizeOfGroupSpecies___\n", df.groupby('species').size(), "\n")
# Histograms
df.hist()
pyplot.savefig("Histogram.png")
# Scatterplot matrix
scatter_matrix(df)
pyplot.savefig("Scatterplot_Matrix.png")

X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for sepal length and width
p = clf.predict([[5.4, 3.2]])
print("___Prediction___\n", p, "\n")