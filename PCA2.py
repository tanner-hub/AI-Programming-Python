import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

# Load Iris Data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Plot Iris Data
f = plt.figure(1)
plt.scatter(X[:,0], X[:,1], c = y)
plt.xlabel('sepals length')
plt.ylabel('sepals width')
plt.title('Original Data')
f.show()

# Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# Plot Data
g = plt.figure(2)
#y = np.choose(y, [1, 2, 0]).astype(float)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')
g.savefig("PCA/ScatterPlot.png")