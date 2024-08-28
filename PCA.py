from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

print("\n___X_VALUES___\n", X)
clf = PCA()
clf.fit(X, y)
print("\n___EXPLAINED_VARIANCE_RATIO___\n", clf.explained_variance_ratio_)
print("\n___SINGULAR_VALUES___\n", clf.singular_values_)