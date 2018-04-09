import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()

X = data.data
y = data.target

fig, axes = plt.subplots(nrows=2, ncols=2)


pca = PCA(n_components=3)
pca.fit(X)
X_reduced = pca.transform(X)
print(X[0], X_reduced[0])


axes[0,0].scatter(X[:,0],X[:,1], c=y)
axes[0,1].scatter(X_reduced[:,0],X_reduced[:,1], c=y)

axes[1,0].scatter(X[:,0],X[:,2], c=y)
axes[1,1].scatter(X_reduced[:,0],X_reduced[:,2], c=y)

plt.show()