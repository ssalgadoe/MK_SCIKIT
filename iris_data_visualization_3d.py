import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

data = load_iris()

X = data.data
y = data.target


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig,  elev=45, azim=130)

# pca = PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)

ax.scatter(X[:,0],X[:,1],X[:,2], c=y)

plt.show()
